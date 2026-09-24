"""semantic commit M1 — 토큰 registry(Phase1→Phase2→SEM 순서, <SEM_END> 151723; 턴 종료 = Phase 2 <EOT> 151722, v0.2 <TURN_END> 151724 는 읽기 전용),
add_semantic_tokens(새 행만 초기화·idempotent·차단 목록·mono 턴 <EOT> 선택),
soft_ce 이벤트 가중·위치 덮어쓰기·SEM 통계, forward pos_weight 연결, load_tokenizer 재로드, trainer 로그. 모델·tokenizer 다운로드 없이 CPU 에서 수 초."""
import json, types
import pytest, torch, torch.nn as nn, torch.nn.functional as F
from vapasr.data.semcommit_tokens import (SEM_SPECIALS, SEM_REGISTRY, ALL_SEM_SPECIALS, TURN_TOKEN, LEGACY_TURN_TOKEN, add_semantic_specials, assert_frozen_semantic,
                                          sem_ids_of, init_rows_mean_noise, semantic_blocked_ids, new_special_rows)
from vapasr.data.dialogue_tokens import FROZEN_REGISTRY, PHASE2_SPECIALS
from vapasr.uslm.interleave_data import SPECIAL_TOKENS as PHASE1_SPECIALS, add_specials
from vapasr.hf.p2_losses import soft_ce

V, D, BASE = 151936, 8, 151705
SEM, TURN, LEGACY_TURN = 151723, 151722, 151724                                           # TURN = <EOT>

class FakeTok:
    """HF add_tokens 흉내: 처음 보는 토큰만 순서대로 len(tok) id 를 받는다(Qwen3-ASR 기본 길이 151705)."""
    def __init__(self, base: int = BASE): self.base, self.vocab = base, {}
    def add_tokens(self, toks, special_tokens=False):
        n = 0
        for t in toks:
            if t not in self.vocab: self.vocab[t] = self.base + len(self.vocab); n += 1
        return n
    def convert_tokens_to_ids(self, t): return self.vocab.get(t, 3)    # unk
    def __len__(self): return self.base + len(self.vocab)

# ── registry
def test_registry_order_and_ids():
    tok = FakeTok(); ids = add_semantic_specials(tok)
    assert list(ids) == PHASE1_SPECIALS + PHASE2_SPECIALS + SEM_SPECIALS and len(tok) == BASE + 19 and "<TURN_END>" not in tok.vocab   # <TURN_END> 폐기
    assert {k: ids[k] for k in FROZEN_REGISTRY} == FROZEN_REGISTRY and {k: ids[k] for k in SEM_SPECIALS} == SEM_REGISTRY == {"<SEM_END>": SEM} and ids[TURN_TOKEN] == TURN
    assert_frozen_semantic(ids)
    assert sem_ids_of(SEM_REGISTRY) == (SEM, None) and sem_ids_of({"<SEM_END>": SEM, TURN_TOKEN: TURN}) == (SEM, TURN)   # 턴 = registry 가 학습 이벤트로 가진 <EOT>
    assert sem_ids_of({"<SEM_END>": SEM, LEGACY_TURN_TOKEN: LEGACY_TURN}) == (SEM, LEGACY_TURN)                     # v0.2 체크포인트
    assert add_semantic_specials(tok) == ids and len(tok) == BASE + 19                     # 반복 호출은 no-op
    t1 = FakeTok(); add_specials(t1); assert add_semantic_specials(t1) == ids                # Phase 1 tokenizer 에 이어 붙여도 같은 id
    t2 = FakeTok(); add_specials(t2); t2.add_tokens(SEM_SPECIALS, special_tokens=True)       # 함정: Phase 1 뒤에 SEM 만 붙이면 <SPK_3>/<SPK_4> id 를 뺏는다
    assert t2.convert_tokens_to_ids("<SEM_END>") == FROZEN_REGISTRY["<SPK_3>"]
    with pytest.raises(AssertionError): assert_frozen_semantic(add_semantic_specials(t2))
    assert ALL_SEM_SPECIALS[-1:] == SEM_SPECIALS

def test_init_rows_only_new_and_blocked_helper():
    torch.manual_seed(0); W = torch.randn(40, 4); W0 = W.clone()
    rows = init_rows_mean_noise(W, [33, 31, 33], upto=30, seed=5); assert rows == [31, 33]
    keep = [r for r in range(40) if r not in rows]; assert torch.equal(W[keep], W0[keep])
    mu = W0[:30].mean(0); assert all((W[r] - mu).abs().max() < 0.2 for r in rows) and not torch.equal(W[31], W[33])
    W2 = W0.clone(); init_rows_mean_noise(W2, [31, 33], upto=30, seed=5); assert torch.equal(W, W2)   # 같은 seed → 결정적
    assert init_rows_mean_noise(W2, [], upto=30) == [] and torch.equal(W, W2)
    ids = add_semantic_specials(FakeTok())
    b0 = semantic_blocked_ids([5, 151706, SEM], ids, lanes=0); assert SEM not in b0 and set(range(151717, 151723)) <= set(b0) and {5, 151706} <= set(b0)   # <EOT> 도 예약 → 차단
    bt = semantic_blocked_ids([5, 151706, SEM, TURN], ids, lanes=0, turn=True); assert SEM not in bt and TURN not in bt and set(range(151717, 151722)) <= set(bt)   # mono 턴 = <EOT> 차단 해제
    b2 = semantic_blocked_ids([5, 151706], ids, lanes=6); assert b2 == [5, 151706]
    assert semantic_blocked_ids([5], ids, lanes=0, block_reserved_phase2=False) == [5]
    assert new_special_rows(ids, {k: ids[k] for k in PHASE1_SPECIALS}) == list(range(151717, 151724))
    with pytest.raises(AssertionError): new_special_rows(ids, {"<NEXT_AUDIO>": 999})

# ── 작은 가짜 모델 (Qwen 없이 VapAsrForStreamingASR 를 만든다: thinker 는 embedding + Linear + tied lm_head)
class _Inner(nn.Module):
    def __init__(s): super().__init__(); s.embed_tokens = nn.Embedding(V, D); s.lin = nn.Linear(D, D)
    def forward(s, inputs_embeds=None, attention_mask=None, **_): return types.SimpleNamespace(last_hidden_state=torch.tanh(s.lin(inputs_embeds)))
class _Thinker(nn.Module):
    def __init__(s): super().__init__(); s.model = _Inner(); s.lm_head = nn.Linear(D, V, bias=False)
    def get_input_embeddings(s): return s.model.embed_tokens
    def set_input_embeddings(s, v): s.model.embed_tokens = v

def tiny_model(**cfg_kw):
    from vapasr.hf import VapAsrConfig, VapAsrForStreamingASR
    torch.manual_seed(0); tok = FakeTok(); sp = add_specials(tok)
    blocked = [sp["<EMPTY_AUDIO>"], sp["<SPK_A>"], sp["<SPK_B>"], 7] + [v for k, v in sp.items() if k.startswith("<DELAY_")]
    cfg = VapAsrConfig(adapter_d_in=4, adapter_d_out=D, adapter_hidden=16, sp_ids=sp, special_tokens=list(PHASE1_SPECIALS), blocked_ids=blocked, act_hidden=4, **cfg_kw)
    return VapAsrForStreamingASR(cfg, thinker=_Thinker()), tok

def test_add_semantic_tokens_mono_new_rows_idempotent_blocked():
    m, tok = tiny_model(); W = m.get_input_embeddings().weight; W0 = W.detach().clone()
    assert m.thinker.lm_head.weight is W                                                     # tied
    ids = m.add_semantic_tokens(tok, seed=0)
    new = list(range(151717, 151724)); keep = torch.ones(V, dtype=torch.bool); keep[new] = False
    assert torch.equal(W.detach()[keep], W0[keep])                                          # Phase 1 행·기본 어휘는 그대로
    mu = W0[:BASE].mean(0); assert all((W.detach()[r] - mu).abs().max() < 0.2 for r in new) and not torch.equal(W.detach()[SEM], W0[SEM])
    c = m.config; assert c.sem_registry == SEM_REGISTRY and c.sp_ids == ids and m.sp_ids == ids and c.lanes == 0 and c.phase2_registry == {}
    assert c.sem_reserved == PHASE2_SPECIALS                                                 # mono: Phase 2 이름은 예약만(→ Phase 2 확장 때 재초기화 대상)
    assert c.special_tokens == PHASE1_SPECIALS + PHASE2_SPECIALS + SEM_SPECIALS
    assert set(range(151717, 151723)) <= set(c.blocked_ids) and SEM not in c.blocked_ids and TURN in c.blocked_ids and 7 in c.blocked_ids   # SEM 만: <EOT> 는 예약·차단
    assert m.blocked.tolist() == c.blocked_ids
    snap = (W.detach().clone(), json.dumps(c.to_dict(), sort_keys=True, default=str))
    assert m.add_semantic_tokens(tok, seed=123) == ids                                       # 다른 seed 로 다시 불러도 아무 행도 바뀌지 않는다
    assert torch.equal(W.detach(), snap[0]) and json.dumps(c.to_dict(), sort_keys=True, default=str) == snap[1] and m.blocked.tolist() == c.blocked_ids
    res = list(range(151717, 151723))
    with torch.no_grad(): W[res] -= 3.0                                                      # 예약 행이 학습 중 full-vocab softmax 음성으로 밀려난 상황 흉내
    Wt = W.detach().clone(); m.add_phase2_tokens(tok, seed=0)                                # mono sem → Phase 2 확장: 예약 행 재초기화 + 차단 해제
    assert not (set(res) & set(m.config.blocked_ids)) and m.config.sem_reserved == [] and m.config.lanes == 6
    kr = torch.ones(V, dtype=torch.bool); kr[res] = False; assert torch.equal(W.detach()[kr], Wt[kr])      # SEM·Phase 1·기본 어휘 행은 그대로
    assert all((W.detach()[r] - mu).abs().max() < 0.2 for r in res)                          # 예약 행은 다시 평균+잡음
    m2, tok2 = tiny_model(); m2.add_phase2_tokens(tok2, seed=0); assert torch.equal(W.detach()[res], m2.get_input_embeddings().weight.detach()[res])   # E2 → Phase 2 경로와 같은 값
    W3 = W.detach().clone(); m.add_semantic_tokens(tok); assert torch.equal(W.detach(), W3) and m.config.sem_reserved == []   # Phase 2 뒤 재호출: 예약 표시 없음·행 불변

def test_add_semantic_tokens_on_phase2_model_keeps_phase2_rows():
    m, tok = tiny_model(); m.add_phase2_tokens(tok); W = m.get_input_embeddings().weight; W0 = W.detach().clone(); b0 = list(m.config.blocked_ids)
    ids = m.add_semantic_tokens(tok)
    keep = torch.ones(V, dtype=torch.bool); keep[[SEM]] = False; assert torch.equal(W.detach()[keep], W0[keep])
    assert m.config.lanes == 6 and m.config.blocked_ids == b0 and m.config.sem_registry == SEM_REGISTRY and m.config.phase2_registry["<EOT>"] == ids["<EOT>"] and m.config.sem_reserved == []
    with pytest.raises(AssertionError, match="mono 전용"): m.add_semantic_tokens(tok, turn=True)            # Phase 2 는 <EOT> 를 이미 phase2_registry 로 학습

def test_add_semantic_tokens_mono_turn_uses_eot():
    m, tok = tiny_model(); ids = m.add_semantic_tokens(tok, turn=True); c = m.config
    assert c.sem_registry == {"<SEM_END>": SEM, TURN_TOKEN: TURN} and ids[TURN_TOKEN] == TURN and "<TURN_END>" not in tok.vocab
    assert TURN not in c.blocked_ids and SEM not in c.blocked_ids and set(range(151717, 151722)) <= set(c.blocked_ids) and TURN_TOKEN not in c.sem_reserved

def test_add_semantic_tokens_rejects_wrong_tokenizer():
    m, _ = tiny_model()
    with pytest.raises(AssertionError): m.add_semantic_tokens(FakeTok(base=BASE + 1))                  # 동결 id 불일치
    m2, _ = tiny_model()
    with pytest.raises(AssertionError): m2.add_semantic_tokens(FakeTok(base=BASE + 1), check_frozen=False)   # config.sp_ids(Phase 1) 와 불일치

# ── soft_ce
def _manual(logits, tgt, wt): ce = F.cross_entropy(logits, tgt, reduction="none"); return (ce * wt).sum() / wt.sum(), ce

def test_soft_ce_event_weights_override_and_stats():
    torch.manual_seed(3); NEXT, S, T = 11, 9, 10; tgt = torch.tensor([3, NEXT, S, NEXT, T, 4, S, NEXT]); logits = 0.1 * torch.randn(8, 12)
    for i in (1, 3, 4, 5, 7): logits[i, tgt[i]] += 5.0                                     # 나머지는 정답이 argmax
    logits[0, S] += 5.0; logits[2, S] += 5.0; logits[6, 1] += 5.0                          # pos 0: 비-SEM 타깃인데 argmax=SEM(오확정), pos 2: SEM 적중, pos 6: SEM 놓침
    po = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])                             # pos 3(NEXT) = hard negative → 1.0
    loss, st = soft_ce(logits, tgt, None, None, NEXT, 0.3, evt_weights={S: 2.0, T: 1.5}, pos_override=po, sem_id=S, turn_id=T)
    wt = torch.tensor([1.0, 0.3, 2.0, 1.0, 1.5, 1.0, 2.0, 0.3]); exp, ce = _manual(logits, tgt, wt)
    assert torch.allclose(loss, exp)
    assert st["n_sem"] == 2 and st["loss_sem"] == pytest.approx(ce[[2, 6]].mean().item(), rel=1e-5) and st["top1_sem"] == 0.5
    assert st["sem_fp"] == pytest.approx(1 / 6) and st["n_turn"] == 1 and st["loss_turn"] == pytest.approx(ce[4].item(), rel=1e-5) and st["n_override"] == 1
    tx = torch.tensor([0, 5]); assert st["loss_text"] == pytest.approx(ce[tx].mean().item(), rel=1e-5)          # text 통계에서 SEM/TURN/NEXT 제외
    assert st["top1_text"] == 0.5
    assert st["loss_next"] == pytest.approx(ce[[1, 3, 7]].mean().item(), rel=1e-5)
    po2 = torch.zeros(8); po2[2] = 0.5; loss2, _ = soft_ce(logits, tgt, None, None, NEXT, 0.3, evt_weights={S: 2.0, T: 1.5}, pos_override=po2, sem_id=S, turn_id=T)
    wt2 = torch.tensor([1.0, 0.3, 0.5, 0.3, 1.5, 1.0, 2.0, 0.3]); assert torch.allclose(loss2, _manual(logits, tgt, wt2)[0])   # 덮어쓰기가 evt 가중보다 우선

def test_soft_ce_defaults_unchanged():
    torch.manual_seed(4); NEXT, EOT = 11, 10; tgt = torch.tensor([3, NEXT, EOT, 9, 4]); logits = torch.randn(5, 12)
    loss, st = soft_ce(logits, tgt, None, None, NEXT, 0.3, eot_id=EOT, eot_weight=2.0)
    exp, ce = _manual(logits, tgt, torch.tensor([1.0, 0.3, 2.0, 1.0, 1.0])); assert torch.allclose(loss, exp)
    assert set(st) == {"loss_next", "loss_text", "loss_eot", "top1_text", "n_soft", "n_eot", "loss_tag", "top1_tag", "n_tag"}
    assert st["loss_text"] == pytest.approx(ce[[0, 3, 4]].mean().item(), rel=1e-5)          # 이벤트 id 를 안 주면 9 도 text
    l2, st2 = soft_ce(logits, tgt, None, None, NEXT, 0.3, eot_id=EOT, eot_weight=2.0, evt_weights=None, pos_override=torch.zeros(5))
    assert torch.allclose(l2, loss) and st2["n_override"] == 0
    l3, st3 = soft_ce(logits, tgt, None, None, NEXT, 0.3, eot_id=EOT, eot_weight=2.0, sem_id=9)   # 통계만(가중 1.0) → 손실 동일, text 에서는 빠진다
    assert torch.allclose(l3, loss) and st3["n_sem"] == 1 and st3["loss_text"] == pytest.approx(ce[[0, 4]].mean().item(), rel=1e-5)

# ── forward 연결
def _batch(sp):
    NEXT = sp["<NEXT_AUDIO>"]; L = 9
    ids = torch.tensor([[1, 0, 50, 60, SEM, NEXT, 0, 70, TURN], [1, 0, 80, NEXT, 0, 90, NEXT, 0, 0]])          # TURN = <EOT>(mono 턴 종료)
    is_audio = torch.tensor([[0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0]], dtype=torch.bool)
    chunk_of = torch.tensor([[-1, 0, -1, -1, -1, -1, 1, -1, -1], [-1, 0, -1, -1, 1, -1, -1, 2, -1]])
    labels = ids.clone(); labels[is_audio] = -100; labels[:, 0] = -100; labels[1, 8] = -100
    labels[1, 3] = -100                                                                       # 'B' 등급 결정 위치 → -100
    pw = torch.zeros(2, L); pw[0, 5] = 1.0                                                    # 'N' 등급: NEXT 결정 위치 가중 1.0
    return dict(ids=ids, is_audio=is_audio, chunk_of=chunk_of, labels=labels, mask=torch.ones(2, L, dtype=torch.long), feats=torch.randn(2, 1, 3, 4)), pw

def test_forward_sem_weights_and_pos_weight():
    m, tok = tiny_model(sem_weight=2.0, turn_weight=1.5); x, pw = _batch(m.config.sp_ids)
    with pytest.raises(AssertionError, match="sem_registry"): m(**x)                         # SEM 라벨인데 add_semantic_tokens 를 빠뜨림 → 평문 학습 대신 실패
    xp = dict(x); xp["labels"] = x["labels"].clone(); xp["labels"][0, 4] = -100; xp["labels"][0, 8] = -100
    base = m(**xp); assert base.loss_sem is None and base.n_sem is None and base.sem_fp is None             # sem_registry 없음 + SEM/TURN 라벨 없음 → 기존 경로
    m.add_semantic_tokens(tok, turn=True); m.eval(); NEXT = m.config.sp_ids["<NEXT_AUDIO>"]
    out = m(**x, pos_weight=pw, next_weight=0.3)
    with torch.no_grad():
        h = m.thinker.model(inputs_embeds=m.build(x["feats"], x["ids"], x["is_audio"], x["chunk_of"])).last_hidden_state
        tgt = x["labels"][:, 1:]; sel = tgt != -100; t = tgt[sel]; lg = m.thinker.lm_head(h[:, :-1][sel]).float()
        wt = torch.ones(len(t)); wt[t == NEXT] = 0.3; wt[t == SEM] = 2.0; wt[t == TURN] = 1.5; p = pw[:, 1:][sel]; wt = torch.where(p > 0, p, wt)
        ce = F.cross_entropy(lg, t, reduction="none"); exp = (ce * wt).sum() / wt.sum()
    assert torch.allclose(out.loss, exp, atol=1e-6) and out.n_sem == 1 and out.n_turn == 1 and out.n_labels == int(sel.sum())
    assert out.loss_sem == pytest.approx(ce[t == SEM].item(), rel=1e-5) and out.loss_turn == pytest.approx(ce[t == TURN].item(), rel=1e-5)
    assert out.top1_sem in (0.0, 1.0) and 0.0 <= out.sem_fp <= 1.0
    out2 = m(**x, next_weight=0.3); assert not torch.allclose(out2.loss, out.loss)            # pos_weight 가 실제로 반영
    x3 = dict(x); x3["labels"] = x["labels"].clone(); x3["labels"][0, 4] = -100; x3["labels"][0, 8] = -100
    out3 = m(**x3, next_weight=0.3); assert out3.n_sem == 0 and out3.loss_sem is None and out3.top1_sem is None and out3.loss_turn is None and out3.sem_fp is not None

# ── config · load_tokenizer
def test_config_defaults_and_roundtrip(tmp_path):
    from vapasr.hf import VapAsrConfig
    old = VapAsrConfig.from_dict({"sp_ids": {"<NEXT_AUDIO>": BASE}, "lanes": 0}); assert old.sem_registry == {} and old.sem_weight == 1.0 and old.turn_weight == 1.0 and old.sem_reserved == []
    c = VapAsrConfig(sem_registry=dict(SEM_REGISTRY), sem_weight=3.0, turn_weight=0.5, sem_reserved=list(PHASE2_SPECIALS)); c.save_pretrained(tmp_path)
    r = VapAsrConfig.from_pretrained(tmp_path); assert r.sem_registry == SEM_REGISTRY and r.sem_weight == 3.0 and r.turn_weight == 0.5 and r.sem_reserved == PHASE2_SPECIALS

def _write_cfg(path, **cfg): path.mkdir(parents=True, exist_ok=True); (path / "config.json").write_text(json.dumps(dict(model_type="vapasr", thinker_name_or_path="fake-qwen", **cfg)))

def test_load_tokenizer_semantic(tmp_path, monkeypatch):
    import transformers
    from vapasr.hf import load_tokenizer
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTok())
    ids = add_semantic_specials(FakeTok())
    _write_cfg(tmp_path / "sem", sp_ids=ids, sem_registry=dict(SEM_REGISTRY)); tok = load_tokenizer(str(tmp_path / "sem"))
    assert tok.convert_tokens_to_ids("<SEM_END>") == SEM and "<TURN_END>" not in tok.vocab and len(tok) == BASE + 19
    _write_cfg(tmp_path / "semturn", sp_ids=ids, sem_registry={"<SEM_END>": SEM, TURN_TOKEN: TURN}); assert load_tokenizer(str(tmp_path / "semturn")).convert_tokens_to_ids(TURN_TOKEN) == TURN
    _write_cfg(tmp_path / "v02", sp_ids=ids, sem_registry={"<SEM_END>": SEM, LEGACY_TURN_TOKEN: LEGACY_TURN})     # v0.2 체크포인트(tokenizer 파일 없음) → <TURN_END> 를 다시 붙인다
    t02 = load_tokenizer(str(tmp_path / "v02")); assert t02.convert_tokens_to_ids(LEGACY_TURN_TOKEN) == LEGACY_TURN and len(t02) == BASE + 20
    p2 = {n: FROZEN_REGISTRY[n] for n in ["<SPK_A>", "<SPK_B>", "<SPK_3>", "<SPK_4>", "<SPK_5>", "<SPK_6>", "<ONSET>", "<EOT>"]}
    _write_cfg(tmp_path / "p2sem", sp_ids=ids, phase2_registry=p2, sem_registry=dict(SEM_REGISTRY)); assert load_tokenizer(str(tmp_path / "p2sem")).convert_tokens_to_ids("<SEM_END>") == SEM
    _write_cfg(tmp_path / "p1", sp_ids={k: ids[k] for k in PHASE1_SPECIALS}); t1 = load_tokenizer(str(tmp_path / "p1"))   # 기존 Phase 1 산출물은 그대로
    assert len(t1) == BASE + 12 and "<SEM_END>" not in t1.vocab
    _write_cfg(tmp_path / "bad", sp_ids=ids, sem_registry={"<SEM_END>": 151717})
    with pytest.raises(AssertionError): load_tokenizer(str(tmp_path / "bad"))

# ── trainer compute_loss/log (transformers.Trainer 를 import 할 수 있는 환경에서만)
def _trainer_cls():
    try:
        from vapasr.hf.trainer import VapAsrTrainer
        return VapAsrTrainer
    except Exception as e: pytest.skip(f"transformers.Trainer import 불가: {type(e).__name__}")

def test_trainer_passes_pos_weight_and_logs_sem(monkeypatch):
    VapAsrTrainer = _trainer_cls(); import transformers
    from vapasr.hf import VapAsrOutput
    seen, logged = [], []
    outs = iter([VapAsrOutput(loss=torch.tensor(1.0), loss_next=0.5, loss_text=1.0, top1_text=0.5, n_labels=10, loss_sem=2.0, top1_sem=1.0, n_sem=2, sem_fp=0.1, n_turn=0),
                 VapAsrOutput(loss=torch.tensor(1.0), loss_next=0.7, loss_text=1.2, top1_text=0.7, n_labels=12, loss_sem=None, top1_sem=None, n_sem=0, sem_fp=0.3, n_turn=1, loss_turn=0.4)])
    def model(**kw): seen.append(sorted(kw)); return next(outs)
    model.config = types.SimpleNamespace(next_weight=0.3, next_weight_ko=0.15)
    tr = VapAsrTrainer.__new__(VapAsrTrainer); tr.model = model; tr._parts = {}; tr._n_parts = 0
    tr.args = types.SimpleNamespace(process_index=0); tr.state = types.SimpleNamespace(global_step=2, max_steps=10); tr.optimizer = None
    inp = dict(ids=torch.zeros(1, 3), labels=torch.zeros(1, 3), pos_weight=torch.zeros(1, 3), lang=["English"])
    tr.compute_loss(model, inp); tr.compute_loss(model, {k: v for k, v in inp.items() if k != "pos_weight"})
    assert "pos_weight" in seen[0] and "pos_weight" not in seen[1]
    monkeypatch.setattr(transformers.Trainer, "log", lambda self, logs, *a: logged.append(dict(logs)))
    tr.log({"loss": 1.0}); lg = logged[-1]
    assert lg["loss_sem"] == 2.0 and lg["top1_sem"] == 1.0 and lg["sem_fp"] == pytest.approx(0.2) and lg["loss_turn"] == 0.4 and lg["sem_per_step"] == 1.0 and lg["loss_text"] == pytest.approx(1.1)
    outs = iter([VapAsrOutput(loss=torch.tensor(1.0), loss_next=0.5, loss_text=1.0, top1_text=0.5, n_labels=10)])     # sem 없는 기존 모델 → sem 로그 없음
    tr.compute_loss(model, {k: v for k, v in inp.items() if k != "pos_weight"}); tr.log({"loss": 1.0})
    assert not ({"loss_sem", "sem_fp", "loss_turn", "top1_sem", "sem_per_step"} & set(logged[-1]))
