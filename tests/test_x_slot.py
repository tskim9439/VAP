"""E14 비귀속 칸(2026-11-03 — cst_dialogue 모듈 설명 x_slot): 학습 게이트의 N3D 놓침 흉내(sim_miss)와 닫힌 슬롯 글자 → X 칸(route_closed — 디코더와 같은 열림),
시퀀스 검사(block_problems), 디코더(BlockState 가 X 칸 학습 시퀀스를 그대로 되짚는다 · 추측 prefill 이 한 토큰씩과 같다), 해석(emitted_from_blocks·parse_emitted X),
X 단어 귀속(attribute_x·merge_x), 확률 활동(n3d_act n3d_soft). CPU."""
import importlib.util, pathlib, random
from types import SimpleNamespace
import numpy as np, pytest, torch
pytest.importorskip("soundfile")
from vapasr.data.cst_dialogue import CstDialogueDataset, slot_activity, X_TOKEN
from vapasr.data.cst_eval import emitted_from_blocks, parse_emitted, attribute_x, merge_x, CHUNK_S
from vapasr.hf.batch_decode import BlockState, block_stops

_HERE = pathlib.Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", _HERE / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


_SB = _load("test_slot_blocks")
KW = dict(delays=(4, 10), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, slot_blocks=True, declare=4)
MISS = dict(miss_drop=0.4, miss_short=0.6, miss_late=0.5)


def _tok(trans=False):
    t = _SB._tok(); t.add_tokens([X_TOKEN] + (["<TRANS>"] if trans else [])); return t


def _trans(d):
    return {f"{d.conv_id}|{u.speaker}|{u.utt_id}": {"German": (np.array([60, 61]), np.array([u.start + 1.0, u.end]))} for u in d.utterances}


def _text_tokens(s):
    """창의 전사 글자 원소 수(사건·번역 빼고) — X 칸은 옮기기만 하므로 슬롯 칸 + X 칸 글자 수와 같아야 한다."""
    D = s["D"]; return sum(1 for k, ev in enumerate(s["streams"][:D]) for x in ev if len(x) == 2)


def test_eval_path_matches_e13_plus_x_box(tmp_path):
    q, d = _SB._multi(tmp_path, n=3); tok = _tok()
    g = CstDialogueDataset([q], tok, slot_gate=12, x_slot=True, **MISS, **KW); e13 = CstDialogueDataset([q], tok, slot_gate=12, **KW)
    out = lambda s: [(s["ids"][p], s["labels"][p], s["chunk_at"][p], s["slot_at"][p]) for p in range(len(s["ids"])) if not s["is_input"][p]]
    tags = lambda s: [(s["tag_chunk"][p], s["slot_at"][p]) for p in range(len(s["ids"])) if s["tag_chunk"][p] >= 0]
    for i in range(len(g)):
        for dl in (4, 10):
            sx, s13 = g.cst_sequence(i, dl), e13.cst_sequence(i, dl)                            # rng 없음(평가) = 놓침 없음
            assert g.problems_of(sx) == [], (g.items[i], dl, g.problems_of(sx))
            assert sx["n_x_text"] == 0 and not sx["act_miss"].any() and out(sx) == out(s13) and tags(sx) == tags(s13)   # E13 + 라운드마다 X 칸
            assert sum(1 for p, t in enumerate(sx["ids"]) if sx["is_input"][p] and t == g.x_id) == sx["K"] + sx["n_rounds"]
    with pytest.raises(ValueError): CstDialogueDataset([q], tok, x_slot=True, **KW)                 # 활동 게이트 없이
    with pytest.raises(ValueError): CstDialogueDataset([q], tok, slot_gate=12, miss_drop=0.3, **KW)   # x_slot 없이 놓침 흉내


@pytest.mark.parametrize("tgt", [None, "German"])
def test_training_misses_route_to_x_and_pass_checks(tmp_path, tgt):
    q, d = _SB._multi(tmp_path, n=3)
    g = CstDialogueDataset([q], _tok(trans=bool(tgt)), slot_gate=12, x_slot=True, **MISS, **(dict(trans=_trans(d), ast_frac=1.0) if tgt else {}), **KW)
    moved = 0
    for i in range(len(g)):
        for dl in (4, 10):
            s = g.block_sequence(i, dl, rng=random.Random(1000 * i + dl), tgt=tgt)
            assert g.problems_of(s) == [], (g.items[i], dl, tgt, g.problems_of(s))
            D = s["D"]; kept = sum(1 for qs in s["x_streams"][:D] for q_ in qs if len(s["streams"][q_[0]][q_[1]]) == 2)
            assert kept + len(s["x_out"]) == _text_tokens(s)                                     # 전사 글자는 슬롯 칸 아니면 X 칸 — 잃거나 겹치지 않는다
            assert (s["n_x_text"] > 0) <= bool(s["act_miss"].any())                             # X 글자는 흉내 낸 놓침에서만
            moved += s["n_x_text"]
    assert moved > 0


@pytest.mark.parametrize("tgt", [None, "German"])
def test_block_state_x_reproduces_training_sequence(tmp_path, tgt):
    q, d = _SB._multi(tmp_path, n=3)
    g = CstDialogueDataset([q], _tok(trans=bool(tgt)), slot_gate=12, x_slot=True, **MISS, **(dict(trans=_trans(d), ast_frac=1.0) if tgt else {}), **KW)
    X, TR = g.x_id, (g.trans_id if tgt else -1); stop = set(g.spk_ids) | {g.csp.next_audio, g.csp.empty_audio, X} | ({TR} if tgt else set())
    hit = 0
    for i in range(len(g)):
        cid, t0, L = g.items[i]
        for dl in (4, 10):
            s = g.block_sequence(i, dl, rng=random.Random(1000 * i + dl), tgt=tgt); ids, inp = s["ids"], s["is_input"]; K, D = s["K"], s["D"]
            act = (np.array(slot_activity(g.eps[cid], s["speakers"], t0, L, K, D)) > 0) & ~s["act_miss"]   # 디코드가 받는 활동 = 놓침을 뺀 정답(gate_fa 0)
            st = BlockState(K, D, 100, 3, 10 ** 6, tuple(g.spk_ids[:D]), gate_lags=12, act=act, trans_id=TR, x_id=X, begin_id=g.begin_id, eot_id=g.eot_id)
            p = next(j for j, c in enumerate(s["chunk_of"]) if c >= 0)
            while True:
                pred = ids[p + 1] if not inp[p + 1] else g.csp.next_audio                       # 정답 다음 출력 또는 멈춤
                kind, v = st.consume(pred, stop)
                if kind == "pad": break
                p += 1
                if kind == "audio": assert inp[p] and s["chunk_of"][p] == v
                elif kind == "tag": assert inp[p] and ids[p] == g.spk_ids[v[1]] and s["tag_chunk"][p] == v[0], (i, dl, p)
                elif kind == "trans": assert inp[p] and ids[p] == TR and s["slot_at"][p] == v[1]
                elif kind == "xtag": assert inp[p] and ids[p] == X and s["tag_chunk"][p] == -1, (i, dl, p)
                elif kind == "text": assert not inp[p] and ids[p] == v
                else: assert kind == "empty" and inp[p] and ids[p] == g.csp.empty_audio
            assert p == len(ids) - 2 and st.forced == 0 and st.n_tags == s["n_tags"]
            em = emitted_from_blocks(ids, inp, s["tag_chunk"], s["slot_at"], tuple(g.spk_ids[:D]) + (X,), trans_id=TR if tgt else None, x_id=X, chunk_at=s["chunk_at"])
            assert st.emitted == em
            if not tgt:
                hs, _ = parse_emitted(em, tuple(g.spk_ids[:D]) + (X,), g.eot_id, begin_id=g.begin_id)
                assert [t for _, t in hs[D]] == s["x_out"]                                        # X 칸 해석 = 옮긴 글자
            hit += s["n_x_text"] > 0
    assert hit > 0


def test_speculative_x_decode_equals_token_by_token():
    from vapasr.hf.realtime import realtime_block_decode
    m, tok, bid = _SB._tiny_blocks(); m.eval(); torch.manual_seed(3); K, D = 30, 4
    tok.add_tokens([X_TOKEN]); xid = int(tok.convert_tokens_to_ids(X_TOKEN)); assert xid < m.get_input_embeddings().weight.shape[0]
    m.sp_ids[X_TOKEN] = xid; m.config.sp_ids[X_TOKEN] = xid; m.config.x_slot = True
    with torch.no_grad(): m.adapter_tagact.weight.copy_(torch.randn_like(m.adapter_tagact.weight))
    m.config.slot_gate = 6; m.config.runaway_cap = 3
    feats = torch.randn(1, K, 4); act = (torch.rand(K, 4) > 0.8).float(); prefix = [5, 6, 7]; emb = m.get_input_embeddings().weight
    EOT = m.sp_ids["<EOT>"]; words = [50, 60, 70, bid, EOT]; NA = m.next_audio
    pick = lambda h: NA if (h % 10) < 6 else words[(h // 10) % len(words)]
    out, forced, ticks, rounds, _, nf = realtime_block_decode(m, _SB._FakeLM(emb, pick), feats, prefix, D, act=act, max_flush_rounds=4)
    lm2 = _SB._FakeLM(emb, pick); lm2.run_all(emb[torch.tensor(prefix)], 0); pos = len(prefix); ce = m.chunk_embed(feats[None])[0]
    st = BlockState(K, D, m.config.runaway_cap, 4, int(6 * K) + 64, tuple(m.slot_ids(D)), gate_lags=6, act=act.numpy() > 0.5, x_id=xid, begin_id=bid, eot_id=EOT)
    stop = block_stops(m, D); x = ce[0]; assert xid in stop
    while True:
        tid = lm2.run(x[None], pos); pos += 1; kind, v = st.consume(tid, stop)
        if kind == "pad": break
        x = ce[v] if kind == "audio" else m.tag_embeds(v[0], act, D)[v[1]] if kind == "tag" else emb[xid] if kind == "xtag" else emb[v] if kind == "text" else emb[m.empty_audio]
    assert out == st.emitted and forced == st.forced and rounds == st.flush_rounds
    assert any(t == xid for _, t in out)                                                          # X 칸 출력이 있다


def test_attribute_x_uses_subthreshold_probability_then_last_active():
    K, d = 60, 10; A = np.zeros((K, 3), np.float32); A[5:15, 0] = 0.9; A[28:33, 1] = 0.3          # 슬롯 1 은 임계 아래 근거만
    t = lambda f: (f + 1 + d) * CHUNK_S                                                            # 방출 시각(오디오 프레임 f 의 단어)
    xw = [("a", t(30)), ("b", t(45)), ("c", t(2))]
    assert attribute_x(xw, A, d) == [1, 0, 0]                                                      # 임계 아래 확률 · 근거 없으면 직전 활성 슬롯 · 아무것도 없으면 0
    hw = merge_x([[("x", t(1))], [], []], xw, [1, 0, 0])
    assert [w for w, _ in hw[0]] == ["x", "c", "b"] and [w for w, _ in hw[1]] == ["a"]                    # 시각순


def test_n3d_soft_returns_probabilities():
    from vapasr.hf.modeling_vapasr import VapAsrForStreamingASR as M
    p = torch.rand(2, 9, 8)
    class FakeExpert:
        cfg = {"threshold": 0.5}
        def probs(self, wav, wav_len, K, train=False): return p
        def act(self, wav, wav_len, K, n, train=False): return (p[..., :n] > 0.5).float()
    m = SimpleNamespace(n3d=FakeExpert(), config=SimpleNamespace(slot_input_slots=4, n3d_soft=True), training=False)
    assert torch.equal(M.n3d_act(m, None, None, None), p[..., :4])
    m.training = True; assert torch.equal(M.n3d_act(m, None, None, None), p[..., :4])
