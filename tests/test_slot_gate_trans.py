"""E13(2026-11-02 — cst_dialogue 모듈 설명 E13): 활동 게이트 슬롯 블록(slot_gate·gate_fa)과 슬롯 블록 번역 모드(<TRANS> 칸) — 데이터(gate_rounds·build_block_sequence·
block_problems·block_sequence tgt), 디코더(BlockState 게이트·번역 칸이 정답 시퀀스를 그대로 되짚는다, 추측 prefill realtime_block_decode 가 한 토큰씩 디코드와 같다),
평가(emitted_from_blocks·parse_emitted4 begin_id 오라클 왕복이 Stage 4 네 흐름 정답과 같다). CPU."""
import importlib.util, pathlib, random
import numpy as np, pytest, torch
pytest.importorskip("soundfile")
from vapasr.data.cst_dialogue import CstDialogueDataset, gate_rounds, slot_activity
from vapasr.data.cst_eval import emitted_from_blocks, parse_emitted, parse_emitted4
from vapasr.hf.batch_decode import BlockState, block_stops

_HERE = pathlib.Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", _HERE / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


_SB = _load("test_slot_blocks")
KW = dict(delays=(4, 10), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, slot_blocks=True, declare=4)


def _trans(d):
    """발화마다 독일어 번역 두 토큰(확정 시각 = 발화 안)."""
    return {f"{d.conv_id}|{u.speaker}|{u.utt_id}": {"German": (np.array([60, 61]), np.array([u.start + 1.0, u.end]))} for u in d.utterances}


def _tok_tr():
    t = _SB._tok(); t.add_tokens(["<TRANS>"]); return t


def test_gate_rounds_window():
    act = np.zeros((10, 2), bool); act[3, 0] = True
    g = gate_rounds(act, 3, 12)
    assert g[0] == {3, 4, 5} and g[1] == set()                                                   # 활동 창 [k − 2, k] 에 프레임 3 이 드는 라운드만, K 넘는 라운드는 창이 비면 닫힌다


def test_gate_layout_problems_and_same_outputs(tmp_path):
    q, d = _SB._multi(tmp_path, n=3)
    e9 = CstDialogueDataset([q], _SB._tok(), **KW); g = CstDialogueDataset([q], _SB._tok(), slot_gate=12, **KW)
    assert len(g) == len(e9) > 0
    NA = g.csp.next_audio
    for i in range(len(g)):
        cid, t0, L = g.items[i]
        for dl in (4, 10):
            s9, sg = e9.cst_sequence(i, dl), g.cst_sequence(i, dl)
            assert g.problems_of(sg) == [], (g.items[i], dl, g.problems_of(sg))
            K, D = sg["K"], sg["D"]
            assert sg["n_tags"] + sg["n_gated"] == (K + sg["n_rounds"]) * D and 0 < sg["n_gated"] and len(sg["ids"]) < len(s9["ids"])   # 태그가 준다
            out = lambda s: [(s["ids"][p], s["labels"][p], s["chunk_at"][p], s["slot_at"][p]) for p in range(len(s["ids"])) if not s["is_input"][p]]
            assert out(sg) == out(s9)                                                            # 출력·label·청크·슬롯은 E9 와 같다
            assert emitted_from_blocks(sg["ids"], sg["is_input"], sg["tag_chunk"], sg["slot_at"], g.spk_ids) == \
                emitted_from_blocks(s9["ids"], s9["is_input"], s9["tag_chunk"], s9["slot_at"], e9.spk_ids)
            assert sum(1 for inp, y in zip(sg["is_input"], sg["labels"]) if inp and y == NA) == sg["n_tags"]   # 칸마다 멈춤 하나(가림 없음)
            act = np.array(slot_activity(g.eps[cid], sg["speakers"], t0, L, K, D)) > 0; want = gate_rounds(act, 12, K + 2)
            tags = {(sg["tag_chunk"][p], sg["slot_at"][p]) for p in range(len(sg["ids"])) if sg["tag_chunk"][p] >= 0}
            outs = {(c, s_) for _, _, c, s_ in out(sg)}
            assert all(((k, s_) in tags) == (k in want[s_] or (k, s_) in outs) for k in range(K) for s_ in range(D))   # 태그 = 활동 창 ∪ 출력


def test_gate_fa_training_only(tmp_path):
    q, d = _SB._multi(tmp_path, n=3)
    g = CstDialogueDataset([q], _SB._tok(), slot_gate=12, gate_fa=1.0, **KW)
    for i in range(len(g)):
        s0, s1 = g.cst_sequence(i, 4), g.cst_sequence(i, 4, rng=random.Random(i))
        assert g.problems_of(s1) == [] and s1["n_tags"] >= s0["n_tags"]
    assert any(g.cst_sequence(i, 4, rng=random.Random(i))["n_tags"] > g.cst_sequence(i, 4)["n_tags"] for i in range(len(g)))   # 헛 구간은 학습 rng 에서만
    with pytest.raises(ValueError): CstDialogueDataset([q], _SB._tok(), slot_gate=12, **{**KW, "slot_blocks": False})
    with pytest.raises(ValueError): CstDialogueDataset([q], _SB._tok(), gate_fa=0.5, **KW)


@pytest.mark.parametrize("gate", [0, 12])
def test_translation_blocks_match_stage4_streams(tmp_path, gate):
    q, d = _SB._multi(tmp_path, n=3)
    ds = CstDialogueDataset([q], _tok_tr(), trans=_trans(d), ast_frac=1.0, slot_gate=gate, **KW)
    TR, B, EOT = ds.trans_id, ds.begin_id, ds.eot_id
    for i in range(len(ds)):
        for dl in (4, 10):
            s = ds.cst_sequence(i, dl, tgt="German")
            assert s["blocks"] and s["trans"] and s["tgt"] == ("German",) * 4 and s["n_tr"] > 0 and ds.problems_of(s) == [], (ds.items[i], dl, ds.problems_of(s))
            ins = [p for p in range(len(s["ids"])) if s["is_input"][p] and s["tag_chunk"][p] >= 0]
            for p in ins:                                                                      # 태그 칸 뒤 첫 입력은 같은 슬롯의 <TRANS>
                nxt = next(j for j in range(p + 1, len(s["ids"])) if s["is_input"][j])
                assert s["ids"][nxt] == TR and s["slot_at"][nxt] == s["slot_at"][p] and s["tag_chunk"][nxt] == -1
            em = emitted_from_blocks(s["ids"], s["is_input"], s["tag_chunk"], s["slot_at"], ds.spk_ids, trans_id=TR)
            src, trn, _ = parse_emitted4(em, ds.spk_ids, TR, EOT, begin_id=B)
            q4 = ds.cst4_sequence(i, dl, "German"); n4 = len(q4["streams"]) // 2
            for k in range(len(s["speakers"])):                                                  # 슬롯별 전사·번역 토큰이 Stage 4 네 흐름 정답과 같다
                assert [t for _, t in src[k]] == [x[0] for x in q4["streams"][k] if len(x) == 2]
                assert [t for _, t in trn[k]] == [x[0] for x in q4["streams"][n4 + k] if len(x) == 2]
            h, _ = parse_emitted(em, ds.spk_ids, EOT, begin_id=B)
            assert sum(len(x) for x in h) > sum(len(x) for x in src)                              # 번역 칸을 모르는 해석은 번역을 전사로 섞는다 — parse_emitted4 를 써야 한다


def test_translation_eot_after_tails_and_masks(tmp_path):
    """번역 없는 행(miss)은 그 슬롯 번역 칸 label 을, 마지막 행에 번역이 없는 episode 의 <EOT> 는 label 을 가린다(전사 칸은 그대로 학습)."""
    q, d = _SB._multi(tmp_path, n=3)
    tr = _trans(d); gone = d.utterances[1]; del tr[f"{d.conv_id}|{gone.speaker}|{gone.utt_id}"]
    ds = CstDialogueDataset([q], _tok_tr(), trans=tr, ast_frac=1.0, slot_gate=12, **KW)
    hit = 0
    for i in range(len(ds)):
        s = ds.cst_sequence(i, 4, tgt="German"); assert ds.problems_of(s) == []
        if not s["eot_hide"]: continue
        hit += 1
        for p, q_ in enumerate(s["src_at"]):
            if q_ is not None and (q_[0], q_[1]) in s["eot_hide"]: assert s["ids"][p] == ds.eot_id and s["labels"][p] == -100
    assert hit > 0


def test_block_state_gate_trans_reproduces_training_sequence(tmp_path):
    q, d = _SB._multi(tmp_path, n=3)
    ds = CstDialogueDataset([q], _tok_tr(), trans=_trans(d), ast_frac=1.0, slot_gate=12, **KW)
    TR = ds.trans_id; stop = set(ds.spk_ids) | {ds.csp.next_audio, ds.csp.empty_audio, TR}
    for tgt in (None, "German"):
        for i in range(len(ds)):
            cid, t0, L = ds.items[i]
            for dl in (4, 10):
                s = ds.block_sequence(i, dl, tgt=tgt); ids, inp = s["ids"], s["is_input"]; K, D = s["K"], s["D"]
                act = np.array(slot_activity(ds.eps[cid], s["speakers"], t0, L, K, D)) > 0
                st = BlockState(K, D, 100, 3, 10 ** 6, tuple(ds.spk_ids[:D]), gate_lags=12, act=act, trans_id=TR if tgt else -1, begin_id=ds.begin_id, eot_id=ds.eot_id)
                p = next(j for j, c in enumerate(s["chunk_of"]) if c >= 0)
                while True:
                    pred = ids[p + 1] if not inp[p + 1] else ds.csp.next_audio                    # 정답 다음 출력 또는 멈춤
                    kind, v = st.consume(pred, stop)
                    if kind == "pad": break
                    p += 1
                    if kind == "audio": assert inp[p] and s["chunk_of"][p] == v
                    elif kind == "tag": assert inp[p] and ids[p] == ds.spk_ids[v[1]] and s["tag_chunk"][p] == v[0], (i, dl, p)
                    elif kind == "trans": assert inp[p] and ids[p] == TR and s["slot_at"][p] == v[1]
                    elif kind == "text": assert not inp[p] and ids[p] == v
                    else: assert kind == "empty" and inp[p] and ids[p] == ds.csp.empty_audio
                assert p == len(ids) - 2 and st.forced == 0 and st.n_tags == s["n_tags"]
                assert st.emitted == emitted_from_blocks(ids, inp, s["tag_chunk"], s["slot_at"], ds.spk_ids, trans_id=TR if tgt else None)


@pytest.mark.parametrize("seed,trans", [(0, False), (1, True), (2, True)])
def test_gated_trans_speculative_decode_equals_token_by_token(seed, trans):
    from vapasr.hf.realtime import realtime_block_decode
    m, tok, bid = _SB._tiny_blocks(); m.eval(); torch.manual_seed(seed); K, D = 30, 4
    if "<TRANS>" not in m.sp_ids:
        tok.add_tokens(["<TRANS>"]); m.sp_ids["<TRANS>"] = int(tok.convert_tokens_to_ids("<TRANS>")); assert m.sp_ids["<TRANS>"] < m.get_input_embeddings().weight.shape[0]
    with torch.no_grad(): m.adapter_tagact.weight.copy_(torch.randn_like(m.adapter_tagact.weight))
    m.config.slot_gate = 6; m.config.runaway_cap = 3
    feats = torch.randn(1, K, 4); act = (torch.rand(K, 4) > 0.8).float(); prefix = [5, 6, 7]; emb = m.get_input_embeddings().weight
    EOT, TR = m.sp_ids["<EOT>"], m.sp_ids["<TRANS>"]; words = [50, 60, 70, bid, EOT]; NA = m.next_audio
    pick = lambda h: NA if (h % 10) < 5 + seed else words[(h // 10) % len(words)]
    out, forced, ticks, rounds, _, nf = realtime_block_decode(m, _SB._FakeLM(emb, pick), feats, prefix, D, act=act, max_flush_rounds=4, trans=trans)
    lm2 = _SB._FakeLM(emb, pick); lm2.run_all(emb[torch.tensor(prefix)], 0); pos = len(prefix); ce = m.chunk_embed(feats[None])[0]
    st = BlockState(K, D, m.config.runaway_cap, 4, int((12 if trans else 6) * K) + 64, tuple(m.slot_ids(D)), gate_lags=6, act=act.numpy() > 0.5,
                    trans_id=TR if trans else -1, begin_id=bid, eot_id=EOT)
    stop = block_stops(m, D, trans); x = ce[0]
    while True:
        tid = lm2.run(x[None], pos); pos += 1; kind, v = st.consume(tid, stop)
        if kind == "pad": break
        x = ce[v] if kind == "audio" else m.tag_embeds(v[0], act, D)[v[1]] if kind == "tag" else emb[TR] if kind == "trans" else emb[v] if kind == "text" else emb[m.empty_audio]
    assert out == st.emitted and forced == st.forced and rounds == st.flush_rounds and len(ticks) == K
    assert len(out) > 0 and st.n_tags < (K + rounds) * D                                         # 말이 있고, 게이트가 태그를 줄였다
    if trans: assert any(t == TR for _, t in out)                                                # 번역 칸 출력이 있다
