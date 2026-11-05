"""디코드 때 재귀속(2026-11-05, batch_decode.BlockRetag · realtime_block_decode(retag)): 활동을 어긴 슬롯 칸을 옮기고 라운드를 학습 꼴로 고쳐 다시 넣는다. CPU.
- 한 토큰씩 도는 참조(라운드를 고치면 기록을 그 꼴로 바꿔 처음부터 다시 넣은 것과 같은 입력)와 출력·셈이 같다 — 가짜 LM 은 위치마다 그때까지 입력 전체로 다음 토큰을 고른다.
- 끝난 출력에 활동을 어긴 글자 칸이 남지 않는다. 활동이 모두 켜져 있으면 재귀속 없는 디코드와 같다.
- LongThinker: 되돌려 더 짧게 덮어쓴 뒤 이어 넣은 hidden = 고친 열을 HF Qwen3 로 처음부터 넣은 것."""
import importlib.util, pathlib
import numpy as np, pytest, torch
pytest.importorskip("soundfile")

_HERE = pathlib.Path(__file__).parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


_SB = _load("_t_test_slot_blocks_r", _HERE / "test_slot_blocks.py")


def _setup(seed, p_act=0.6, gate=6):
    m, tok, bid = _SB._tiny_blocks(); m.eval(); torch.manual_seed(seed); K, D = 30, 4
    with torch.no_grad(): m.adapter_tagact.weight.copy_(torch.randn_like(m.adapter_tagact.weight))
    m.config.slot_gate = gate; m.config.runaway_cap = 3
    feats = torch.randn(1, K, 4); act = (torch.rand(K, 4) > p_act).float(); emb = m.get_input_embeddings().weight
    words = [50, 60, 70, bid, m.sp_ids["<EOT>"]]; NA = m.next_audio
    pick = lambda h: NA if (h % 10) < 4 + seed % 3 else words[(h // 10) % len(words)]
    return m, feats, act, emb, pick, K, D


def _reference(m, emb, pick, feats, act, prefix, D, retag, max_flush):
    """한 토큰씩: 입력 기록 hist(위치 = 칸)에 넣고, 예측 = pick(hash(그때까지 입력 바이트)) — _FakeLM 과 같은 규칙. 재귀속이면 라운드 끝에 기록을 고친 꼴로 바꾼다."""
    from vapasr.hf.batch_decode import block_gate, block_stops
    cfg = m.config; ce = m.chunk_embed(feats[None])[0].to(emb.dtype); A = m.block_act(feats, act); K = ce.shape[0]
    tags, stop = m.slot_ids(D), block_stops(m, D); gl, ga = block_gate(m, A); BG, EOT = int(m.sp_ids["<ONSET>"]), int(m.sp_ids["<EOT>"])
    cap, max_total = cfg.runaway_cap, int(6.0 * K) + 64
    hist = [emb[i] for i in prefix]; out, utt, st = [], [0] * D, dict(forced=0, total=0)
    pred = lambda: pick(hash(tuple(x.detach().numpy().tobytes() for x in hist)))

    def rnd(k, head):
        hist.append(head)
        if gl:
            lo, hi = max(0, k - gl + 1), min(ga.shape[0], k + 1)
            sl = [s for s in range(D) if utt[s] > 0 or (s < ga.shape[1] and hi > lo and bool(ga[lo:hi, s].any()))]
        else: sl = list(range(D))
        te = m.tag_embeds(k, A, D, None).to(emb.dtype); mark, o0, box, n_round = len(hist), len(out), {}, 0
        for s in sl:
            hist.append(te[s]); tid = pred(); n = 0; tagged = False
            while True:
                if tid in stop or n >= cap or st["total"] >= max_total: st["forced"] += int(tid not in stop); break
                if not tagged: out.append((k, tags[s])); tagged = True
                out.append((k, int(tid))); box.setdefault(s, []).append(int(tid)); n += 1; n_round += 1; st["total"] += 1
                if gl and tid in (BG, EOT): utt[s] = utt[s] + 1 if tid == BG else max(0, utt[s] - 1)
                hist.append(emb[tid]); tid = pred()
        if retag is not None and box:
            own = {s: list(box.get(s, [])) for s in sl}; moved = []
            for s in sl:
                toks = box.get(s, [])
                if not any(t_ not in (BG, EOT) for t_ in toks): continue
                t = retag.target(k, s, D)
                if t != s: own[s] = [t_ for t_ in toks if t_ == EOT]; moved.append((s, t, [t_ for t_ in toks if t_ != EOT]))
            if moved:
                for s, t, mv in moved:
                    own.setdefault(t, []); own[t] += mv; nb = mv.count(BG)
                    if gl and nb: utt[s] = max(0, utt[s] - nb); utt[t] += nb
                order = sorted(own); out[o0:] = [y for s in order if own[s] for y in [(k, tags[s])] + [(k, t_) for t_ in own[s]]]
                del hist[mark:]; hist.extend(v for s in order for v in [te[s]] + [emb[t_] for t_ in own[s]])
        return n_round

    for k in range(K): rnd(k, ce[k])
    rounds = 0
    for r_ in range(max_flush):
        rounds += 1
        if rnd(K + r_, emb[m.empty_audio]) == 0: break
    return out, st["forced"], rounds


def _violations(out, tags, retag, D, BG, EOT):
    """끝난 출력에서 글자 칸(라운드 k · 슬롯 s)마다 retag.target(k, s) ≠ s 인 수."""
    cur, bad, seen = -1, 0, set()
    for k, t in out:
        if t in tags: cur = tags.index(t); continue
        if t in (BG, EOT) or (k, cur) in seen: continue
        seen.add((k, cur)); bad += int(retag.target(k, cur, D) != cur)
    return bad


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_retag_decode_equals_reference_and_leaves_no_violation(seed):
    from vapasr.hf.batch_decode import BlockRetag
    from vapasr.hf.realtime import realtime_block_decode
    m, feats, act, emb, pick, K, D = _setup(seed); A = m.block_act(feats, act); prefix = [5, 6, 7]
    R = BlockRetag(A, delta=2); out, forced, _, rounds, _, nf = realtime_block_decode(m, _SB._FakeLM(emb, pick), feats, prefix, D, act=act, max_flush_rounds=4, retag=R)
    ref = _reference(m, emb, pick, feats, act, prefix, D, BlockRetag(A, delta=2), 4)
    assert (out, forced, rounds) == ref
    tags = list(m.slot_ids(D)); BG, EOT = int(m.sp_ids["<ONSET>"]), int(m.sp_ids["<EOT>"])
    assert _violations(out, tags, BlockRetag(A, delta=2), D, BG, EOT) == 0
    assert R.stats["rounds"] <= R.stats["moved"] <= R.stats["boxes"] and R.stats["tokens"] >= R.stats["moved"]


def test_retag_moves_something_and_changes_later_context():
    """여러 씨앗 중 옮긴 칸이 있고, 그런 씨앗에서는 재귀속 없는 디코드와 출력이 다르다(사후 재귀속과 달리 뒤 라운드가 고친 문맥을 본다)."""
    from vapasr.hf.batch_decode import BlockRetag
    from vapasr.hf.realtime import realtime_block_decode
    moved = 0
    for seed in range(6):
        m, feats, act, emb, pick, K, D = _setup(seed); A = m.block_act(feats, act); R = BlockRetag(A, delta=2)
        a = realtime_block_decode(m, _SB._FakeLM(emb, pick), feats, [5, 6, 7], D, act=act, max_flush_rounds=4, retag=R)
        b = realtime_block_decode(m, _SB._FakeLM(emb, pick), feats, [5, 6, 7], D, act=act, max_flush_rounds=4)
        moved += R.stats["moved"]
        if R.stats["moved"]: assert a[0] != b[0]
        else: assert a[0] == b[0] and R.stats["rounds"] == 0
    assert moved > 0


def test_all_active_equals_no_retag_and_rejects_trans():
    from vapasr.hf.batch_decode import BlockRetag
    from vapasr.hf.realtime import realtime_block_decode
    m, feats, act, emb, pick, K, D = _setup(1, p_act=-1.0)                                         # 모든 슬롯 활동 켜짐
    R = BlockRetag(m.block_act(feats, act), delta=2)
    a = realtime_block_decode(m, _SB._FakeLM(emb, pick), feats, [5, 6, 7], D, act=act, max_flush_rounds=4, retag=R)
    b = realtime_block_decode(m, _SB._FakeLM(emb, pick), feats, [5, 6, 7], D, act=act, max_flush_rounds=4)
    assert a[:2] == b[:2] and R.stats["moved"] == 0 and R.stats["boxes"] > 0
    with pytest.raises(AssertionError): realtime_block_decode(m, _SB._FakeLM(emb, pick), feats, [5, 6, 7], D, act=act, max_flush_rounds=4, retag=R, trans=True)


def test_block_retag_target_rule_matches_follow_stats():
    """target = follow_stats 의 옮김(단어 방출 시각 (k + 1)·80 ms → f = k − δ, 창 [f − 3, f + 1], 다른 슬롯만 켜지면 활동 합 최대 슬롯)."""
    from vapasr.hf.batch_decode import BlockRetag
    from vapasr.data.cst_eval import follow_stats
    rng = np.random.default_rng(0); K, N, D, d = 40, 4, 4, 3
    for _ in range(50):
        A = rng.random((K, N)) > 0.7; R = BlockRetag(A, delta=d)
        for k in range(K + 3):
            for s in range(D):
                _, mv = follow_stats([[("w", (k + 1) * 0.08)] if x == s else [] for x in range(D)], A, d)
                assert R.target(k, s, D) == next(i for i, x in enumerate(mv) if x)


def test_long_thinker_rewind_shorter_overwrite_equals_hf():
    """LongThinker: 위치 [0, 40) 을 넣은 뒤 32 부터 5 개로 다시 덮어쓰고 이어 6 개를 넣으면, 마지막 hidden = 고친 열(43 위치)의 HF forward."""
    LS = _load("_t_test_long_session_r", _HERE / "test_long_session.py")
    from vapasr.hf.long_session import LongThinker
    m, th = LS.tiny_qwen3(); torch.manual_seed(5); E = torch.randn(40, 48); E2 = torch.randn(5, 48); E3 = torch.randn(6, 48)
    lt = LongThinker(th, blocked_ids=[3], max_len=16, blk=5); lt.hidden(E[:30], 0); lt.hidden(E[30:40], 30)
    lt.hidden(E2, 32); h = lt.hidden(E3, 37)
    seq = torch.cat([E[:32], E2, E3])
    with torch.no_grad(): ref = m.model(inputs_embeds=seq[None]).last_hidden_state[0, -6:]
    assert torch.allclose(th.model.norm(h)[0], ref, atol=1e-5), (th.model.norm(h)[0] - ref).abs().max()
