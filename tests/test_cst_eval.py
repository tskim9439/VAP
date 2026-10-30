"""Stage 3 평가 도우미(vapasr/data/cst_eval.py) — 출력 해석, 화자 귀속 오류율(slot·cp·화자 무시), EOT 짝·합산. CPU."""
from vapasr.data.cst_eval import parse_emitted, score_window, match_eot, eot_tol, pool, ed, align, attr_errors

A, B, EOT = 900, 901, 902


def test_parse_emitted_slots_eot_and_drop():
    em = [(0, 5), (1, A), (1, 6), (2, EOT), (3, B), (3, 7), (4, EOT), (5, A), (5, 8)]
    slots, eots = parse_emitted(em, (A, B), EOT, drop_chunks={5})
    assert slots[0] == [(0.08, 5), (0.16, 6)] and slots[1] == [(0.32, 7)]                    # 첫 태그 앞 토큰은 A, 청크 5 는 버림
    assert eots == [(0, 0.24), (1, 0.4)]


def test_score_window_permutation_and_agnostic():
    ref = [[("hello", 1.0), ("there", 1.5)], [("yes", 3.0)]]
    same = score_window(ref, [[("hello", 1.2), ("there", 1.6)], [("yes", 3.1)]], "English")
    assert same == dict(n_ref=3, n_hyp=3, slot=0, cp=0, agn=0, swap=False, perm=[0, 1])
    sw = score_window(ref, [[("yes", 3.1)], [("hello", 1.2), ("there", 1.6)]], "English")
    assert sw["slot"] == 4 and sw["cp"] == 0 and sw["agn"] == 0 and sw["swap"] and sw["perm"] == [1, 0]
    ko = score_window([[("안녕 하세요", 1.0)], []], [[("안녕하세오", 1.0)], []], "Korean")
    assert ko["n_ref"] == 5 and ko["cp"] == 1                                                  # 공백 없는 글자 단위


def test_match_eot_counts_latency_and_hold():
    ref = [(0, 2.0, 1.0), (1, 4.0, 0.0), (0, 6.0, 0.5), (1, 8.0, None)]
    hyp = [(0, 2.5), (1, 4.2), (0, 6.3), (0, 9.9)]
    m = match_eot(ref, hyp, [0, 1], tol=eot_tol(4))
    assert (m["n_pos"], m["n_hold"], m["n_both"], m["hit_pos"], m["hit_hold"], m["hit_both"], m["unmatched"]) == (1, 1, 1, 1, 1, 1, 1) and m["lat"] == [0.5]
    sw = match_eot(ref, [(1, 2.5)], [1, 0], tol=1.0); assert sw["hit_pos"] == 1                  # 가설 슬롯 B 가 순열로 참조 A 에 붙는다
    assert match_eot(ref, [(0, 1.9)], [0, 1], tol=1.0)["unmatched"] == 1                         # 끝보다 이른 EOT 는 짝이 아니다


def test_pool():
    w = dict(n_ref=10, slot=2, cp=1, agn=1, swap=False, eot=dict(n_hyp=2, n_pos=2, n_hold=1, n_both=0, hit_pos=1, hit_hold=1, hit_both=0, unmatched=0, lat=[0.4]))
    p = pool([w, dict(w, swap=True, eot=dict(w["eot"], lat=[0.6]))])
    assert p["err_cp"] == 10.0 and p["err_slot"] == 20.0 and p["swap_share"] == 0.5 and p["eot_recall"] == 0.5 and p["eot_precision"] == 0.5 and p["eot_hold_fire"] == 1.0
    assert p["eot_lat_median"] == 0.5 and ed("abc", "abd") == 1


def test_emitted_from_sequence_matches_decode_layout():
    from vapasr.data.cst_eval import emitted_from_sequence
    NEXT, EMPTY, PAD = 1000, 1001, 999
    ids = [1, 2, PAD, A, 5, NEXT, PAD, NEXT, EMPTY, 6, EOT, NEXT, EMPTY, NEXT]
    is_input = [True, True, True, False, False, False, True, False, True, False, False, False, True, False]
    chunk_of = [-1, -1, 0, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1]
    assert emitted_from_sequence(ids, is_input, chunk_of, NEXT, EMPTY) == [(0, A), (0, 5), (2, 6), (2, EOT)]   # flush 는 k = K(=2)


def test_align_matches_ed():
    assert align("abc", "abd") == [(0, 0), (1, 1), (2, 2)] and align("ab", "b") == [(0, None), (1, 0)] and align("", "x") == [(None, 0)]
    for a, b in (("kitten", "sitting"), ("abcd", "dcba"), ("", "")):
        pr = align(a, b); assert sum(1 for i, j in pr if i is None or j is None or a[i] != b[j]) == ed(a, b)


def test_attr_errors_types_and_overlap():
    w = lambda s, ts: [(x, t) for x, t in zip(s.split(), ts)]
    ref = [w("a b c d e f", (1, 2, 3, 4, 8, 9)), w("x y z", (5, 6, 7))]                       # run A[a–d] · B[x–z] · A[e f]
    hyp = [w("a c x", (1, 3, 5)), w("b d y z e f", (2, 4, 6, 7, 8, 9))]                        # b 헛교대 · d 일찍 · x 늦게 · e f 통째
    busy = [[(0.5, 4.2), (7.8, 9.2)], [(3.9, 7.2)]]                                            # d(4.0)는 B 와 겹친다
    r = attr_errors(ref, hyp, "English", [0, 1], busy)
    assert r["units"] == {"inner/solo": 1, "early/ov": 1, "late/solo": 1, "whole_short/solo": 2} and r["n_mis"] == 5 and r["agn"] == 0
    assert (r["runs"], r["runs_short"], r["whole_short_runs"], r["whole_long_runs"], r["ref_changes"], r["hyp_changes"], r["late_words"]) == (3, 2, 1, 0, 2, 5, [1])
    assert (r["whole_iso_units"], r["whole_streak_units"]) == (2, 0)
    ab = [w("a b c d e f g h", (1, 2, 3, 4, 9, 10, 11, 12)), w("w x y z", (5, 6, 7, 8))]
    fl = attr_errors(ab, [w("a b c d w x y z", (1, 2, 3, 4, 5, 6, 7, 8)), w("e f g h", (9, 10, 11, 12))], "English", [0, 1])
    assert fl["units"] == {"whole_long/solo": 8} and (fl["whole_iso_units"], fl["whole_streak_units"], fl["whole_streak_edge_units"]) == (0, 8, 8)   # 둘째 턴부터 정체가 뒤바뀜
    long = attr_errors([w("a b c d", (1, 2, 3, 4)), []], [[], w("a b c d", (1, 2, 3, 4))], "English", [0, 1])
    assert long["units"] == {"whole_long/solo": 4} and long["whole_long_runs"] == 1
    assert attr_errors([w("a b c d", (1, 2, 3, 4)), []], [[], w("a b c d", (1, 2, 3, 4))], "English", [1, 0])["n_mis"] == 0   # cp 순열 적용
    gap = attr_errors([w("a b c", (1, 2, 3)), []], [w("a c", (1, 3)), []], "English", [0, 1])
    assert gap["n_mis"] == 0 and gap["agn"] == 1 and gap["n_pair"] == 2                       # 빠진 단어는 귀속 오류가 아니다
    ko = attr_errors([[("안녕 하세요", 1.0)], []], [[], [("안녕하세요", 1.0)]], "Korean", [0, 1])
    assert ko["n_ref"] == 5 and ko["units"] == {"whole_short/solo": 5}                         # 글자 단위로 세고, short 는 단어(2 개) 기준


def test_parse_emitted4_streams_and_drops():
    from vapasr.data.cst_eval import parse_emitted4
    A, B, TR, EOT = 1002, 1003, 1005, 1004
    em = [(5, A), (5, 10), (5, TR), (5, 30), (7, A), (7, 11), (8, EOT), (9, B), (9, TR), (9, 40), (9, EOT), (12, B), (12, 20), (12, TR), (12, 41)]
    s, t, e = parse_emitted4(em, (A, B), TR, EOT)
    assert s == [[(0.48, 10), (0.64, 11)], [(1.04, 20)]] and t == [[(0.48, 30)], [(0.8, 40), (1.04, 41)]] and e == [(0, 0.72), (1, 0.8)]
    s2, t2, e2 = parse_emitted4(em, (A, B), TR, EOT, drop_chunks={7}, drop_tr_chunks={12})
    assert s2 == [[(0.48, 10)], [(1.04, 20)]] and t2 == [[(0.48, 30)], [(0.8, 40)]] and e2 == e        # 버린 청크: 전사·번역 / 번역만
    assert parse_emitted4([(3, 50), (3, TR), (3, 51)], (A, B), TR, EOT)[:2] == ([[(0.32, 50)], []], [[(0.32, 51)], []])   # 첫 태그 앞 = 슬롯 A


def test_lag_matched():
    from vapasr.data.cst_eval import lag_matched
    ref = [("a", 1.0), ("b", 1.2), ("c", 2.0)]; hyp = [("a", 1.16), ("x", 1.3), ("c", 2.4)]
    assert lag_matched(ref, hyp) == [0.16, 0.4] and lag_matched(ref, []) == [] and lag_matched(ref, ref) == [0.0, 0.0, 0.0]


def test_en_norm_whisper_switch():
    """2026-10-27: 영어 채점 정규화 — 기본 ours 는 예전 그대로, whisper 는 채움말을 지우고 철자·숫자를 표준화한다(단어마다). 끝나면 ours 로 되돌린다."""
    import pytest
    pytest.importorskip("whisper_normalizer")
    from vapasr.data import cst_eval
    ref = [[("Um", 0.1), ("the", 0.2), ("colour", 0.3), ("twenty", 0.4)]]; hyp = [[("the", 0.2), ("color", 0.3), ("20", 0.4)]]
    base = cst_eval.score_window(ref, hyp, "English"); assert base["n_ref"] == 4 and base["cp"] >= 2       # ours: 채움말·영국 철자가 남는다
    try:
        cst_eval.set_en_norm("whisper"); w = cst_eval.score_window(ref, hyp, "English")
        assert w["n_ref"] == 3 and w["cp"] == 0 and w["agn"] == 0                                          # um 제거, colour → color, twenty → 20
    finally:
        cst_eval.set_en_norm("ours")
    assert cst_eval.score_window(ref, hyp, "English") == base
    with pytest.raises(AssertionError): cst_eval.set_en_norm("chime8")


def test_relabel_by_activity():
    """2026-10-28 E7 진단: 가설 단어를 방출 청크 − δ 근처 활동이 가장 큰 슬롯으로 다시 나눈다(근처 활동이 약하면 넓힌다)."""
    import numpy as np
    from vapasr.data.cst_eval import relabel_by_activity
    act = np.zeros((40, 3), np.float32); act[0:10, 2] = 1; act[10:25, 0] = 0.9; act[30:40, 1] = 1                 # 무음 25–29
    t = lambda k: (k + 1) * 0.08                                                                              # 방출 시각 = 청크 끝
    hyp = [[("a", t(5 + 4)), ("b", t(20 + 4))], [("c", t(33 + 4))], [("d", t(28 + 4))]]                        # δ 4: 오디오 프레임 5·20·33·28(근처 25–29 무음)
    out = relabel_by_activity(hyp, act, delta=4)
    assert [w for w, _ in out[2]] == ["a"] and [w for w, _ in out[0]] == ["b"] and set(w for w, _ in out[1]) == {"c", "d"}   # d(무음) 는 ±12 로 넓혀 30–39 의 1 을 고른다
    assert sum(len(x) for x in out) == 4 and all(x == sorted(x, key=lambda z: z[1]) for x in out)


def test_act_runs_roundtrip():
    """2026-10-30 활동 추종: 행에 싣는 활동 구간(act_runs)과 되돌리기."""
    import numpy as np
    from vapasr.data.cst_eval import act_runs, act_from_runs
    A = np.zeros((30, 3), np.uint8); A[0:4, 0] = 1; A[10:12, 0] = 1; A[29, 0] = 1; A[5:30, 2] = 1
    r = act_runs(A); assert r == [[[0, 4], [10, 12], [29, 30]], [], [[5, 30]]]
    assert (act_from_runs(r, 30) == A).all() and act_from_runs([[], []], 5).shape == (5, 2)


def test_follow_stats_categories_truth_and_moved():
    """단어 오디오 프레임 f = 방출 청크 − δ, 창 [f − 3, f + 1]: on·off_other·off_none·단독·겹침, 정답으로 누가 맞았나, 어긋난 단어만 활동 슬롯으로 옮긴다."""
    import numpy as np
    from vapasr.data.cst_eval import follow_stats, score_window
    t = lambda f, d=4: (f + d + 1) * 0.08                                                                         # 오디오 프레임 f 의 단어가 나오는 방출 시각
    act = np.zeros((60, 2), np.uint8); act[0:20, 0] = 1; act[15:30, 1] = 1; act[40:50, 1] = 1                    # 겹침 15–19, 무음 30–39·50–59
    ref = np.zeros((60, 2), np.uint8); ref[0:20, 0] = 1; ref[20:30, 1] = 1; ref[40:50, 0] = 1                     # 40–49 는 실제로 화자 0(활동이 틀림)
    hyp = [[("a", t(5)), ("b", t(25)), ("c", t(45)), ("x", t(35))], [("d", t(17)), ("e", t(8))]]
    st, moved = follow_stats(hyp, act, 4, ref_act=ref, perm=[0, 1])
    # a: 단독 on(맞음) · b: 단독 off_other(활동 1 = 정답 1 → 활동만 맞음) · c: off_other(LLM 0 = 정답 0, 활동 1 틀림 → LLM 만 맞음) · x: 무음(정답도 무음 → LLM 틀림)
    # d: 겹침 on(창 14–18 에 정답 1 없음 → 틀림) · e: 단독 off_other(활동 0, 정답 0 → 활동만 맞음)
    assert {k: st[k] for k in ("n", "on", "off_other", "off_none", "sg", "sg_off", "ov", "ov_off")} == dict(n=6, on=2, off_other=3, off_none=1, sg=4, sg_off=3, ov=1, ov_off=0)
    assert {k: st[k] for k in ("on_ok", "none_ok", "oo_l1e1", "oo_l1e0", "oo_l0e1", "oo_l0e0")} == dict(on_ok=1, none_ok=0, oo_l1e1=0, oo_l1e0=1, oo_l0e1=2, oo_l0e0=0)
    assert [w for w, _ in moved[0]] == ["a", "e", "x"] and [w for w, _ in moved[1]] == ["d", "b", "c"]          # 어긋난 b·c·e 만 옮긴다, 시각순
    ref_w = [[("a", 0.5), ("e", 0.9), ("d", 1.5), ("c", 3.9)], [("b", 2.2)]]
    s0, s1 = score_window(ref_w, hyp, "English"), score_window(ref_w, moved, "English")
    assert (s0["cp"], s1["cp"]) == (5, 4) and s0["agn"] == s1["agn"] == 1                                       # 옮기면 귀속만 바뀌고 받아쓰기(agn)는 그대로
    m = np.ones(60, bool); m[20:30] = False                                                                      # 마스크 프레임 단어(b)는 세지 않고 슬롯은 그대로
    st2, moved2 = follow_stats(hyp, act, 4, frame_mask=m)
    assert st2["skipped"] == 1 and st2["n"] == 5 and "on_ok" not in st2 and "b" in [w for w, _ in moved2[0]]


def test_follow_pool_and_line():
    from vapasr.data.cst_eval import follow_pool, follow_line
    f = lambda **k: dict(dict.fromkeys(("n", "on", "off_other", "off_none", "sg", "sg_off", "ov", "ov_off", "skipped", "on_ok", "none_ok", "oo_l1e1", "oo_l1e0", "oo_l0e1", "oo_l0e0"), 0), **k)
    rows = [dict(n_ref=10, cp=3, follow=f(n=8, on=6, off_other=2, sg=7, sg_off=2, ov=1, oo_l1e0=1, oo_l0e1=1, on_ok=5), cp_follow=2, gated=dict(text=3, begin=1)),
            dict(n_ref=10, cp=1, follow=f(n=2, on=1, off_none=1, sg=1, none_ok=1), cp_follow=1), dict(n_ref=5, cp=0)]
    p = follow_pool(rows)
    assert p["words"] == 10 and p["on"] == 70.0 and p["off_other"] == 20.0 and p["off_none"] == 10.0 and p["oo_llm_only"] == 50.0 and p["oo_act_only"] == 50.0
    assert p["err_cp_follow"] == 15.0 and p["gated_text"] == 3 and p["gated_begin"] == 1 and p["none_ok"] == 100.0
    assert "따르면 cp 15.00" in follow_line(p) and follow_pool([dict(n_ref=1, cp=0)]) is None
