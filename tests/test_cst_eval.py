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
