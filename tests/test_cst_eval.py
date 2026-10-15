"""Stage 3 평가 도우미(vapasr/data/cst_eval.py) — 출력 해석, 화자 귀속 오류율(slot·cp·화자 무시), EOT 짝·합산. CPU."""
from vapasr.data.cst_eval import parse_emitted, score_window, match_eot, eot_tol, pool, ed

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
