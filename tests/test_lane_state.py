"""Phase 2 lane 파서·평가 보조 — 태그/ONSET/텍스트/EOT 규약 파싱, 비정상 열(stray EOT·태그 없는 텍스트·미닫힘), 편집거리·지연·이벤트 매칭."""
from vapasr.hf.lane_state import LaneParser, edit_distance, cer, wer, token_latency, match_events
REG = {"<SPK_A>": 900, "<SPK_B>": 901, "<SPK_3>": 902, "<SPK_4>": 903, "<SPK_5>": 904, "<SPK_6>": 905, "<ONSET>": 910, "<EOT>": 911}

def test_parser_two_lanes_and_flags():
    p = LaneParser(REG, R=6)
    em = [(0, 7), (3, 900), (3, 910), (5, 900), (5, 11), (5, 12), (7, 901), (7, 910), (8, 901), (8, 21), (9, 900), (9, 911), (12, 901), (12, 911), (14, 902), (14, 911), (15, 903), (15, 31), (16, 33)]
    segs, st = p.parse(em)                                                 # (0,7): 태그 전 텍스트 → text_no_lane
    assert [(s.lane, s.k_on, s.k_eot, [t for t, _ in s.tokens]) for s in segs] == [(1, 3, 9, [11, 12]), (2, 7, 12, [21]), (4, 15, None, [31, 33])]
    assert st["stray_eot"] == 1 and st["implicit"] == 1 and st["unclosed"] == 1 and st["text_no_lane"] == 1
    assert segs[0].start == 4 * 0.08 and abs(segs[0].end - 10 * 0.08) < 1e-9

def test_parser_reopen_ignored():
    p = LaneParser(REG); segs, st = p.parse([(0, 900), (0, 910), (2, 900), (2, 910), (3, 5), (4, 900), (4, 911)])
    assert len(segs) == 1 and segs[0].k_on == 0 and segs[0].k_eot == 4 and st["reopen"] == 1

def test_metrics():
    assert edit_distance(list("abc"), list("abd")) == 1 and cer("가나 다", "가나다") == 0.0 and wer("a b c", "a b d") == 1 / 3 and cer("x", "") is None
    lat, n = token_latency([(1, 5), (2, 6), (3, 9)], [(1, 0.3), (2, 0.4), (4, 0.5)]); assert n == 2 and abs(lat[0] - (6 * 0.08 - 0.3)) < 1e-9
    assert match_events([1.0, 2.0, 9.0], [1.2, 2.5, 5.0], 0.4) == (1, 3, 3) and match_events([1.0, 2.0, 9.0], [1.2, 2.5, 5.0], 0.6) == (2, 3, 3)

def test_lane_states_and_act_close_parse():
    from vapasr.hf.lane_state import LaneStates
    ls = LaneStates(3, act_close_chunks=2, act_thr=0.5); assert ls.next_lane() == 1
    ls.open(1); ls.open(2); assert ls.next_lane() == 3; ls.open(3); assert ls.next_lane() is None
    ls.close(2, 10); ls.close(1, 20); assert ls.next_lane() == 2                       # 가장 오래전에 닫힌 HELD
    assert ls.tick(21, [0.9, 0.0, 0.1]) == [] and ls.tick(22, [0.9, 0.0, 0.1]) == [3] and ls.state[3] == "HELD" and ls.act_closed == [(3, 22)]
    p = LaneParser(REG); segs, st = p.parse([(0, 900), (0, 910), (1, 5), (6, 900), (6, 910), (7, 6), (9, 900), (9, 911)], act_closed=[(1, 4)])
    assert len(segs) == 2 and segs[0].k_act_close == 4 and segs[0].k_eot is None and abs(segs[0].end - 5 * 0.08) < 1e-9 and segs[1].k_on == 6 and segs[1].k_eot == 9 and st["act_closed"] == 1 and st["reopen"] == 0
