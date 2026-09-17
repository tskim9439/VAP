"""평가 지표 v2 — 축 분리: lane 만 바꾼 가설은 A(인식) 불변·B 의 cp 는 전역 매핑으로 불변·lane 전환만 증가, 텍스트만 바꾼 가설은 A 만 악화. 턴·EOT·DER·CI."""
import numpy as np
from vapasr.hf.lane_metrics import utterance_assigned, pooled, cp_error, lane_der, lane_switches, event_metrics, turn_change_metrics, eot_in_turn, activity_f1, bootstrap_ci, edit_ops

REF = [dict(speaker="A", start=0.0, end=2.0, text="안녕 하세요"), dict(speaker="B", start=2.5, end=4.0, text="네 반가워요"), dict(speaker="A", start=4.5, end=6.0, text="오늘 날씨 좋네요"), dict(speaker="B", start=6.2, end=7.0, text="그러게요")]
HYP = [dict(lane=1, start=0.1, end=2.1, text="안녕 하세요"), dict(lane=2, start=2.6, end=4.1, text="네 반가워요"), dict(lane=1, start=4.6, end=6.1, text="오늘 날씨 좋네요"), dict(lane=2, start=6.3, end=7.1, text="그러게요")]

def test_edit_ops(): assert edit_ops(list("abd"), list("abc")) == (1, 0, 0) and edit_ops(list("ab"), list("abc")) == (0, 1, 0) and edit_ops(list("abcd"), list("abc")) == (0, 0, 1)

def test_perfect():
    a = utterance_assigned(REF, HYP, "cer"); c = cp_error(REF, HYP, "cer"); assert a["rate"] == 0.0 and c["rate"] == 0.0 and c["mapping"] == {1: "A", 2: "B"} and lane_switches(REF, HYP)["switches"] == 0

def test_lane_swap_only_hurts_speaker_axis():
    swapped = [dict(h, lane=(2 if h["lane"] == 1 else 1)) for h in HYP]                     # 전역 번호 뒤바뀜 → A 불변, cp 불변(매핑이 흡수), 전환 0
    assert utterance_assigned(REF, swapped, "cer")["rate"] == 0.0 and cp_error(REF, swapped, "cer")["rate"] == 0.0 and lane_switches(REF, swapped)["switches"] == 0
    mid = [dict(h) for h in HYP]; mid[2]["lane"] = 2                                         # 중간 한 구간만 lane 오배정 → A 불변, cp 악화, 전환 1(A: 1→2), DER 혼동 > 0
    assert utterance_assigned(REF, mid, "cer")["rate"] == 0.0
    c = cp_error(REF, mid, "cer"); assert c["rate"] > 0.0 and lane_switches(REF, mid)["switches"] == 1
    d = lane_der(REF, mid, c["mapping"]); assert d["conf"] > 0.0 and d["miss"] == 0.0

def test_text_error_only_hurts_recognition_axis():
    bad = [dict(h) for h in HYP]; bad[1]["text"] = "네 반갑습니다"
    a = utterance_assigned(REF, bad, "cer"); c = cp_error(REF, bad, "cer")
    assert a["rate"] > 0.0 and a["S"] + a["D"] + a["I"] == c["errors"] and lane_switches(REF, bad)["switches"] == 0 and c["mapping"] == {1: "A", 2: "B"}

def test_missing_and_spurious():
    miss = HYP[:2]; a = utterance_assigned(REF, miss, "cer"); assert a["D"] == len("오늘날씨좋네요") + len("그러게요") and a["I"] == 0
    spur = HYP + [dict(lane=3, start=8.5, end=9.0, text="음")]; a2 = utterance_assigned(REF, spur, "cer"); assert a2["I"] == 1 and a2["unassigned_units"] == 1
    c = cp_error(REF, spur, "cer"); assert c["unmatched_lanes"] == [3] and c["errors"] == 1

def test_events_turns_eot():
    ev = event_metrics([0.1, 2.7, 4.6], [0.0, 2.5, 4.5, 6.2]); assert ev["0.4"]["hit"] == 3 and ev["0.4"]["r"] == 0.75 and abs(ev["0.4"]["err_median"] - 0.1) < 1e-9
    tc = turn_change_metrics(REF, HYP); assert tc["ref"] == 3 and tc["hyp"] == 3 and tc["f1"] == 1.0 and abs(tc["latency_median"] - 0.1) < 1e-9
    e = eot_in_turn(REF, [1.0, 2.1, 5.0], {1: "A", 2: "B"}, [1, 1, 2]); assert e["count"] == 1        # 1.0 s 는 A 발화 중(오방출), 2.1 은 끝 근처, 5.0 은 lane 2→B 인데 B 는 말하지 않음
    ra = np.zeros((10, 2)); ra[:5, 0] = 1; ha = np.zeros((10, 3)); ha[:5, 1] = 1; f = activity_f1(ra, ha, {2: 0}); assert f["f1"] == 1.0
    lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4], n=200); assert lo is not None and lo <= 0.25 <= hi
