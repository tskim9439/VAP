"""Phase 2 lane 규약 fixture (정본 output-phase2-lane-plan §12). 실제 tokenizer 없이 정수 ID 로 검사한다."""
import pytest
from vapasr.data.dialogue import Dialogue, Utterance, Episode, build_episodes, chunk_of
from vapasr.data.lane_alloc import allocate, eot_chunk
from vapasr.data.eot_soft import assign_p_end, classify, P_END
from vapasr.data.dialogue_interleave import LaneSpecials, serialize, flatten, lane_activity, render

NEXT, EMPTY, ONSET, EOT, PAD = 900, 901, 902, 903, 999
SP = LaneSpecials(next_audio=NEXT, empty_audio=EMPTY, lanes=[801, 802, 803, 804, 805, 806], onset=ONSET, eot=EOT)
NAMES = {NEXT: "<NEXT>", EMPTY: "<EMPTY>", ONSET: "<ONSET>", EOT: "<EOT>", **{801 + i: f"<L{i+1}>" for i in range(6)}}

def utt(spk, s, e, words, uid=""):
    """단어 하나 = 토큰 하나, end_time 을 [s,e] 에 균등 배치. 토큰 id = 100 + 단어 인덱스(전역 증가 없음, 검사용)."""
    n = len(words); toks = [(100 + i, s + (e - s) * (i + 1) / n) for i in range(n)]
    return Utterance(speaker=spk, start=s, end=e, text=" ".join(words), tokens=toks, utt_id=uid)

def dlg(utts, dur=20.0, observed=None, speakers=None):
    spk = speakers or sorted({u.speaker for u in utts})
    return Dialogue(conv_id="t", corpus="test", lang="Korean", split="train", duration_s=dur, speakers=spk, utterances=utts, observed_until_s=observed)

def prep(d, R=6, policy="lazy_free", **kw):
    eps = build_episodes(d); st = allocate(eps, R=R, policy=policy, **kw); assign_p_end(eps, d.observed()); return eps, st

# ── 보고서 §4 예시: 민수·지수·철수 + 4번째 영희, R=3 ─────────────────────────────────────────
EXAMPLE = [utt("민수", 0.5, 4.0, ["안녕하세요", "저는", "김입니다"]), utt("지수", 2.0, 2.6, ["네"]),
           utt("민수", 5.0, 7.0, ["그리고요"]), utt("지수", 7.5, 11.25, ["그렇군요", "그러면"]),
           utt("철수", 8.5, 12.25, ["동의해요", "다만"]), utt("영희", 14.0, 16.25, ["잠깐만요"])]

def test_example_lane_assignment_r3():
    eps, st = prep(dlg(EXAMPLE), R=3)
    lane = {(e.speaker, round(e.start, 1)): e.lane for e in eps}
    assert lane[("민수", 0.5)] == 1 and lane[("지수", 2.0)] == 2 and lane[("민수", 5.0)] == 1          # 규칙 1: 재개는 같은 lane
    assert lane[("철수", 8.5)] == 3                                                                   # 규칙 2: FREE 우선(lane 1 은 HELD)
    assert lane[("영희", 14.0)] == 1 and st.reassigned == 1 and st.exhausted == 0                  # 규칙 3: closed 가장 오래된 lane 1
    gen = {e.speaker: e.generation for e in eps}; assert gen["민수"] == 0 and gen["영희"] == 1

def test_example_r6_no_reassignment():
    eps, st = prep(dlg(EXAMPLE), R=6)
    assert st.reassigned == 0 and {e.speaker: e.lane for e in eps}["영희"] == 4

def test_never_free_equals_lazy_free_when_n_le_r():        # fixture 20
    d = dlg(EXAMPLE); a, _ = prep(d, R=6, policy="lazy_free"); b, _ = prep(d, R=6, policy="never_free")
    assert [(e.lane, e.generation) for e in a] == [(e.lane, e.generation) for e in b]
    sa, _ = serialize(a, d.duration_s, SP); sb, _ = serialize(b, d.duration_s, SP)
    assert [[x.tid for x in em] for _, em in sa] == [[x.tid for x in em] for _, em in sb]

def test_never_free_exhausts_when_n_gt_r():
    eps, st = prep(dlg(EXAMPLE), R=3, policy="never_free")
    assert st.exhausted == 1 and [e for e in eps if e.speaker == "영희"][0].lane is None

# ── lane 상태·재배정 세부 ───────────────────────────────────────────────────────────────
def test_open_lane_not_reassignable_and_exhausted_counted():   # fixture 22
    # 세 lane 이 모두 OPEN(모두 말하는 중)일 때 4번째 시작 → 재배정 불가, lane None
    d = dlg([utt("a", 0, 5, ["x"]), utt("b", 0.5, 5, ["x"]), utt("c", 1.0, 5, ["x"]), utt("d", 2.0, 3, ["x"])])
    eps, st = prep(d, R=3); assert st.exhausted == 1 and [e for e in eps if e.speaker == "d"][0].lane is None
    chunks, sst = serialize(eps, d.duration_s, SP); assert sst.skipped_episodes == 1

def test_reassignment_requires_prev_eot_before_new_onset():  # fixture 21 + 순서 보장
    # a 가 0–2 s 에 말하고 끝난 직후 2.1 s 에 d 가 시작. R=1 이면 a 의 EOT 청크(δ=4 → ≥ 2.32 s 청크)가 d 의 ONSET 청크(2.1 s)보다 뒤 → 재배정 불가
    d = dlg([utt("a", 0, 2.0, ["x", "y"]), utt("d", 2.1, 4.0, ["z"])])
    eps, st = prep(d, R=1); assert st.exhausted == 1
    d2 = dlg([utt("a", 0, 2.0, ["x", "y"]), utt("d", 3.0, 4.0, ["z"])])
    eps2, st2 = prep(d2, R=1); assert st2.reassigned == 1 and eps2[1].lane == 1 and eps2[1].generation == 1
    chunks, _ = serialize(eps2, d2.duration_s, SP); flat = [(k, x.kind, x.lane) for k, em in chunks for x in em]
    i_eot = flat.index((eot_chunk(eps2[0], 4), "eot", 1)); i_on = [i for i, f in enumerate(flat) if f[1] == "onset" and f[2] == 1][1]
    assert i_eot < i_on

def test_same_speaker_resume_new_episode_same_lane():        # fixture 18
    d = dlg([utt("a", 0, 1.0, ["x"]), utt("a", 1.5, 2.0, ["y"])]); eps, _ = prep(d)
    assert len(eps) == 2 and eps[0].lane == eps[1].lane == 1 and eps[0].generation == eps[1].generation == 0

def test_gap_merge_below_025():
    d = dlg([utt("a", 0, 1.0, ["x"]), utt("a", 1.2, 2.0, ["y"])]); eps = build_episodes(d)
    assert len(eps) == 1 and eps[0].end == 2.0 and len(eps[0].tokens) == 2

# ── 직렬화 ────────────────────────────────────────────────────────────────────────────
def test_silence_only_blocks():                               # fixture 17
    d = dlg([utt("a", 5.0, 6.0, ["x"])], dur=8.0); eps, _ = prep(d); chunks, st = serialize(eps, 8.0, SP)
    assert all([x.kind for x in em] == ["next"] for k, em in chunks if k < chunk_of(5.0, 0)) and st.chunks == 100

def test_tags_before_onset_eot_and_on_change_only():
    d = dlg([utt("a", 0, 1.0, ["x", "y"]), utt("b", 0.5, 1.5, ["z"])]); eps, _ = prep(d); chunks, _ = serialize(eps, 3.0, SP)
    seq = [(x.kind, x.lane) for _, em in chunks for x in em if x.kind != "next"]
    # 첫 ONSET(a) 앞 태그, b 의 ONSET 앞 태그, 이후 lexical 은 lane 이 바뀔 때만 태그
    assert seq[:2] == [("tag", 1), ("onset", 1)] and ("tag", 2) in seq
    for i, (kind, lane) in enumerate(seq):
        if kind in ("onset", "eot"): assert seq[i - 1] == ("tag", lane)

def test_overlap_three_speakers_share_one_audio_token():      # fixture 19
    d = dlg([utt("a", 0, 2, ["x"]), utt("b", 0.2, 2, ["y"]), utt("c", 0.4, 2, ["z"])]); eps, _ = prep(d)
    chunks, _ = serialize(eps, 3.0, SP); f = flatten(chunks, 38, PAD, EMPTY)
    assert sum(f["is_audio"]) == 38 and {e.lane for e in eps} == {1, 2, 3}

def test_eot_position_after_last_lexical():
    d = dlg([utt("a", 0, 1.0, ["x", "y", "z"])]); eps, _ = prep(d); chunks, _ = serialize(eps, 3.0, SP)
    ks = [(k, x.kind) for k, em in chunks for x in em if x.kind in ("text", "eot")]
    assert ks[-1][1] == "eot" and ks[-1][0] == max(k for k, _ in ks) and ks[-1][0] == max(chunk_of(1.0, 4), chunk_of(1.24, 0))

def test_soft_label_and_mask():                               # fixture 24
    d = dlg([utt("a", 0, 1.0, ["x"]), utt("b", 1.2, 2.0, ["y"])], dur=6.0, observed=6.0); eps, _ = prep(d)
    assert eps[0].outcome == "shift" and eps[0].p_end == 1.0 and eps[1].outcome == "silence" and eps[1].p_end == 0.8
    chunks, _ = serialize(eps, 6.0, SP); f = flatten(chunks, 75, PAD, EMPTY, prefix_ids=[1, 2])
    assert f["labels"][:2] == [-100, -100] and all(f["labels"][i] == -100 for i, a in enumerate(f["is_audio"]) if a)
    assert len(f["soft_pos"]) == 2 and f["soft_w"] == [1.0, 0.8] and all(f["ids"][p] == EOT for p in f["soft_pos"])
    assert all(f["soft_alt"][i] == f["ids"][p + 1] for i, p in enumerate(f["soft_pos"]))
    # 미관측(EOF 근처) → mask
    d2 = dlg([utt("a", 0, 1.0, ["x"])], dur=2.0, observed=2.0); eps2, _ = prep(d2)
    assert eps2[0].outcome == "unobserved" and eps2[0].p_end is None
    f2 = flatten(*serialize(eps2, 2.0, SP)[:1], 25, PAD, EMPTY); j = f2["ids"].index(EOT); assert f2["labels"][j] == -100 and f2["soft_pos"] == []

def test_outcome_classes():
    eps = [Episode(0, "a", 0, 1.0), Episode(1, "a", 1.5, 2.0)]; assert classify(eps[0], eps, 10.0) == "hold_short"
    eps = [Episode(0, "a", 0, 1.0), Episode(1, "a", 3.0, 4.0)]; assert classify(eps[0], eps, 10.0) == "hold_long"
    eps = [Episode(0, "a", 0, 1.0), Episode(1, "b", 0.5, 2.0), Episode(2, "a", 2.5, 3.0)]; assert classify(eps[0], eps, 10.0) == "both"
    eps = [Episode(0, "a", 0, 1.0), Episode(1, "b", 5.0, 6.0)]; assert classify(eps[0], eps, 10.0) == "silence"
    assert set(P_END) == {"shift", "silence", "both", "hold_long", "hold_short", "unobserved"}

def test_future_suffix_changes_only_soft_weights():         # fixture 25
    base = [utt("a", 0, 1.0, ["x"])]
    d1 = dlg(base + [utt("b", 2.0, 3.0, ["y"])], dur=6.0); d2 = dlg(base + [utt("a", 2.0, 3.0, ["y"])], dur=6.0)
    e1, _ = prep(d1); e2, _ = prep(d2); c1, _ = serialize(e1, 6.0, SP); c2, _ = serialize(e2, 6.0, SP)
    cut = chunk_of(2.0, 0)   # 2.0 s 이전 청크의 토큰열은 동일
    assert [[x.tid for x in em] for k, em in c1 if k < cut] == [[x.tid for x in em] for k, em in c2 if k < cut]
    assert e1[0].p_end != e2[0].p_end

def test_tail_flush_uses_empty_rounds():
    d = dlg([utt("a", 0, 1.0, ["x", "y"])], dur=1.0); eps, _ = prep(d); chunks, st = serialize(eps, 1.0, SP)
    assert chunks[-1][0] >= st.chunks and any(x.kind == "eot" for x in chunks[-1][1]) and st.overflow == 0
    f = flatten(chunks, st.chunks, PAD, EMPTY); assert f["ids"].count(EMPTY) == sum(1 for k, _ in chunks if k >= st.chunks)

def test_lane_activity():
    d = dlg([utt("a", 0, 0.5, ["x"]), utt("b", 0.4, 1.0, ["y"])]); eps, _ = prep(d); act = lane_activity(eps, 13, 6)
    assert act[0] == [1, 0, 0, 0, 0, 0] and act[5] == [1, 1, 0, 0, 0, 0] and act[10] == [0, 1, 0, 0, 0, 0]

def test_max_per_chunk_carries_over_in_order():
    d = dlg([utt("a", 0, 0.08, ["w1", "w2", "w3", "w4"])], dur=2.0); eps, _ = prep(d)
    chunks, st = serialize(eps, 2.0, SP, max_per_chunk=2); texts = [x.tid for _, em in chunks for x in em if x.kind == "text"]
    assert texts == [100, 101, 102, 103] and st.overflow > 0

def test_render_roundtrip_example():
    eps, _ = prep(dlg(EXAMPLE), R=3); chunks, st = serialize(eps, 20.0, SP)
    s = render(chunks, {**NAMES, **{100 + i: f"w{i}" for i in range(10)}}, st.chunks)
    assert "<L1> <ONSET>" in s and "<L3> <ONSET>" in s and s.count("<EOT>") == 6
