"""합성 2 인 대화(vapasr/data/sim_dialogue.py) — words 행·Phase 2 대화에서 발화 단위 만들기, 말소리 구간, 실제 대화 시간 구조, 세션 합성의 불변식,
Stage 3 데이터셋에서 EOT 가 가려지는지. 오디오·모델 없이 CPU."""
import json, random
import numpy as np, pytest
from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef
from vapasr.data.sim_dialogue import (Unit, units_from_words_row, units_from_dialogue, speech_span, timing_stats, synth_session, overlap_stats,
                                      SimConfig, TimingStats, _swbd_time, persons_of, person_ids, split_points, split_unit)


def row(set_="librispeech-960-rg", segs=(("19-198-0001", 0.5, 3.0), ("19-198-0002", 4.0, 2.0)), toks=((11, 1.0), (12, 3.2), (13, 4.9), (14, 5.95)), **kw):
    return dict(id="r", set=set_, lang="English", duration_s=6.5, tokens=[list(t) for t in toks],
                segments=[dict(utt_id=u, offset_s=o, silence_before_s=0.5, dur_s=d, path=f"/x/{u}.flac", raw_text=u) for u, o, d in segs], **kw)


def test_units_from_words_row_splits_segments_and_tokens():
    us = units_from_words_row(row())
    assert [(u.key, u.order, u.uid) for u in us] == [("ls:19-198", 1.0, "19-198-0001"), ("ls:19-198", 2.0, "19-198-0002")]
    assert us[0].tokens == [(11, 0.5), (12, 2.7)] and us[1].tokens == [(13, 0.9), (14, 1.95)] and us[1].dur == 2.0 and us[0].src_offset is None
    assert units_from_words_row(row(set_="kspon-full")) == [] and units_from_words_row(row(training_eligible=False)) == []
    sw = units_from_words_row(row(set_="swbd-train", segs=(("sw02600B_83240625_8593125", 0.5, 2.712),), toks=((5, 1.0),)))
    assert sw[0].key == "swbd:sw02600B" and abs(sw[0].order - 83.240625) < 1e-6
    cv = units_from_words_row(row(set_="cv4-en", segs=(("common_voice_en_1", 0.0, 1.2),), toks=((5, 0.6),), speaker="abc"))
    assert cv[0].key == "cv:abc" and units_from_words_row(row(set_="cv4-en", segs=(("common_voice_en_1", 0.0, 1.2),), toks=((5, 0.6),))) == []
    vp = units_from_words_row(row(set_="voxpopuli-train", segs=(("20160427-0900-PLENARY-13-en_20160427-18:26:41_4", 0.9, 8.8),), toks=((5, 1.2),)))
    assert vp[0].key == "vp:20160427-0900-PLENARY-13-en_20160427-18:26:41" and vp[0].order == 4.0
    assert abs(_swbd_time("1759975", "183242", 7.266) - 175.9975) < 1e-6


def test_units_from_dialogue_path_and_pieces():
    d = Dialogue(conv_id="c", corpus="kec", lang="German", split="train", duration_s=30.0, speakers=["x", "y"],
                 utterances=[Utterance("x", 2.0, 4.0, "ab", tokens=[(1, 3.0), (2, 3.9)], utt_id="u1"), Utterance("y", 5.0, 6.0, "", tokens=None, utt_id="u2"),
                             Utterance("y", 7.0, 8.0, "cd", tokens=[(3, 7.8)], utt_id="u3", flags=["asr_disagree"]), Utterance("y", 9.0, 10.0, "ef", tokens=[(4, 9.7)], utt_id="u4")],
                 channels={"x": ChannelRef(path="/a.wav#ch0"), "y": ChannelRef(pieces=[("/p1.wav", 0.0), ("/p2.wav", 8.8)])})
    us = units_from_dialogue(d)
    assert [u.uid for u in us] == ["u1", "u4"]                                                     # 전사 없음·플래그 발화는 뺀다
    assert (us[0].path, us[0].src_offset, us[0].dur, us[0].tokens, us[0].key) == ("/a.wav#ch0", 1.9, 2.2, [(1, 1.1), (2, 2.0)], "kec|c|x")
    assert (us[1].path, us[1].src_offset, us[1].tokens) == ("/p2.wav", 0.1, [(4, 0.8)])            # 조각: 조각 안 시각


def test_persons_of_reads_aihub_ids_and_named_speakers():
    D = lambda cid, spk: Dialogue(conv_id=cid, corpus="c", lang="Korean", split="train", duration_s=1.0, speakers=spk, utterances=[])
    assert persons_of(D("71631:2_6450G2A5_6453G2A5_T1_2D08T0370C001132_029641", ["0", "1"])) == {"6450G2A5", "6453G2A5"}
    assert persons_of(D("ztc:DWAR1c", ["DW", "AR"])) == {"DW", "AR"} and persons_of(D("oto:200", ["0", "1"])) == {"oto:200"}
    nsc = Dialogue(conv_id="nsc:m023_malabo", corpus="nsc", lang="German", split="train", duration_s=1.0, speakers=["speaker", "interlocutor"], utterances=[])
    assert person_ids(nsc, "speaker") == {"nsc:m023"} and person_ids(nsc, "interlocutor") == {"nsc:malabo"}            # 역할 이름 → 세션 이름의 사람
    kec = Dialogue(conv_id="kec:rec_001_p1", corpus="kec", lang="German", split="train", duration_s=1.0, speakers=["AS_002", "IS_001"], utterances=[])
    assert person_ids(kec, "IS_001") == {"kec:IS_001"} and person_ids(D("71631:2_6450G2A5_6453G2A5_T1", ["0", "1"]), "0") == {"6450G2A5", "6453G2A5"}


def test_speech_span_trims_silence_but_protects_tokens():
    sr = 16000; x = np.zeros(int(3.0 * sr), np.float32); t = np.arange(int(1.0 * sr)) / sr
    x[int(1.0 * sr): int(2.0 * sr)] = 0.3 * np.sin(2 * np.pi * 220 * t); x += 1e-4 * np.random.RandomState(0).randn(len(x)).astype(np.float32)
    a, b = speech_span(x, [(1, 1.5), (2, 1.95)])
    assert 0.9 <= a <= 1.0 and 2.0 <= b <= 2.15
    a2, b2 = speech_span(x, [(1, 0.6), (2, 2.6)])                                                  # 토큰이 에너지 밖이면 토큰을 지킨다
    assert a2 <= 0.35 and b2 >= 2.7


def test_split_points_cut_long_units_at_pauses():
    sr = 16000; t = np.arange(int(2.5 * sr)) / sr; burst = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32); gap = np.zeros(int(0.5 * sr), np.float32)
    x = np.concatenate([gap, burst, gap, burst, gap, burst, gap]) + 1e-4 * np.random.RandomState(1).randn(int(9.5 * sr)).astype(np.float32)   # 말 2.5 s × 3, 쉼 0.5 s
    toks = [(1, 1.5), (2, 2.9), (3, 4.0), (4, 5.9), (5, 7.0), (6, 8.9)]
    parts = split_points(x, toks, max_s=4.0)
    assert len(parts) == 3 and all(abs(a - e) < 0.15 for (a, _), e in zip(parts[1:], (3.25, 6.25)))   # 쉼 가운데에서 자른다
    u = Unit(key="k", order=0, path="/p.flac", src_offset=None, dur=9.5, tokens=toks, text="t", uid="u")
    subs = split_unit(u, parts); assert [x.uid for x in subs] == ["u~0", "u~1", "u~2"] and subs[1].tokens == [(3, round(4.0 - parts[1][0], 3)), (4, round(5.9 - parts[1][0], 3))]
    assert subs[2].src_offset == parts[2][0] and split_points(x[: int(3 * sr)], toks[:2], max_s=4.0) == [(0.0, 3.0)] and split_unit(u, [(0.0, 9.5)]) == [u]


def test_synth_session_uses_expand_and_requeues_rest():
    st = TimingStats(fto=[0.3], hold=[0.5], p_hold=0.0, turn=[1.0], bc_rate=0.0, bc_pos=[0.5])
    long = [Unit(key="A", order=0, path="/A.flac", src_offset=None, dur=9.0, tokens=[(1, 2.0), (2, 5.0), (3, 8.0)], uid="L")]
    ex = lambda u: split_unit(u, [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]) if u.uid == "L" else [u]
    d = synth_session([long, fake_units("B", 10)], st, random.Random(0), 60.0, lambda u: (0.0, u.dur), "x", "sim_en", "English", cfg=SimConfig(p_hold=0.0), expand=ex)
    a_ids = [u.utt_id for u in d.utterances if u.speaker == "a"]
    assert a_ids == ["a:L~0", "a:L~1", "a:L~2"] and d.meta["split"] == 1                            # 조각이 화자 a 의 다음 턴들로 이어진다


def test_timing_stats_measures_fto_hold_and_backchannels():
    U = lambda s, a, b, n=4: Utterance(s, a, b, "w", tokens=[(9, a + (b - a) * (k + 1) / n) for k in range(n)], utt_id=f"{s}{a}")
    d = Dialogue(conv_id="t", corpus="t", lang="English", split="train", duration_s=40.0, speakers=["x", "y"],
                 utterances=[U("x", 0.0, 3.0), U("y", 2.5, 6.0), U("y", 7.0, 9.0), U("x", 9.5, 15.0), U("y", 11.0, 11.5, 1)])
    st = timing_stats([d])
    assert st.fto == [-0.5, 0.5] and st.hold == [1.0] and abs(st.p_hold - 1 / 3) < 1e-9 and st.n_bc == 1 and abs(st.bc_pos[0] - 1.5 / 5.5) < 1e-9
    assert sorted(st.turn) == [2.0, 3.0, 3.5, 5.5]


def fake_units(prefix, n, dur=2.0, short_every=0):
    out = []
    for i in range(n):
        d = 0.8 if short_every and i % short_every == 0 else dur; k = 1 if d < 1 else 4
        out.append(Unit(key=prefix, order=i, path=f"/{prefix}/{i}.wav", src_offset=None, dur=d, tokens=[(100 + j, round(d * (j + 1) / (k + 1), 3)) for j in range(k)],
                        text=f"{prefix}{i}", uid=f"{prefix}{i}", pool="t"))
    return out


def test_synth_session_invariants_and_overlap():
    st = TimingStats(fto=[-0.6, -0.3, 0.2, 0.4], hold=[0.5], p_hold=0.2, turn=[3.0, 5.0], bc_rate=2.0, bc_pos=[0.5], n_dialogues=1)
    span = lambda u: (0.0, u.dur)
    d = synth_session([fake_units("A", 30, short_every=5), fake_units("B", 30, short_every=4)], st, random.Random(0), 60.0, span, "s0", "sim_en", "English")
    assert d.meta["mask_eot"] and d.meta["synthetic"] == "sim" and d.meta["turns"] >= 5 and d.meta["bc"] >= 1 and set(d.speakers) == {"a", "b"}
    o = overlap_stats(d); assert o["self_overlap"] == 0 and o["overlap_s"] > 0 and o["changes_per_min"] > 0
    for u in d.utterances:
        pc = [p for p in d.channels[u.speaker].pieces if abs(p[1] - u.start) < 1e-6]
        assert len(pc) == 1 and abs(pc[0][3] - (u.end - u.start)) < 1e-6                         # 발화마다 같은 시각·길이의 조각
        ts = [t for _, t in u.tokens]; assert ts == sorted(ts) and u.start - 1e-6 <= ts[0] and ts[-1] <= u.end + 1e-6
    assert d.duration_s >= max(u.end for u in d.utterances) and d.utterances[0].start >= 0.5
    d2 = Dialogue.from_json(d.to_json()); assert d2.channels["a"].pieces[0] == tuple(d.channels["a"].pieces[0]) and len(d2.utterances) == len(d.utterances)
    one = synth_session([fake_units("A", 2), fake_units("B", 30)], st, random.Random(1), 60.0, span, "s1", "sim_en", "English")
    assert sum(u.speaker == "a" for u in one.utterances) <= 2 and one.duration_s < 60.0                # 한 화자가 떨어지면 끝낸다


def test_cst_dataset_masks_eot_on_synthetic(tmp_path):
    pytest.importorskip("soundfile")
    from test_dialogue_dataset import FakeTok
    from vapasr.data.cst_dialogue import CstDialogueDataset
    st = TimingStats(fto=[-0.4, 0.3], hold=[0.6], p_hold=0.2, turn=[3.0], bc_rate=1.0, bc_pos=[0.5])
    d = synth_session([fake_units("A", 40, short_every=6), fake_units("B", 40, short_every=5)], st, random.Random(2), 80.0, lambda u: (0.0, u.dur), "s2", "sim_en", "English")
    p = tmp_path / "sim_en.refined.dialogues.jsonl"; p.write_text(d.to_json() + "\n")
    ds = CstDialogueDataset([str(p)], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=10.0, seed=0, backchannel=(1.0, 3))
    assert len(ds) > 0
    for i in range(len(ds)):
        s = ds.cst_sequence(i, 4); assert ds.problems_of(s) == [] and s["soft_pos"] == [] and s["n_eot"] > 0
        assert all(s["labels"][p] == -100 for p in s["eot_pos"])                                     # <EOT> 는 문맥에만, 감독 없음
        assert any(l != -100 for l, inp in zip(s["labels"], s["is_input"]) if not inp)              # 전사·태그는 학습


# ── Stage 4: 번역 토큰을 발화와 같이 나누고 놓기
def test_words_row_and_split_carry_translation():
    tr = {"German": ([51, 52, 53, 54], [1.0, 3.2, 4.9, 5.95])}                                       # 구간 0(0.5–3.5)·1(4.0–6.0) 의 확정 시각
    us = units_from_words_row(row(), trans=tr)
    assert us[0].trans == {"German": [(51, 0.5), (52, 2.7)]} and us[1].trans == {"German": [(53, 0.9), (54, 1.95)]}
    assert units_from_words_row(row())[0].trans == {}                                               # 번역 없으면 비움
    u = Unit(key="k", order=0, path="/p", src_offset=None, dur=9.5, tokens=[(1, 1.5), (2, 2.9), (3, 4.0)], uid="u", trans={"Korean": [(7, 0.0), (8, 2.9), (9, 4.0)]})
    a, b, c = split_unit(u, [(0.0, 3.25), (3.25, 6.25), (6.25, 9.5)])[:2] + [None]
    assert a.trans == {"Korean": [(7, 0.0), (8, 2.9)]} and b.trans == {"Korean": [(9, 0.75)]}       # 첫 조각은 0 시각 확정도 받는다


def test_dialogue_units_split_row_translation_over_pieces():
    d = Dialogue(conv_id="c", corpus="kec", lang="German", split="train", duration_s=30.0, speakers=["x", "y"],
                 utterances=[Utterance("x", 2.0, 4.0, "", tokens=[(1, 3.0), (2, 3.9)], utt_id="u1#0"), Utterance("x", 4.6, 6.0, "", tokens=[(3, 5.5)], utt_id="u1#1"),
                             Utterance("y", 9.0, 10.0, "ef", tokens=[(4, 9.7)], utt_id="u4")],
                 channels={"x": ChannelRef(path="/a.wav#ch0"), "y": ChannelRef(path="/a.wav#ch1")})
    tr = {"c|x|u1": {"English": ([61, 62, 63], [3.0, 4.3, 5.5])}, "c|y|u4": {"English": ([64], [9.7])}}   # 4.3 은 조각 사이 쉼 → 다음 조각 시작 4.6
    us = {u.uid: u for u in units_from_dialogue(d, trans=tr)}
    assert us["u1#0"].trans == {"English": [(61, 1.1)]} and us["u1#1"].trans == {"English": [(62, 0.1), (63, 1.0)]} and us["u4"].trans == {"English": [(64, 0.8)]}


def test_synth_session_places_translation_like_tokens():
    st = TimingStats(fto=[0.3], hold=[0.5], p_hold=0.0, turn=[1.0], bc_rate=0.0, bc_pos=[0.5])
    A = fake_units("A", 6); B = fake_units("B", 6)
    for u in A: u.trans = {"German": [(900, u.tokens[-1][1]), (901, u.tokens[-1][1] + 5.0)]}     # 끝 뒤 확정(5 s 넘게) → 발화 끝으로 자른다
    d = synth_session([A, B], st, random.Random(0), 40.0, lambda u: (0.0, u.dur), "x", "sim_en-de", "English+German", meta=dict(spk_lang={"a": "English", "b": "German"}))
    tr = d.meta["trans"]; ua = [u for u in d.utterances if u.speaker == "a"]
    assert set(tr) == {u.utt_id for u in ua} and all(tr[u.utt_id]["German"] == [[900, u.tokens[-1][1]], [901, u.end]] for u in ua)
    assert d.meta["spk_lang"] == {"a": "English", "b": "German"}


def test_row_integrity_cont_pieces_and_empty_translation():
    d = Dialogue(conv_id="c", corpus="aihub71631", lang="Korean", split="train", duration_s=30.0, speakers=["x", "y"],
                 utterances=[Utterance("x", 2.0, 4.0, "", tokens=[(1, 3.0)], utt_id="u1#0"), Utterance("x", 4.6, 6.0, "", tokens=[(3, 5.5)], utt_id="u1#1"),
                             Utterance("y", 9.0, 10.0, "", tokens=[(4, 9.7)], utt_id="u4#0"), Utterance("y", 10.5, 10.9, "", tokens=[(5, 10.3)], utt_id="u4#1"),
                             Utterance("y", 11.0, 12.0, "", tokens=[(6, 11.9)], utt_id="u4#2")],
                 channels={"x": ChannelRef(path="/a.wav#ch0"), "y": ChannelRef(path="/a.wav#ch1")})
    tr = {"c|x|u1": {"English": ([61, 62], [5.5, 5.9])}, "c|y|u4": {"English": ([64], [11.9])}}      # 한국어처럼 행 끝에서 몰아 확정
    us = {u.uid: u for u in units_from_dialogue(d, trans=tr)}
    assert us["u1#0"].trans == {"English": []} and not us["u1#0"].cont and us["u1#1"].cont and len(us["u1#1"].trans["English"]) == 2   # 빈 목록 = 이 조각에선 확정 없음
    assert "u4#1" not in us and us["u4#0"].trans == {} and us["u4#2"].trans == {}                  # 조각이 빠진 행(토큰이 조각 앞 — 단위로 못 씀)은 번역을 떼어 낸다
    u = Unit(key="k", order=0, path="/p", src_offset=None, dur=6.0, tokens=[(1, 1.0), (2, 4.0)], uid="v", trans={"English": [(9, 4.0)]})
    p0, p1 = split_unit(u, [(0.0, 3.0), (3.0, 6.0)]); assert p0.trans == {"English": []} and not p0.cont and p1.cont and p1.trans == {"English": [(9, 1.0)]}
    st = TimingStats(fto=[0.3], hold=[0.5], p_hold=0.0, turn=[3.0], bc_rate=50.0, bc_pos=[0.5])
    B = fake_units("B", 8, short_every=1)
    for x in B: x.cont = True                                                                   # 이어지는 조각뿐이면 맞장구로 쓰지 않는다
    dd = synth_session([fake_units("A", 8), B], st, random.Random(0), 30.0, lambda u: (0.0, u.dur), "s", "sim_ko", "Korean")
    assert dd.meta["bc"] == 0


def test_interrupt_raises_overlap_and_off_is_unchanged():
    """D6 끼어들기(SimConfig.p_interrupt, 2026-10-26): 0 이면 예전과 같은 세션(난수를 더 뽑지 않음), 1 이면 교대가 모두 U(interrupt_fto) 라 겹침이 는다. 자기 겹침은 없다."""
    st = TimingStats(fto=[0.2, 0.4, 0.8], hold=[0.5], p_hold=0.2, turn=[3.0, 5.0], bc_rate=0.0, bc_pos=[0.5], n_dialogues=1)
    span = lambda u: (0.0, u.dur)
    mk = lambda cfg: synth_session([fake_units("A", 30), fake_units("B", 30), fake_units("C", 30)], st, random.Random(3), 80.0, span, "s", "sim_en", "English", cfg=cfg)
    base, off = mk(SimConfig()), mk(SimConfig(p_interrupt=0.0, interrupt_fto=(-1.5, -0.5)))
    assert base.to_json() == off.to_json() and overlap_stats(base)["overlap_s"] == 0
    hi = overlap_stats(mk(SimConfig(p_interrupt=1.0, interrupt_fto=(-1.5, -0.5)))); assert hi["overlap_s"] > 5.0 and hi["self_overlap"] == 0


def test_full_overlap_starts_with_previous_turn_and_off_is_unchanged():
    """완전 겹침(SimConfig.p_full, 2026-11-08): 0 이면 예전과 같은 세션, 1 이면 교대 턴이 앞 턴 시작 + U(full_start) 에 시작해 겹침이 크게 는다. 자기 겹침은 없다."""
    st = TimingStats(fto=[0.2, 0.4, 0.8], hold=[0.5], p_hold=0.0, turn=[3.0, 5.0], bc_rate=0.0, bc_pos=[0.5], n_dialogues=1)
    span = lambda u: (0.0, u.dur)
    mk = lambda cfg: synth_session([fake_units("A", 30), fake_units("B", 30)], st, random.Random(5), 80.0, span, "s", "sim_en", "English", cfg=cfg)
    base, off = mk(SimConfig()), mk(SimConfig(p_full=0.0, full_start=(0.0, 0.2)))
    assert base.to_json() == off.to_json()
    full = mk(SimConfig(p_full=1.0, full_start=(0.0, 0.2))); o = overlap_stats(full)
    assert o["self_overlap"] == 0 and o["overlap_s"] > overlap_stats(base)["overlap_s"] + 10.0 and full.meta["full"] >= 3
    starts = sorted(u.start for u in full.utterances)
    assert any(b_ - a_ <= 0.2 + 1e-6 for a_, b_ in zip(starts, starts[1:]))                           # 거의 함께 시작한 두 발화가 있다


def test_overlay_two_dialogues_into_four_speakers():
    """sim_overlay_dialogues(2026-11-10): 2 인 + 2 인 → 4 인, B 의 화자·시각·채널 조각이 밀리고 이름이 c·d 로 바뀐다. Dialogue 로 다시 읽힌다."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from sim_overlay_dialogues import overlay
    mk = lambda cid: dict(conv_id=cid, corpus="sim_en-ovl2", lang="English", split="train", duration_s=10.0, speakers=["a", "b"], observed_until_s=None, meta={},
                          utterances=[dict(speaker="a", start=1.0, end=2.0, text="", raw="x", tokens=[[5, 1.5]], utt_id="a:1", word_timing=None, flags=None),
                                      dict(speaker="b", start=3.0, end=4.0, text="", raw="y", tokens=[[6, 3.5]], utt_id="b:1", word_timing=None, flags=None)],
                          channels={"a": dict(path="", pieces=[["/p/a.flac", 1.0, 0.0, 1.0]]), "b": dict(path="", pieces=[["/p/b.flac", 3.0, 0.0, 1.0]])})
    d = overlay(mk("A"), mk("B"), 2.0, "sim_en-ovl3", 7)
    assert d["speakers"] == ["a", "b", "c", "d"] and d["duration_s"] == 12.0 and d["meta"]["overlay"] == ["A", "B"]
    c = [u for u in d["utterances"] if u["speaker"] == "c"][0]
    assert c["start"] == 3.0 and c["tokens"] == [[5, 3.5]] and d["channels"]["c"]["pieces"][0][1] == 3.0
    D = Dialogue.from_json(json.dumps(d)); assert len(D.utterances) == 4 and set(D.channels) == {"a", "b", "c", "d"}
