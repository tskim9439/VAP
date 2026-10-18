"""Stage 3 CST 대화 직렬화(vapasr/data/cst_dialogue.py) — 두 슬롯 prefix, 화자 태그, 턴 끝 <EOT> soft label, flush 태그, 마스크·미관측,
데이터셋(창 화자 순서·3 화자 창 제외·held-out 제외)·collate, 모델 sem_only_mask·<EOT> 행 재초기화, Stage 4 네 흐름(전사·번역).
모델·tokenizer 다운로드 없이 CPU."""
import json
import numpy as np, pytest, torch, torch.nn.functional as F
from vapasr.data.cst_dialogue import (cst_head_text, build_cst_sequence, read_back, tag_flush, window_speakers, window_streams, dialogue_episodes, is_backchannel,
                                      CstDialogueDataset, collate_cst, Recover, pick_flip, retag, slot_activity, load_revive, build_cst4_sequence, read_back4,
                                      clamp_to_spans)
from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef, chunk_of
from vapasr.data.interleave import Specials
from vapasr.data.semcommit_dataset import build_interleaved_events
from vapasr.uslm.mono_data import build_mono_sequence

NEXT, EMPTY, A, B, EOT, PAD = 1000, 1001, 1002, 1003, 1004, 999
SP = Specials(next_audio=NEXT, empty_audio=EMPTY, spk=(A, B)); PRE = [1, 2, 3]


def ev(t_end):                                                     # 명시 청크 이벤트(<EOT>, end, k_min) — cst_dialogue.window_streams 와 같은 규칙
    return (EOT, t_end, chunk_of(t_end + 0.24))


def body(s):
    return [t for t, inp in zip(s["ids"], s["is_input"]) if not inp]


def test_head_text():
    assert cst_head_text(["Korean", "Korean"]) == "language <SPK_A>Korean <SPK_B>Korean<asr_text>"
    assert cst_head_text(["Korean", "English"], ["English", "Korean"]) == "language <SPK_A>Korean <SPK_B>English\ntranslate <SPK_A>English <SPK_B>Korean<asr_text>"


def test_turn_end_soft_label_at_eot():
    st = [[(10, 0.5), (11, 1.0), ev(1.1)], [(20, 1.6), (21, 2.0), ev(2.1)]]
    s = build_cst_sequence(st, {(0, 2): 0.8, (1, 2): 1.0}, 50, PRE, PAD, SP, delay=2, decision_weight=2.0)
    out = body(s); assert out.count(NEXT) == 51 and out.count(A) == 1 and out.count(B) == 1        # 50 청크 + flush 끝 1
    assert [t for t in out if t != NEXT] == [A, 10, 11, EOT, B, 20, 21, EOT]
    assert read_back(s["ids"], s["is_input"], SP, EOT) == ([[10, 11, EOT], [20, 21, EOT]], [])
    i0, i1 = s["eot_pos"]
    assert s["chunk_of"][i0 - 1] == 16 and s["chunk_of"][i1 - 1] == 29                          # 청크 = max(마지막 토큰 청크 14, ⌊1.34/0.08⌋=16)
    assert s["soft_pos"] == [i0, i1] and [s["labels_alt"][i] for i in s["soft_pos"]] == [NEXT, NEXT] and [s["soft_w"][i] for i in s["soft_pos"]] == [0.8, 1.0]
    assert s["labels"][i0] == EOT and s["pos_weight"][i0] == 2.0 and sum(1 for w in s["pos_weight"] if w) == 2


def test_eot_after_other_speaker_opens_tag_and_moves_decision():
    st = [[(10, 0.5), ev(1.1)], [(20, 1.15), (21, 2.0), ev(2.1)]]                               # B 가 A 의 EOT 청크(16)에서 먼저 말한다
    s = build_cst_sequence(st, {(0, 1): 1.0, (1, 2): 0.5}, 50, PRE, PAD, SP, delay=2)
    p16 = s["chunk_of"].index(16); assert s["ids"][p16: p16 + 6] == [PAD, B, 20, A, EOT, NEXT]
    d = s["soft_pos"][0]; assert s["ids"][d] == A and s["ids"][d + 1] == EOT and s["labels_alt"][d] == NEXT     # 결정 = A 태그(그 EOT 만을 위해 열림)
    assert s["labels"][d + 1] == EOT and s["labels_alt"][d + 1] == -100                                        # 태그 뒤 <EOT> 는 hard
    got, bad = read_back(s["ids"], s["is_input"], SP, EOT); assert got == [[10, EOT], [20, 21, EOT]] and bad == []


def test_flush_gets_speaker_tags_and_matches_mono_layout():
    st = [[(10, 0.2), (11, 0.75), ev(0.8)], [(20, 0.5), ev(0.55)]]                              # K = 12, δ4: A 의 11·EOT 는 flush
    chunks, _, srcs = build_interleaved_events([list(x) for x in st], (12 - 0.5) * 0.08, SP, delay_frames=4, with_src=True)
    assert [e for e in chunks[-1][1] if e != EMPTY] == [11, EOT]                                  # 원래 래퍼는 flush 에 태그를 넣지 않는다
    s = build_cst_sequence(st, {(0, 2): 0.8, (1, 1): 1.0}, 12, PRE, PAD, SP, delay=4)
    tail = s["ids"][s["ids"].index(EMPTY):]; assert tail == [EMPTY, A, 11, EOT, NEXT, EMPTY, NEXT]
    assert read_back(s["ids"], s["is_input"], SP, EOT) == ([[10, 11, EOT], [20, EOT]], [])
    tag_flush(chunks, srcs, 12, SP); ids, is_input, cof, n_rounds, n_flush = build_mono_sequence(chunks, 12, PRE, PAD, SP, 0)
    assert (ids, is_input, cof) == (s["ids"], s["is_input"], s["chunk_of"]) and n_rounds == s["n_rounds"] == 2   # 배치는 build_mono_sequence 와 같다
    d = s["soft_pos"][-1]; assert s["ids"][d] == EOT and s["labels_alt"][d] == NEXT                          # flush 안 EOT 도 soft


def test_unobserved_and_masked_eot_have_no_label():
    st = [[(10, 0.5), (11, 1.0), ev(1.1)], [(20, 1.6), ev(1.7)]]
    s = build_cst_sequence(st, {(0, 2): None, (1, 1): 1.0}, 50, PRE, PAD, SP, delay=2, mask_chunks={22, 24})   # B 의 20 은 청크 22, EOT 는 ⌊1.94/0.08⌋ = 24
    i0, i1 = s["eot_pos"]; assert s["labels"][i0] == -100 and s["labels"][i1] == -100 and s["soft_pos"] == []
    k22 = [i for i in range(len(s["ids"])) if s["src_at"][i] is not None and s["src_at"][i][0] == 1 and s["ids"][i] == 20][0]
    assert s["labels"][k22] == -100 and s["ids"][k22 + 1] == NEXT and s["labels"][k22 + 1] == -100   # 마스크 청크(22)의 토큰·NEXT
    s2 = build_cst_sequence(st, {(0, 2): None, (1, 1): 1.0}, 50, PRE, PAD, SP, delay=2, mask_chunks={22})
    assert s2["soft_pos"] == [s2["eot_pos"][1]]                                                  # EOT 청크(24)가 마스크 밖이면 soft 는 남는다


# ── 정체 회복 학습(태그 뒤집기)
class R:                                                           # 각본대로 고르는 rng: random() 값 목록, choice(seq) → seq[각본 index]
    def __init__(self, rand, pick): self.rand, self.pick = list(rand), list(pick)
    def random(self): return self.rand.pop(0)
    def choice(self, seq): return seq[self.pick.pop(0)]


ST3 = [[(10, 0.5), (11, 1.0), (12, 2.6), (13, 3.0)], [(20, 1.6), (21, 2.0)]]                     # run A[10 11] · B[20 21] · A[12 13]
RC = Recover(p=1.0, boundary_frac=0.5, max_lag=8, max_boundaries=2)


def flip_seq(rng):
    return build_cst_sequence(ST3, {}, 50, PRE, PAD, SP, delay=2, recover=RC, rng=rng)


def lab_of(s, tid):
    return s["labels"][s["ids"].index(tid)]


def masked_between(s, a, b):
    return all(s["labels"][p] == -100 for p in range(s["ids"].index(a), s["ids"].index(b)) if not s["is_input"][p])


def test_retag_without_flip_is_identity():
    for st in (ST3, [[(10, 0.5), (11, 1.0), ev(1.1)], [(20, 1.15), (21, 2.0), ev(2.1)]], [[(10, 0.2), (11, 0.75), ev(0.8)], [(20, 0.5), ev(0.55)]]):
        for K, dl in ((50, 2), (12, 4)):
            chunks, _, srcs = build_interleaved_events([list(x) for x in st], (K - 0.5) * 0.08, SP, delay_frames=dl, with_src=True); tag_flush(chunks, srcs, K, SP)
            assert retag(chunks, srcs, set(), SP) == (chunks, srcs)


def test_flip_missed_change_recovers_at_next_boundary_without_tag():
    s = flip_seq(R([0.0, 0.0], [0, 0, 0]))                                                     # missed(n0 = 20) → boundary(n1 = 12)
    assert s["flip"]["kind"] == "missed/boundary" and [t for t in body(s) if t != NEXT] == [A, 10, 11, 20, 21, 12, 13]
    assert lab_of(s, 20) == lab_of(s, 21) == -100 and masked_between(s, 20, 12) and lab_of(s, 12) == 12   # 목소리 A 가 돌아오면 태그 없이 A 로 이어 쓴다
    assert s["flip"]["p_end"] == s["ids"].index(12) and read_back(s["ids"], s["is_input"], SP, EOT)[0] == [[10, 11, 20, 21, 12, 13], []]


def test_flip_missed_change_recovers_inside_run_with_tag():
    s = flip_seq(R([0.0, 0.99], [0, 0, 0]))                                                    # missed(n0 = 20) → inrun(n1 = 21)
    assert s["flip"]["kind"] == "missed/inrun" and [t for t in body(s) if t != NEXT] == [A, 10, 11, 20, B, 21, A, 12, 13]
    pb = s["ids"].index(B); assert lab_of(s, 20) == -100 and s["labels"][pb] == B and s["flip"]["p_end"] == pb   # 늦게라도 바로잡는 B 태그
    assert lab_of(s, 21) == 21 and s["labels"][s["ids"].index(A, pb)] == A


def test_flip_spurious_change_masks_wrong_tag_and_recovers():
    s = flip_seq(R([0.0, 0.0], [1, 0, 0]))                                                     # spurious(n0 = 11) → boundary(n1 = 20)
    assert s["flip"]["kind"] == "spurious/boundary" and [t for t in body(s) if t != NEXT] == [A, 10, B, 11, 20, 21, A, 12, 13]
    pb = s["ids"].index(B); assert s["labels"][pb] == -100 and lab_of(s, 11) == -100 and s["flip"]["p_start"] == pb   # 헛 태그는 문맥에만
    assert lab_of(s, 20) == 20 and lab_of(s, 10) == 10                                         # 문맥이 이미 B → 태그 없이 B 의 말


def test_flip_points_are_word_starts():
    s = build_cst_sequence(ST3, {}, 50, PRE, PAD, SP, delay=2, recover=RC, rng=R([0.0, 0.0], [1, 0, 0]), word_start=lambda t: t != 11)   # 11 = 단어 중간
    assert s["flip"]["kind"] == "spurious/boundary" and s["flip"]["n0"] == (1, 1) and [t for t in body(s) if t != NEXT] == [A, 10, 11, B, 20, A, 21, 12, 13]
    assert lab_of(s, 21) == -100 and lab_of(s, 12) == 12 and lab_of(s, 11) == 11


def test_pick_flip_respects_mask_text_and_probability():
    seq = [(0, True, True), (0, True, True), (1, True, False), (1, False, True), (0, True, True)]   # 교대 지점 2 는 마스크, 3 은 EOT
    for k in range(50):
        r = pick_flip(seq, __import__("random").Random(k), RC)
        assert r is None or (seq[r[0]][1] and seq[r[0]][2] and seq[r[1]][1] and seq[r[1]][2] and r[0] < r[1])
    assert pick_flip([(0, True, True)], R([0.0], []), RC) is None                               # 원소 하나 — 고를 곳이 없다
    long_a = [(0, True, True)] * 30 + [(1, True, True)] * 3                                     # 교대 경계가 하나뿐인 창도 경계 회복을 고를 수 있다
    r = pick_flip(long_a, R([0.0], [0, 0, 0]), RC); assert r == (1, 30, "spurious/boundary")
    s = build_cst_sequence(ST3, {}, 50, PRE, PAD, SP, delay=2, recover=Recover(p=0.3), rng=R([0.5], []))
    assert s["flip"] is None and s == build_cst_sequence(ST3, {}, 50, PRE, PAD, SP, delay=2)    # 확률 밖이면 원래 시퀀스


# ── 데이터셋(가짜 tokenizer · 합성 wav)
sf = pytest.importorskip("soundfile")
from test_dialogue_dataset import FakeTok  # noqa: E402


def make(tmp_path, third=False):
    SR = 16000; dur = 60.0; p = str(tmp_path / "st.wav"); t = np.arange(int(dur * SR)) / SR
    sf.write(p, np.stack([0.3 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 660 * t), 0.3 * np.sin(2 * np.pi * 880 * t)], 1).astype(np.float32), SR)
    utts = []
    for k in range(10):                                       # x: 0–2, 6–8, …  y: 3–5, 9–11, … (1 s 무음 사이, 교대)
        s = k * 6.0; utts.append(Utterance("x", s, s + 2.0, "ab", "ab", tokens=[(11, s + 1.0), (12, s + 2.0)], utt_id=f"x{k}"))
        utts.append(Utterance("y", s + 3.0, s + 5.0, "cd", "cd", tokens=[(13, s + 4.0), (14, s + 5.0)], utt_id=f"y{k}"))
    if third: utts.append(Utterance("z", 42.2, 42.8, "e", "e", tokens=[(15, 42.6)], utt_id="z0"))
    spk = ["x", "y"] + (["z"] if third else [])
    d = Dialogue(conv_id="c1", corpus="test", lang="Korean", split="train", duration_s=dur, speakers=spk, utterances=utts,
                 channels={s: ChannelRef(path=f"{p}#ch{i}") for i, s in enumerate(spk)})
    q = str(tmp_path / "test.refined.dialogues.jsonl"); open(q, "w").write(d.to_json() + "\n"); return q


def test_dataset_sequences_are_consistent(tmp_path):
    ds = CstDialogueDataset([make(tmp_path)], FakeTok(), delays=(2, 4), window_s=(20.0, 20.0), hop_s=5.0, seed=1, decision_weight=2.0)
    assert len(ds) > 0 and ds.stats["cst_dropped_speakers"] == 0
    for i in range(len(ds)):
        for dl in (2, 4): assert ds.target_problems(i, dl) == [], (ds.items[i], dl)
        s = ds.cst_sequence(i, 4); cid, t0, L = ds.items[i]
        first = min((max(u.start, t0), u.speaker) for u in ds.dlgs[cid].utterances if u.end > t0 and u.start < t0 + L)[1]
        assert s["speakers"][0] == first                                                          # 먼저 말한 화자 = A
        assert s["ids"][: len(ds._pre)] == ds._pre and s["ids"][len(ds.cst_prefix("Korean", 4)) - 1] == ds.delay_ids[4]
        assert all(s["soft_w"][d] == 1.0 for d in s["soft_pos"])                                  # 교대(다른 화자가 1 s 뒤 시작) → shift = 1.0
    eps = dialogue_episodes(ds.dlgs["c1"]); assert eps[-1].p_end is None and eps[-1].outcome == "unobserved"   # 대화 끝 3 s 안 → 미관측
    b = [ds[i] for i in range(3)]; c = collate_cst(b)
    assert c["labels_alt"].shape == c["ids"].shape == c["soft_w_full"].shape == c["pos_weight"].shape and torch.equal(c["sem_only_mask"], c["mask"].bool())
    assert c["n_soft"] == sum(x["n_soft"] for x in b) and int((c["labels_alt"] != -100).sum()) == c["n_soft"]
    from vapasr.hf.packing import pack_batch
    p = pack_batch(c); assert p["sem_only_mask"].shape == p["ids"].shape == p["labels_alt"].shape


def test_dataset_recover_flips_only_with_rng(tmp_path):
    import random
    q = make(tmp_path); base = CstDialogueDataset([q], FakeTok(), delays=(2, 4), window_s=(20.0, 20.0), hop_s=5.0, seed=1, decision_weight=2.0)
    ds = CstDialogueDataset([q], FakeTok(), delays=(2, 4), window_s=(20.0, 20.0), hop_s=5.0, seed=1, decision_weight=2.0, recover=Recover(p=1.0))
    kinds = set()
    for i in range(len(ds)):
        assert ds.cst_sequence(i, 4)["ids"] == base.cst_sequence(i, 4)["ids"]                     # rng 없음(평가·점검) = 뒤집지 않음
        s0 = base.cst_sequence(i, 2); assert retag(s0["chunks"], s0["srcs"], set(), ds.csp) == (s0["chunks"], s0["srcs"])
        for k in range(6):
            for dl in (2, 4):
                s = ds.cst_sequence(i, dl, rng=random.Random(100 * i + k)); assert ds.problems_of(s) == [], (ds.items[i], dl, k, s["flip"] and s["flip"]["kind"])
                if s["flip"]: kinds.add(s["flip"]["kind"])
    assert kinds == {"missed/boundary", "missed/inrun", "spurious/boundary", "spurious/inrun"}
    random.seed(0); b = [ds[i] for i in range(3)]; c = collate_cst(b); assert c["n_flip"] == sum(x["n_flip"] for x in b) > 0
    with pytest.raises(ValueError): CstDialogueDataset([q], FakeTok(), delays=(4,), recover=Recover(p=1.5))


def test_slot_activity_labels_and_collate(tmp_path):
    q = make(tmp_path); ds = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1, spk_activity=True)
    eps = ds.eps["c1"]; act = slot_activity(eps, ["y", "x"], 2.5, 20.0, 250)                   # 2.5 s 시작: y(3–5 s) 가 A, x(6–8 s) 가 B
    k = lambda t: int((t - 2.5) / 0.08)
    assert act[k(4.0)] == [1, 0] and act[k(7.0)] == [0, 1] and act[k(5.5)] == [0, 0] and act[0] == [0, 0] and len(act) == 250
    assert slot_activity(eps, ["x"], 0.0, 20.0, 250)[int(4.0 / 0.08)] == [0, 0]                # 슬롯에 없는 화자(y)는 0
    b = [ds[i] for i in range(3)]; assert all(x["activity"].shape == (x["K"], 2) for x in b)
    c = collate_cst(b); Km = max(x["K"] for x in b); assert c["activity"].shape == (3, Km, 2) and c["activity_mask"].shape == (3, Km)
    assert all(int(c["activity_mask"][i].sum()) == b[i]["K"] for i in range(3)) and torch.equal(c["activity"][0, : b[0]["K"]], b[0]["activity"])
    from vapasr.hf.packing import pack_batch
    p = pack_batch(c); assert p["activity"].shape[:2] == p["activity_mask"].shape and "activity" not in collate_cst([CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1)[0]])


def test_dataset_hard_eot_mode(tmp_path):
    q = make(tmp_path)
    soft = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1)
    hard = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1, eot_mode="hard")
    for i in range(len(soft)):
        a, b = soft.cst_sequence(i, 4), hard.cst_sequence(i, 4)
        assert a["soft_pos"] == b["soft_pos"] and [b["soft_w"][d] for d in b["soft_pos"]] == [float(a["soft_w"][d] >= 0.5) for d in a["soft_pos"]]
    with pytest.raises(ValueError): CstDialogueDataset([q], FakeTok(), delays=(4,), eot_mode="x")


def test_dataset_drops_three_speaker_windows_and_excludes(tmp_path):
    q = make(tmp_path, third=True)
    ds = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1)
    assert ds.stats["cst_dropped_speakers"] > 0 and all(not (t0 < 42.8 and t0 + L > 42.2) for _, t0, L in ds.items)
    assert len(CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, exclude={"c1"})) == 0
    ds2 = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), windows=[("c1", 2.5, 20.0), ("nope", 0.0, 20.0)])
    assert ds2.items == [("c1", 2.5, 20.0)] and ds2.cst_sequence(0, 4)["speakers"][0] == "y"     # 2.5 s 시작: y(3 s) 가 먼저 → A
    ds3 = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), windows=[("c1", 36.0, 2.5)])   # x 만 말하는 창(36–38.5 s)
    assert ds3.cst_sequence(0, 4)["speakers"] == ["x"] and ds3.target_problems(0, 4) == []


def test_backchannel_option_changes_outcome_and_drops_its_eot():
    u = [Utterance("x", 0.0, 3.0, tokens=[(11, 1.0), (12, 3.0)], utt_id="x0"), Utterance("y", 3.2, 3.6, tokens=[(13, 3.5)], utt_id="y0"),
         Utterance("x", 4.5, 7.0, tokens=[(14, 6.0)], utt_id="x1"), Utterance("y", 7.5, 9.0, tokens=[(15, 8.0), (16, 9.0)], utt_id="y1")]
    d = Dialogue(conv_id="b", corpus="t", lang="German", split="train", duration_s=20.0, speakers=["x", "y"], utterances=u)
    e0 = dialogue_episodes(d); assert [e.outcome for e in e0][:2] == ["both", "shift"]                      # 맞장구(y 0.4 s)가 상대 발화로 잡힌다
    e1 = dialogue_episodes(d, (1.0, 3)); assert [e.outcome for e in e1][:2] == ["hold_long", "backchannel"] and e1[1].p_end is None
    assert is_backchannel(e1[1], (1.0, 3)) and not is_backchannel(e1[0], (1.0, 3))
    st, eot_p, w = window_streams(e1, ["x", "y"], 0.0, 20.0, EOT)
    assert w["eot_backchannel"] == 1 and st[1][:1] == [(13, 3.5)] and all(x[0] != EOT for x in st[1][:2])   # 맞장구 전사는 남고 EOT 만 없다


# ── 격리 발화 되살리기(revive)
def test_revive_masks_text_but_keeps_tags_and_eot():
    st = [[(10, 0.5), (11, 1.0), ev(1.1), (12, 2.6), (13, 3.0)], [(20, 1.6), (21, 2.0), ev(2.1)]]   # B 의 20·21 = 되살린 발화(1.3–2.0 s)
    eot = {(0, 2): 1.0, (1, 2): 1.0}; base = build_cst_sequence(st, eot, 50, PRE, PAD, SP, delay=2)
    assert build_cst_sequence(st, eot, 50, PRE, PAD, SP, delay=2, lex_mask=(), lex_chunks=()) == base           # 끄면 그대로
    lk = set(range(chunk_of(1.3, 2), chunk_of(2.0, 2) + 1))
    s = build_cst_sequence(st, eot, 50, PRE, PAD, SP, delay=2, lex_mask={(1, 0), (1, 1)}, lex_chunks=lk)
    assert s["ids"] == base["ids"] and s["lex"] == {(1, 0), (1, 1)}
    assert lab_of(s, 20) == lab_of(s, 21) == -100 and lab_of(s, 10) == 10 and lab_of(s, 12) == 12              # 전사만 입력 문맥으로
    pb = s["ids"].index(B); pa = s["ids"].index(A, pb); assert s["labels"][pb] == B and s["labels"][pa] == A   # 교대 태그는 학습
    assert all(s["labels"][p] == EOT for p in s["eot_pos"]) and s["soft_pos"] == base["soft_pos"]              # <EOT> 결정도 학습
    nxt = [p for p, t in enumerate(s["ids"]) if t == NEXT and not s["is_input"][p]]
    assert all((s["labels"][p] == -100) == (s["chunk_at"][p] in lk) for p in nxt)                               # <NEXT_AUDIO> 는 방출 구간 안만
    assert {s["ids"][p] for p in range(len(s["ids"])) if s["labels"][p] != base["labels"][p]} == {20, 21, NEXT}
    import random
    for k in range(40):                                                                         # 뒤집기 시작·회복 지점은 되살린 원소가 아니다
        f = build_cst_sequence(ST3, {}, 50, PRE, PAD, SP, delay=2, recover=RC, rng=random.Random(k), lex_mask={(1, 0), (1, 1)})["flip"]
        assert f is None or (f["n0"] not in {(1, 0), (1, 1)} and f["n1"] not in {(1, 0), (1, 1)})


def make_revive(tmp_path):
    """make() 대화에서 y1(9–11 s)·x3(18–20 s)을 정제 단계처럼 격리(asr_disagree: 전사·토큰 지움)하고 그 토큰을 되살리기 재료로 돌려준다.
    y2 는 다른 사유가 섞여(anon) 되살리지 않고, y3 은 재료에 없다(정렬 없음). x5(30–32 s)는 재료에 있어도 결손 조각(meta.missing — 오디오 없음)이라 되살리지 않는다."""
    q = make(tmp_path); d = Dialogue.from_json(open(q).read()); keep = {}
    for i, u in enumerate(d.utterances, 1):
        if u.utt_id in ("y1", "x3", "y2", "y3", "x5"):
            if u.utt_id in ("y1", "x3", "x5"): keep[u.utt_id] = list(u.tokens)
            if u.utt_id == "x5": d.meta["missing"] = [i]
            u.flags = ["asr_disagree", "anon"] if u.utt_id == "y2" else ["asr_disagree"]; u.text = ""; u.tokens = None
    open(q, "w").write(d.to_json() + "\n"); return q, {"c1": keep}


def test_dataset_revive_restores_turn_tags_inside_masked_turns(tmp_path):
    q, rv = make_revive(tmp_path)
    off = CstDialogueDataset([q], FakeTok(), delays=(2, 4), window_s=(20.0, 20.0), hop_s=5.0, seed=1)
    on = CstDialogueDataset([q], FakeTok(), delays=(2, 4), window_s=(20.0, 20.0), hop_s=5.0, seed=1, revive=rv)
    assert on.items == off.items and on.stats["cst_revived"] == 2 and on.stats["cst_revive_missing"] == 1 and "cst_revived" not in off.stats
    assert set(on.revived["c1"]) == {"y1", "x3"}                                                # x5 = 결손 조각(오디오 없음)
    for i in range(len(on)):
        for dl in (2, 4): assert on.target_problems(i, dl) == [] and off.target_problems(i, dl) == [], (on.items[i], dl)
    w = [("c1", 5.5, 20.0)]                                                                     # x1 · y1(되살림) · x2 · y2(마스크) · x3(되살림) · y3(마스크) · x4
    a = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), windows=w)
    b = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), windows=w, revive=rv)
    s0, s = a.cst_sequence(0, 4), b.cst_sequence(0, 4); K = s["K"]
    m0, m1 = a.mask_chunks(a.dlgs["c1"], 5.5, 20.0, K, 4), b.mask_chunks(b.dlgs["c1"], 5.5, 20.0, K, 4)
    assert m1 < m0 and chunk_of(10.0 - 5.5) in m0 and chunk_of(10.0 - 5.5) not in m1 and chunk_of(16.0 - 5.5) in m1   # y1 은 마스크에서 빠지고 y2 는 남는다
    rev = [p for p, x in enumerate(s["src_at"]) if x is not None and (x[0], x[1]) in s["lex"]]
    assert len(rev) == 4 and sorted(s["ids"][p] for p in rev) == [11, 12, 13, 14] and all(s["labels"][p] == -100 for p in rev)
    tags = lambda z: [(z["ids"][p], z["labels"][p]) for p in range(len(z["ids"])) if not z["is_input"][p] and z["ids"][p] in b.spk_ids]
    assert [t for t, _ in tags(s0)] == [b.spk_ids[0]]                                           # 끄면 y 의 말이 텍스트에 없어 교대가 하나도 없다
    assert tags(s) == [(b.spk_ids[0],) * 2, (b.spk_ids[1],) * 2, (b.spk_ids[0],) * 2]           # y1 앞 B, x2 앞 A 태그를 학습한다
    assert s["n_eot"] == s0["n_eot"] + 2 and b.problems_of(s) == [] and b.est_lens()[0] > a.est_lens()[0]   # y1·x3 의 <EOT>
    assert collate_cst([b[0]])["n_rev"] == 4 and collate_cst([a[0]])["n_rev"] == 0
    p = tmp_path / "test.revive.jsonl"; p.write_text(json.dumps({"conv_id": "c1", "utts": {k: [list(t) for t in v] for k, v in rv["c1"].items()}}) + "\n")
    assert load_revive([str(p)]) == rv


# ── Stage 4: 두 화자 × (전사·번역) 네 흐름
TR = 1005


def test_cst4_tags_translation_and_eot_after_translation():
    st = [[(10, 0.5), (11, 1.0)], [(20, 1.6), (21, 2.0)], [(30, 0.5), (31, 1.0), ev(1.1)], [(40, 2.0), ev(2.1)]]   # 흐름 2·3 = A·B 번역, <EOT> 는 번역 꼬리 뒤
    s = build_cst4_sequence(st, {(2, 2): 0.8, (3, 1): 1.0}, 50, PRE, PAD, SP, TR, delay=2)
    assert [t for t in body(s) if t != NEXT] == [A, 10, TR, 30, A, 11, TR, 31, EOT, B, 20, 21, TR, 40, EOT]   # 같은 청크 안은 (시각, 흐름) — 전사가 먼저
    assert read_back4(s["ids"], s["is_input"], SP, TR, EOT) == ([[10, 11], [20, 21], [30, 31], [40]], [1, 1], [])
    i0, i1 = s["eot_pos"]; assert s["chunk_of"][i0 - 1] == 16 and s["chunk_of"][i1 - 1] == 29 and s["n_tr"] == 3          # 청크 = max(번역 꼬리 14, ⌊1.34/0.08⌋)
    assert s["soft_pos"] == [i0, i1] and [s["labels_alt"][d] for d in s["soft_pos"]] == [NEXT, NEXT] and s["soft_w"][i0] == 0.8
    assert all(s["labels"][p] == t for p, t in enumerate(s["ids"]) if not s["is_input"][p])                    # 마스크가 없으면 출력 전부 학습


def test_cst4_eot_decision_with_opened_tag_and_trans_after_eot():
    st = [[(10, 0.5)], [(20, 0.65)], [(30, 0.5), ev(0.6)], []]                                    # B 가 A 의 <EOT> 청크(10)에 먼저 말한다 → A 태그를 그 <EOT> 만을 위해 연다
    s = build_cst4_sequence(st, {(2, 1): 1.0}, 50, PRE, PAD, SP, TR, delay=2)
    d = s["soft_pos"][0]; assert s["ids"][d - 2: d + 3] == [B, 20, A, EOT, NEXT] and s["labels_alt"][d] == NEXT
    st = [[(10, 0.1)], [(20, 0.7)], [(30, 0.1), (EOT, 0.2, 12), (35, 0.75)], []]                 # flush: B 20 → A <EOT> → A 번역 35(<TRANS>) — A 태그는 어차피 필요했다
    s = build_cst4_sequence(st, {(2, 1): 0.5}, 12, PRE, PAD, SP, TR, delay=4)
    tail = s["ids"][s["ids"].index(EMPTY):]; assert tail[:8] == [EMPTY, B, 20, A, EOT, TR, 35, NEXT]
    d = s["soft_pos"][0]; assert s["ids"][d] == EOT and s["labels_alt"][d] == TR and s["soft_w"][d] == 0.5      # 결정 = <EOT>(대안 <TRANS>)
    assert read_back4(s["ids"], s["is_input"], SP, TR, EOT)[1:] == ([1, 0], [])


def test_cst4_input_only_translation_and_missing_rows():
    st = [[(10, 0.5), (11, 1.0)], [(20, 1.6), (21, 2.0), ev(2.1)], [(30, 0.5), (31, 1.0), ev(1.1)], []]
    eot = {(2, 2): 1.0, (1, 2): 1.0}; base = build_cst4_sequence(st, eot, 50, PRE, PAD, SP, TR, delay=2)
    s = build_cst4_sequence(st, eot, 50, PRE, PAD, SP, TR, delay=2, tr_input={(2, 0), (2, 1)})       # A 번역 = 되살린 발화의 번역(입력만)
    assert s["ids"] == base["ids"] and lab_of(s, 30) == lab_of(s, 31) == -100 and lab_of(s, 10) == 10
    trs = [p for p, t in enumerate(s["ids"]) if t == TR]; assert all(s["labels"][p] == -100 for p in trs)      # 입력만 번역을 연 <TRANS> 도
    assert [s["labels"][p] for p in s["eot_pos"]] == [EOT, EOT] and s["soft_pos"] == base["soft_pos"]        # <EOT> 결정은 학습
    mk = set(range(chunk_of(1.3, 2), chunk_of(2.0, 2) + 1))                                     # B(1.3–2.0 s) 행에 번역이 없다
    m = build_cst4_sequence(st, eot, 50, PRE, PAD, SP, TR, delay=2, miss_chunks=mk, miss_eot={(1, 2)})
    assert lab_of(m, 20) == 20 and lab_of(m, 21) == 21 and m["labels"][m["ids"].index(B)] == -100           # 전사는 학습, 태그·NEXT 는 가린다
    nxt = [p for p, t in enumerate(m["ids"]) if t == NEXT and not m["is_input"][p]]
    assert all((m["labels"][p] == -100) == (m["chunk_at"][p] in mk) for p in nxt) and m["labels"][m["eot_pos"][1]] == -100 and len(m["soft_pos"]) == 1
    assert read_back4([TR, 30], [False, False], SP, TR, EOT)[2] == ["trans_untagged", "untagged"]


def test_cst4_eot_waits_for_transcript_tail():
    st = [[(10, 0.5), (11, 1.0)], [], [(30, 0.5), ev(1.0)], []]                                   # 번역은 첫 단어에서 끝났다 — <EOT>(번역 흐름)는 전사 꼬리 11 을 기다려야 한다
    raw = build_cst4_sequence(st, {(2, 1): 1.0}, 50, PRE, PAD, SP, TR, delay=4)
    assert [t for t in body(raw) if t != NEXT] == [A, 10, TR, 30, EOT, A, 11] and raw["eot_moved"] == 0      # 같은 흐름 앞 원소만 기다리면 말 끝 단어보다 먼저
    s = build_cst4_sequence(st, {(2, 1): 1.0}, 50, PRE, PAD, SP, TR, delay=4, eot_tail={(2, 1): [(0, 1), (2, 0)]})
    assert [t for t in body(s) if t != NEXT] == [A, 10, TR, 30, A, 11, EOT] and s["eot_moved"] == 1
    p = s["eot_pos"][0]; assert s["chunk_at"][p] == s["chunk_at"][p - 1] == chunk_of(1.0, 4) and s["soft_pos"] == [p] and s["labels_alt"][p] == NEXT
    assert read_back4(s["ids"], s["is_input"], SP, TR, EOT) == ([[10, 11], [], [30], []], [1, 0], [])


def test_clamp_to_spans():
    sp = [(1.0, 2.0), (2.5, 3.0)]
    assert [clamp_to_spans(t, sp) for t in (0.5, 1.0, 1.7, 2.2, 2.5, 2.9, 3.4)] == [1.0, 1.0, 1.7, 2.5, 2.5, 2.9, 3.0]   # 쉼(2.0–2.5)은 다음 조각 시작


def make4(tmp_path):
    """make() 대화 + 행 번역. x4 를 두 조각(x4#0 24–25 s, x4#1 25.5–26.4 s — 0.5 s 쉼이라 episode 둘)으로 나누고, y1 은 격리(되살리기 재료),
    y2 는 번역이 없다(판정 탈락). 번역 토큰 = 행 원문 단어 끝 시각(x: 31·32, y: 41·42, 되살린 y1: 45·46).
    x6 은 0.1 s 쉼의 두 조각(같은 episode)이고 둘째 조각 토큰 시각 36.95 가 조각 시작 37.0 보다 앞선다(전사는 37.0 으로 당겨진다).
    y3 의 번역은 첫 단어(22 s)에서 끝난다(<EOT> 가 전사 꼬리 23 s 를 기다려야 한다)."""
    q = make(tmp_path); d = Dialogue.from_json(open(q).read()); us = []; rv = {}
    for u in d.utterances:
        if u.utt_id == "x4":
            us += [Utterance("x", 24.0, 25.0, "", "", tokens=[(11, 24.5)], utt_id="x4#0"), Utterance("x", 25.5, 26.4, "", "", tokens=[(12, 26.0)], utt_id="x4#1")]; continue
        if u.utt_id == "x6":
            us += [Utterance("x", 36.0, 36.9, "", "", tokens=[(11, 36.5)], utt_id="x6#0"), Utterance("x", 37.0, 38.0, "", "", tokens=[(12, 36.95)], utt_id="x6#1")]; continue
        if u.utt_id == "y1": rv[u.utt_id] = list(u.tokens); u.flags = ["asr_disagree"]; u.text = ""; u.tokens = None
        us.append(u)
    d.utterances = us; open(q, "w").write(d.to_json() + "\n")
    tr, tri = {}, {}
    for u in us:
        b = u.utt_id.split("#")[0]; rid = f"c1|{u.speaker}|{b}"
        if b == "y2" or rid in tr: continue
        if b == "x4": tr[rid] = {"English": (np.array([31, 32]), np.array([24.5, 26.0]))}
        elif b == "x6": tr[rid] = {"English": (np.array([31, 32]), np.array([36.5, 36.95]))}
        elif b == "y3": tr[rid] = {"English": (np.array([41, 42]), np.array([22.0, 22.0]))}
        elif b == "y1": tri[rid] = {"English": (np.array([45, 46]), np.array([u.start + 1.0, u.end]))}
        else: tr[rid] = {"English": (np.array([31, 32] if u.speaker == "x" else [41, 42]), np.array([u.start + 1.0, u.end]))}
    return q, {"c1": rv}, tr, tri


def tok4():
    t = FakeTok(); t.add_tokens(["<TRANS>"]); return t


def test_dataset_stage4_translation_mode(tmp_path):
    import random
    q, rv, tr, tri = make4(tmp_path)
    s3 = CstDialogueDataset([q], tok4(), delays=(2, 4), window_s=(20.0, 20.0), hop_s=5.0, seed=1, revive=rv)
    ds = CstDialogueDataset([q], tok4(), delays=(2, 4), window_s=(20.0, 20.0), hop_s=5.0, seed=1, revive=rv, trans=tr, trans_input=tri, ast_frac=0.5)
    assert ds.items == s3.items and ds.trans_on and ds.trans_id == ds.tok.convert_tokens_to_ids("<TRANS>")
    rows = ds._rows("c1"); assert rows[("x", "x4")]["eps"] == [e.ep_id for e in ds.eps["c1"] if e.utt_ids[0].startswith("x4")] and len(rows[("x", "x4")]["eps"]) == 2
    assert rows[("y", "y1")]["kind"] == "r" and rows[("x", "x0")]["kind"] == "n"
    for i in range(len(ds)):
        for dl in (2, 4):
            assert ds.cst_sequence(i, dl)["ids"] == s3.cst_sequence(i, dl)["ids"]                 # rng·tgt 없음 = Stage 3 형식
            s = ds.cst_sequence(i, dl, tgt="English"); assert ds.problems_of(s) == [], (ds.items[i], dl, ds.problems_of(s))
            cid, t0, L = ds.items[i]; A_, B_ = (s["streams"][2], s["streams"][3]) if s["speakers"][0] == "x" else (s["streams"][3], s["streams"][2])
            pieces = lambda b: sorted((u.start, u.end) for u in ds.dlgs["c1"].utterances if u.utt_id.split("#")[0] == b)
            want = sorted(round(t - t0, 3) for b, ts in ((r.split("|")[2], v["English"][1]) for r, v in tr.items() if r.startswith("c1|x|"))
                          for t in (clamp_to_spans(float(x), pieces(b)) for x in ts) if t0 <= t <= t0 + L and pieces(b)[0][0] >= t0 - 1e-6)
            assert sorted(round(x[1], 3) for x in A_ if len(x) == 2) == want                      # x 번역: 행마다 한 번(x4 두 조각 → 31 은 앞 episode, 32 는 뒤)
            if t0 <= 36.0 and t0 + L >= 38.0:                                                     # x6: 번역 32(36.95 → 37.0)는 전사 12(37.0) 뒤
                at = {(q[0], q[1]): p for p, q in enumerate(s["src_at"]) if q is not None}; sx = s["speakers"].index("x")
                pt = [at[(2 + sx, j)] for j, x in enumerate(s["streams"][2 + sx]) if len(x) == 2 and abs(x[1] - (37.0 - t0)) < 1e-6]
                pw = [at[(sx, j)] for j, x in enumerate(s["streams"][sx]) if len(x) == 2 and abs(x[1] - (37.0 - t0)) < 1e-6]
                assert len(pt) == len(pw) == 1 and pt[0] > pw[0]
            if t0 <= 21.0 and t0 + L >= 23.3 and dl >= 4: assert s["win"]["eot_moved"] >= 1                 # y3: <EOT> 를 전사 꼬리 뒤로 옮겼다
            ins = [p for p, x in enumerate(s["src_at"]) if x is not None and (x[0], x[1]) in s["tr_input"]]
            assert all(s["ids"][p] in (45, 46) and s["labels"][p] == -100 for p in ins) and len(ins) == sum(1 for x in B_ if len(x) == 2 and x[0] in (45, 46))
            assert s["ids"][len(ds._pre):].index(ds.delay_ids[dl]) == len(ds.cst_prefix("Korean", dl, "English")) - len(ds._pre) - 1
    w = CstDialogueDataset([q], tok4(), delays=(4,), windows=[("c1", 11.0, 20.0)], revive=rv, trans=tr, trans_input=tri, ast_frac=1.0)   # y2(15–17 s) 번역 없음
    s = w.cst_sequence(0, 4, tgt="English"); assert w.problems_of(s) == [] and s["win"]["miss_miss"] == 1 and s["win"]["tr_input"] == 0
    e = [p for p in s["eot_pos"] if abs(s["streams"][s["src_at"][p][0]][s["src_at"][p][1]][1] - (17.0 - 11.0)) < 1e-6]
    assert len(e) == 1 and s["labels"][e[0]] == -100 and s["src_at"][e[0]][0] < 2                    # 마지막 행에 번역이 없는 episode: <EOT> 는 전사 흐름에, 학습 안 함
    c = CstDialogueDataset([q], tok4(), delays=(4,), windows=[("c1", 25.2, 20.0)], revive=rv, trans=tr, trans_input=tri, ast_frac=1.0)   # x4 두 조각 사이에서 시작
    s = c.cst_sequence(0, 4, tgt="English"); assert c.problems_of(s) == [] and s["win"]["miss_cut"] == 1 and 12 in [x[0] for x in s["streams"][0]]
    assert all(abs(x[1] - (26.0 - 25.2)) > 1e-6 for x in s["streams"][2])                          # 창 밖 원문에 기대는 번역(x4 의 32)은 넣지 않는다(전사는 Stage 3 규칙대로)
    assert (ds.est_lens() > s3.est_lens()).all()
    random.seed(3); b = [ds[i] for i in range(len(ds))]; assert any(x["n_tr"] > 0 for x in b) and any(x["n_tr"] == 0 for x in b)   # ast_frac 0.5: 두 형식이 섞인다
    cc = collate_cst(b); assert cc["n_tr"] == sum(x["n_tr"] for x in b)


# ── 모델: sem_only_mask · <EOT> soft · 예약 <EOT> 행 재초기화
from test_semcommit_tokens_loss import tiny_model, SEM, TURN, BASE  # noqa: E402


def test_reinit_turn_row_only_when_reserved():
    m, tok = tiny_model(); m.add_semantic_tokens(tok); W = m.get_input_embeddings().weight
    with torch.no_grad(): W[TURN] -= 3.0                                                         # 예약 행이 음성 학습으로 눌린 상황
    W0 = W.detach().clone(); m.add_semantic_tokens(tok, turn=True); assert torch.equal(W.detach(), W0)   # 기본: 행은 그대로(이전 동작)
    m2, tok2 = tiny_model(); m2.add_semantic_tokens(tok2); W2 = m2.get_input_embeddings().weight
    with torch.no_grad(): W2[TURN] -= 3.0
    W20 = W2.detach().clone(); m2.add_semantic_tokens(tok2, turn=True, reinit_turn=True)
    keep = torch.ones(W2.shape[0], dtype=torch.bool); keep[TURN] = False; assert torch.equal(W2.detach()[keep], W20[keep])
    mu = W20[:BASE].mean(0); assert (W2.detach()[TURN] - mu).abs().max() < 0.2 and TURN not in m2.config.blocked_ids
    W21 = W2.detach().clone(); m2.add_semantic_tokens(tok2, turn=True, reinit_turn=True); assert torch.equal(W2.detach(), W21)   # 한 번만


def test_spk_head_activity_loss_and_ddp_tie():
    from vapasr.hf import VapAsrForStreamingASR
    m, tok = tiny_model(); m.add_semantic_tokens(tok, turn=True); assert m.spk_head is None
    m.add_spk_head(2, 0.7); assert m.config.spk_slots == 2 and m.config.spk_weight == 0.7 and m.spk_head[-1].out_features == 2
    h0 = m.spk_head; m.add_spk_head(2, 0.3); assert m.spk_head is h0 and m.config.spk_weight == 0.3          # 같은 슬롯 수면 가중만
    NX = m.config.sp_ids["<NEXT_AUDIO>"]
    ids = torch.tensor([[1, 0, 50, NX, 0, 60, NX]]); is_audio = torch.tensor([[0, 1, 0, 0, 1, 0, 0]], dtype=torch.bool)
    chunk_of = torch.tensor([[-1, 0, -1, -1, 1, -1, -1]]); labels = ids.clone(); labels[is_audio] = -100; labels[:, 0] = -100
    x = dict(ids=ids, is_audio=is_audio, chunk_of=chunk_of, labels=labels, mask=torch.ones(1, 7, dtype=torch.long), feats=torch.randn(1, 1, 2, 4))
    act = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]); m.train()
    base = m(**x, next_weight=0.3); out = m(**x, activity=act, activity_mask=torch.ones(1, 2), next_weight=0.3)
    assert base.loss_act is None and out.loss_act is not None and out.act_acc is not None
    assert torch.allclose(out.loss, base.loss + 0.3 * out.loss_act, atol=1e-5)                                   # 가중 config.spk_weight
    m.zero_grad(); base.loss.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) == 0.0 for p in m.spk_head.parameters())       # 라벨 없는 행: 0 기울기로 그래프에 묶임
    m.eval(); assert torch.allclose(m(**x, next_weight=0.3).loss, base.loss)
    cfg = m.config.to_dict(); cfg.pop("model_type", None)
    from vapasr.hf import VapAsrConfig
    m2 = VapAsrForStreamingASR(VapAsrConfig(**cfg), thinker=type(m.thinker)()); assert m2.spk_head is not None and any(k.startswith("spk_head.") for k in m2.state_dict())   # 저장 config 로 다시 만들면 헤드도 생긴다


def test_forward_sem_only_mask_and_eot_soft():
    m, tok = tiny_model(turn_weight=1.5); m.add_semantic_tokens(tok, turn=True); m.eval(); NX = m.config.sp_ids["<NEXT_AUDIO>"]
    ids = torch.tensor([[1, 0, 50, TURN, NX, 0, 60, NX]]); is_audio = torch.tensor([[0, 1, 0, 0, 0, 1, 0, 0]], dtype=torch.bool)
    chunk_of = torch.tensor([[-1, 0, -1, -1, -1, 1, -1, -1]]); labels = ids.clone(); labels[is_audio] = -100; labels[:, 0] = -100
    la = torch.full_like(ids, -100); la[0, 3] = NX; sw = torch.ones(1, 8); sw[0, 3] = 0.8; so = torch.ones(1, 8, dtype=torch.bool)
    x = dict(ids=ids, is_audio=is_audio, chunk_of=chunk_of, labels=labels, mask=torch.ones(1, 8, dtype=torch.long), feats=torch.randn(1, 1, 2, 4))
    with torch.no_grad():                                                                     # tied lm_head: <SEM_END> 행을 hidden 방향으로 키워 logit 이 softmax 를 지배하게
        h0 = m.thinker.model(inputs_embeds=m.build(x["feats"], ids, is_audio, chunk_of)).last_hidden_state[0].mean(0)
        m.get_input_embeddings().weight[SEM] = 40.0 * h0 / h0.norm()
    out = m(**x, labels_alt=la, soft_w=sw, sem_only_mask=so, next_weight=0.3)
    with torch.no_grad():
        h = m.thinker.model(inputs_embeds=m.build(x["feats"], ids, is_audio, chunk_of)).last_hidden_state
        tgt = labels[:, 1:]; sel = tgt != -100; t = tgt[sel]; lg = m.thinker.lm_head(h[:, :-1][sel]).float(); lg[:, SEM] = float("-inf")
        ce = F.cross_entropy(lg, t, reduction="none"); a = la[:, 1:][sel]; w = sw[:, 1:][sel]; hs = a != -100
        ce[hs] = w[hs] * ce[hs] + (1 - w[hs]) * F.cross_entropy(lg[hs], a[hs], reduction="none")
        wt = torch.ones(len(t)); wt[t == NX] = 0.3; wt[t == TURN] = 1.5; exp = (ce * wt).sum() / wt.sum()
    assert torch.allclose(out.loss, exp, atol=1e-6) and out.n_turn == 1 and out.n_soft == 1
    assert not torch.allclose(m(**x, labels_alt=la, soft_w=sw, next_weight=0.3).loss, out.loss)            # 마스크가 실제로 반영
    bad = dict(x); bad["labels"] = labels.clone(); bad["ids"] = ids.clone(); bad["ids"][0, 2] = SEM; bad["labels"][0, 2] = SEM
    with pytest.raises(AssertionError, match="sem_only_mask"): m(**bad, sem_only_mask=so)
