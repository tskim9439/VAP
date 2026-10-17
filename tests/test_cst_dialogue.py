"""Stage 3 CST 대화 직렬화(vapasr/data/cst_dialogue.py) — 두 슬롯 prefix, 화자 태그, 턴 끝 <EOT> soft label, flush 태그, 마스크·미관측,
데이터셋(창 화자 순서·3 화자 창 제외·held-out 제외)·collate, 모델 sem_only_mask·<EOT> 행 재초기화. 모델·tokenizer 다운로드 없이 CPU."""
import json
import numpy as np, pytest, torch, torch.nn.functional as F
from vapasr.data.cst_dialogue import (cst_head_text, build_cst_sequence, read_back, tag_flush, window_speakers, window_streams, dialogue_episodes, is_backchannel,
                                      CstDialogueDataset, collate_cst, Recover, pick_flip, retag, slot_activity)
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
