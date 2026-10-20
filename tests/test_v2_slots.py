"""V2 화자 슬롯 4 개 — 모델 enable_slots(<SPK_3>·<SPK_4> = C·D: 예약 행 재초기화·차단 해제·idempotent), add_spk_head 2 → 4 확장(A·B 행 유지). CPU."""
import importlib.util, pathlib
import pytest, torch

_HERE = pathlib.Path(__file__).parent
def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", _HERE / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _mono_model():
    t = _load("test_semcommit_tokens_loss"); m, tok = t.tiny_model(); m.add_semantic_tokens(tok, turn=True); return m, tok


def test_enable_slots_reinit_unblock_idempotent():
    m, tok = _mono_model(); c = m.config; W = m.get_input_embeddings().weight
    assert {"<SPK_3>", "<SPK_4>"} <= set(c.sem_reserved) and {151717, 151718} <= set(c.blocked_ids) and c.slot_tokens == ["<SPK_A>", "<SPK_B>"]
    with torch.no_grad(): W[[151717, 151718]] -= 3.0                                           # 예약 행이 full-vocab softmax 음성으로 밀려난 상황
    W0 = W.detach().clone(); ids = m.enable_slots(tok, 4)
    assert ids == [151707, 151708, 151717, 151718] and c.slot_tokens == ["<SPK_A>", "<SPK_B>", "<SPK_3>", "<SPK_4>"]
    assert not {151717, 151718} & set(c.blocked_ids) and m.blocked.tolist() == c.blocked_ids and not {"<SPK_3>", "<SPK_4>"} & set(c.sem_reserved)
    assert {"<SPK_5>", "<SPK_6>"} <= set(c.sem_reserved) and {151719, 151720} <= set(c.blocked_ids)    # 쓰지 않는 예약은 그대로
    mu = W0[:151705].mean(0); assert all((W.detach()[r] - mu).abs().max() < 0.2 for r in (151717, 151718))
    keep = torch.ones(W.shape[0], dtype=torch.bool); keep[[151717, 151718]] = False; assert torch.equal(W.detach()[keep], W0[keep])   # A·B 와 나머지 행은 그대로
    W1 = W.detach().clone(); assert m.enable_slots(tok, 4) == ids and torch.equal(W.detach(), W1)  # 반복 호출은 아무것도 안 바꾼다
    with pytest.raises(AssertionError): m.enable_slots(tok, 5)


def test_spk_head_expands_keeping_ab():
    m, _ = _mono_model(); m.add_spk_head(2, 0.5); h = m.spk_head; w2 = h[-1].weight.detach().clone(); b2 = h[-1].bias.detach().clone(); w0 = h[0].weight.detach().clone()
    m.add_spk_head(4, 0.7); assert m.config.spk_slots == 4 and m.config.spk_weight == 0.7 and m.spk_head[-1].out_features == 4
    assert torch.equal(m.spk_head[-1].weight[:2], w2) and torch.equal(m.spk_head[-1].bias[:2], b2) and torch.equal(m.spk_head[0].weight, w0)
    assert torch.allclose(m.spk_head[-1].weight[2], w2.mean(0)) and torch.allclose(m.spk_head[-1].weight[3], w2.mean(0))
    x = torch.randn(3, m.config.hidden_size); assert m.spk_head(x).shape == (3, 4)
    m.add_spk_head(4, 0.5); assert m.spk_head[-1].out_features == 4
    with pytest.raises(AssertionError, match="줄이지"): m.add_spk_head(2)
    sd = m.state_dict(); from vapasr.hf import VapAsrForStreamingASR
    r = VapAsrForStreamingASR(m.config, thinker=_load("test_semcommit_tokens_loss")._Thinker()); r.load_state_dict(sd, strict=True)   # 4 슬롯 config 로 다시 만들면 같은 모양


# ── 시퀀스 N 슬롯(vapasr/data/cst_dialogue.py)
from vapasr.data.cst_dialogue import (build_cst_sequence, read_back, build_cst4_sequence, read_back4, slot_activity, CstDialogueDataset, collate_cst,
                                      cst_head_text, dialogue_episodes)
from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef, chunk_of
from vapasr.data.interleave import Specials

NEXT, EMPTY, A, B, C, D, EOT, PAD, TR = 1000, 1001, 1002, 1003, 1010, 1011, 1004, 999, 1005
SP4 = Specials(next_audio=NEXT, empty_audio=EMPTY, spk=(A, B, C, D)); PRE = [1, 2, 3]
ev = lambda t: (EOT, t, chunk_of(t + 0.24))


def test_head_text_four_slots():
    assert cst_head_text(["English"] * 4) == "language <SPK_A>English <SPK_B>English <SPK_3>English <SPK_4>English<asr_text>"
    assert cst_head_text(["English"] * 3, ["German"] * 3) == "language <SPK_A>English <SPK_B>English <SPK_3>English\ntranslate <SPK_A>German <SPK_B>German <SPK_3>German<asr_text>"


def test_three_slot_sequence_tags_and_read_back():
    st = [[(10, 0.5), ev(0.6)], [(20, 1.5), ev(1.6)], [(30, 2.5), (31, 2.9), ev(3.0)]]
    s = build_cst_sequence(st, {(0, 1): 1.0, (1, 1): 1.0, (2, 2): 0.5}, 60, PRE, PAD, SP4, delay=2)
    assert [t for t, inp in zip(s["ids"], s["is_input"]) if not inp and t != NEXT] == [A, 10, EOT, B, 20, EOT, C, 30, 31, EOT]
    got, bad = read_back(s["ids"], s["is_input"], SP4, EOT); assert got == [[10, EOT], [20, EOT], [30, 31, EOT], []] and bad == []
    assert len(s["soft_pos"]) == 3


def test_three_slot_translation_flows():
    st = [[(10, 0.5), ev(0.6)], [(20, 1.5)], [(30, 2.5)], [(11, 0.6)], [(21, 1.6)], [(31, 2.6)]]      # 전사 0–2, 번역 3–5
    s = build_cst4_sequence(st, {(0, 1): 1.0}, 60, PRE, PAD, SP4, TR, delay=2)
    out = [t for t, inp in zip(s["ids"], s["is_input"]) if not inp and t != NEXT]
    assert out[:4] == [A, 10, TR, 11] and B in out and C in out and s["n_tr"] == 3
    got, eots, bad = read_back4(s["ids"], s["is_input"], SP4, TR, EOT, n=3)
    assert got == [[10], [20], [30], [11], [21], [31]] and eots == [1, 0, 0] and bad == []


def test_slot_activity_four():
    eps = dialogue_episodes(Dialogue(conv_id="c", corpus="t", lang="English", split="train", duration_s=10.0, speakers=["p", "q", "r"],
                                     utterances=[Utterance("p", 0.0, 1.0, "", "", tokens=[(1, 1.0)], utt_id="p0"), Utterance("r", 2.0, 3.0, "", "", tokens=[(1, 3.0)], utt_id="r0")],
                                     channels={}))
    act = slot_activity(eps, ["p", "r"], 0.0, 4.0, 50, n_slots=4); assert act[5] == [1, 0, 0, 0] and act[30] == [0, 1, 0, 0] and len(act[0]) == 4


sf = pytest.importorskip("soundfile")
from test_dialogue_dataset import FakeTok  # noqa: E402


def make_multi(tmp_path, n=4, dur=72.0):
    """n 화자가 차례로 2 s 씩 말하고 1 s 쉰다(p0 0–2, p1 3–5, …) — 20 s 창에 n 명 모두 든다."""
    import numpy as np
    SR = 16000; p = str(tmp_path / "m.wav"); t = np.arange(int(dur * SR)) / SR
    sf.write(p, np.stack([0.3 * np.sin(2 * np.pi * (300 + 100 * i) * t) for i in range(n)], 1).astype(np.float32), SR)
    spk = [f"p{i}" for i in range(n)]; utts = []; k = 0; s = 0.0
    while s + 2.0 <= dur - 4.0:
        w = spk[k % n]; utts.append(Utterance(w, s, s + 2.0, "ab", "ab", tokens=[(11 + k % n, s + 1.0), (21 + k % n, s + 2.0)], utt_id=f"{w}u{k}")); k += 1; s += 3.0
    d = Dialogue(conv_id="m1", corpus="mtest", lang="English", split="train", duration_s=dur, speakers=spk, utterances=utts,
                 channels={w: ChannelRef(path=f"{p}#ch{i}") for i, w in enumerate(spk)})
    q = str(tmp_path / "mtest.refined.dialogues.jsonl"); open(q, "w").write(d.to_json() + "\n"); return q, d


def test_dataset_four_speakers_consistent_and_declared(tmp_path):
    import random
    q, d = make_multi(tmp_path)
    two = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1)
    assert len(two) == 0 and two.stats["cst_dropped_speakers"] > 0                                  # 2 슬롯(V1): 3 명 이상 창은 버린다
    ds = CstDialogueDataset([q], FakeTok(), delays=(2, 4), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, spk_activity=True, declare_all_p=1.0)
    assert len(ds) > 0 and ds.n_slots == 4 and len(ds.spk_ids) == 4 and ds.spk_ids[2:] == tuple(ds.tok.convert_tokens_to_ids(t) for t in ("<SPK_3>", "<SPK_4>"))
    for i in range(len(ds)):
        for dl in (2, 4):
            s = ds.cst_sequence(i, dl); assert ds.problems_of(s) == [], (ds.items[i], dl, ds.problems_of(s))
            assert len(s["speakers"]) == 4 and len(s["langs"]) == 4                                 # 평가(rng 없음): 선언 = max(2, 화자 수)
    w = CstDialogueDataset([q], FakeTok(), delays=(4,), windows=[("m1", 0.0, 6.0)], max_speakers=4, declare_all_p=1.0)   # 두 명만 말하는 창
    assert len(w.cst_sequence(0, 4)["langs"]) == 2 and len(w.cst_sequence(0, 4, rng=random.Random(0))["langs"]) == 4     # 학습 rng: 4 슬롯 모두 선언
    w4 = CstDialogueDataset([q], FakeTok(), delays=(4,), windows=[("m1", 0.0, 6.0)], max_speakers=4, declare=4)
    s = w4.cst_sequence(0, 4); assert len(s["langs"]) == 4 and w4.problems_of(s) == [] and s["ids"][: len(w4.cst_prefix(["English"] * 4, 4))] == w4.cst_prefix(["English"] * 4, 4)
    b = [ds[i] for i in range(2)]; assert all(x["activity"].shape == (x["K"], 4) for x in b) and collate_cst(b)["activity"].shape[-1] == 4
    with pytest.raises(ValueError): CstDialogueDataset([q], FakeTok(), delays=(4,), max_speakers=5)


def test_dataset_three_speakers_translation_mode(tmp_path):
    import numpy as np
    q, d = make_multi(tmp_path, n=3)
    tr = {f"m1|{u.speaker}|{u.utt_id}": {"German": (np.array([60, 61]), np.array([u.start + 1.0, u.end]))} for u in d.utterances}
    tok = FakeTok(); tok.add_tokens(["<TRANS>"])
    ds = CstDialogueDataset([q], tok, delays=(2, 4), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, trans=tr, ast_frac=1.0)
    assert len(ds) > 0
    for i in range(len(ds)):
        for dl in (2, 4):
            s = ds.cst_sequence(i, dl, tgt="German"); assert ds.problems_of(s) == [], (ds.items[i], dl, ds.problems_of(s))
            assert len(s["streams"]) == 6 and s["n_tr"] > 0 and s["tgt"] == ("German",) * 3


# ── 창 draw(epoch 마다 다른 이어 덮기)·화자 상한 줄이기·sampler
def test_window_draws_tile_and_sampler_switches_per_epoch(tmp_path):
    from vapasr.data.cst_dialogue import window_speakers
    from vapasr.hf.packing import PackedBudgetSampler
    q, d = make_multi(tmp_path, n=2, dur=400.0)
    ds = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 90.0), seed=1, window_draws=3)
    assert ds.n_draws == 3 and len(ds.item_draw) == len(ds.items) and set(ds.item_draw) == {0, 1, 2}
    for r in range(3):
        w = sorted((t0, L) for (c, t0, L), k in zip(ds.items, ds.item_draw) if k == r)
        assert all(20.0 - 1e-6 <= L <= 90.0 + 1e-6 for _, L in w)
        assert all(b[0] >= a[0] + a[1] - 5.0 - 1e-6 for a, b in zip(w, w[1:]))                       # 이어 덮기: 앞 창과 5 s 넘게 겹치지 않는다
        assert sum(L for _, L in w) > 0.8 * (d.duration_s - 90.0)                                       # 대화 대부분을 덮는다
    starts = [sorted(t0 for (c, t0, L), k in zip(ds.items, ds.item_draw) if k == r) for r in range(3)]; assert starts[0] != starts[1] != starts[2]
    for i in range(len(ds)): assert ds.target_problems(i, 4) == [], ds.items[i]
    sp = PackedBudgetSampler(ds, max_tokens=3000, max_bs=8, seed=0)
    seen = []
    for e in range(3):
        sp.set_epoch(e); got = {i for b in sp for i in b}; assert got and {ds.item_draw[i] for i in got} == {e % 3}; seen.append(got)
        assert len(list(sp)) == len(sp)                                                                 # epoch 마다 같은 길이
    assert seen[0] != seen[1]
    legacy = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 40.0), hop_s=10.0, seed=1)
    assert legacy.n_draws == 1 and set(legacy.item_draw) == {0} and len(PackedBudgetSampler(legacy, 3000).by_draw) == 1


def test_shrink_over_cap_windows(tmp_path):
    from vapasr.data.cst_dialogue import window_speakers
    q, d = make_multi(tmp_path, n=4, dur=120.0)
    drop = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(5.0, 30.0), seed=1, window_draws=2, max_speakers=2)
    keep = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(5.0, 30.0), seed=1, window_draws=2, max_speakers=2, shrink_speakers=True)
    assert drop.stats["cst_dropped_speakers"] > 0 and keep.stats["cst_shrunk_speakers"] > 0 and len(keep) > len(drop)
    assert len(keep.items) == len(keep.item_draw) and all(len(window_speakers(keep.eps[c], t0, L)) <= 2 and L >= 5.0 for c, t0, L in keep.items)
    for i in range(len(keep)): assert keep.target_problems(i, 4) == []


# ── 평가 N 슬롯·합성 N 화자
def test_score_window_four_slots_permutation():
    from vapasr.data.cst_eval import score_window, parse_emitted, attr_errors
    W = lambda *ws: [(w, float(i)) for i, w in enumerate(ws)]
    ref = [W("a", "b"), W("c"), W("d", "e", "f")]
    hyp = [W("d", "e", "f"), W("a", "b"), [], W("c")]                                             # 가설 슬롯 0 = 참조 2, 1 = 0, 3 = 1
    sc = score_window(ref, hyp, "English")
    assert sc["cp"] == 0 and sc["slot"] == 9 and sc["swap"] and sc["perm"][0] == 2 and sc["perm"][1] == 0 and sc["perm"][3] == 1 and sc["n_ref"] == 6
    two = score_window([W("a"), W("b")], [W("a"), W("b")], "English"); assert two["perm"] == [0, 1] and not two["swap"] and two["cp"] == 0
    tie = score_window([W("a"), W("a")], [W("a"), W("a")], "English"); assert tie["perm"] == [0, 1]                    # 같으면 항등
    slots, eots = parse_emitted([(0, 7), (1, 50), (2, 8), (3, 50), (4, 9), (5, 99)], (7, 8, 9, 10), 99); assert slots == [[(0.16, 50)], [(0.32, 50)], [], []] and eots == [(2, 0.48)]
    ae = attr_errors(ref, hyp, "English", sc["perm"], busy=[[(0.0, 2.0)], [(0.0, 1.0)], [(0.0, 3.0)]]); assert ae["n_mis"] == 0 and ae["n_ref"] == 6


def test_synth_session_four_speakers():
    import random
    from vapasr.data import sim_dialogue as sd
    def units(key, seed):
        r = random.Random(seed)
        return [sd.Unit(key=key, order=i, path="/x", src_offset=None, dur=r.uniform(0.5, 5.0), tokens=[(100, 0.3)], text="w", uid=f"{key}{i}") for i in range(40)]
    st = sd.TimingStats(fto=[-0.5, 0.2, 0.4, 1.0], hold=[0.3, 0.6], p_hold=0.3, turn=[1.5, 3.0, 6.0], bc_rate=0.05, bc_pos=[0.3, 0.6])
    d = sd.synth_session([units(k, i) for i, k in enumerate("pqrs")], st, random.Random(3), 120.0, lambda u: (0.0, u.dur), "c", "sim", "English")
    assert d.speakers == ["a", "b", "c", "d"] and set(d.channels) == set(d.speakers) and sd.overlap_stats(d)["self_overlap"] == 0
    with pytest.raises(AssertionError): sd.synth_session([units("p", 0)], st, random.Random(0), 60.0, lambda u: (0.0, u.dur), "c", "sim", "English")


def test_shrink_never_admits_extra_speaker_rounding(tmp_path):
    """줄인 길이를 반올림하면 (max+1) 번째 화자의 첫 발화가 창에 들어 slot_activity 가 IndexError(2026-10-20 v2-p1 첫 step). 시작 시각에 ms 이하 소수를 둔다."""
    import numpy as np, random
    from vapasr.data.cst_dialogue import window_speakers
    SR = 16000; dur = 300.0; n = 5; p = str(tmp_path / "r.wav"); t = np.arange(int(dur * SR)) / SR
    sf.write(p, np.stack([0.2 * np.sin(2 * np.pi * (300 + 50 * i) * t) for i in range(n)], 1).astype(np.float32), SR)
    rr = random.Random(7); spk = [f"q{i}" for i in range(n)]; utts = []; s = 0.0; k = 0
    while s + 2.0 < dur - 3.0:
        w = spk[k % n]; a = round(s + rr.uniform(0.0, 0.9), 4); utts.append(Utterance(w, a, a + 1.5, "ab", "ab", tokens=[(11, a + 0.7), (12, a + 1.5)], utt_id=f"{w}u{k}"))
        k += 1; s = a + 1.5 + rr.uniform(0.2, 1.5)
    d = Dialogue(conv_id="r1", corpus="rtest", lang="English", split="train", duration_s=dur, speakers=spk, utterances=utts, channels={w: ChannelRef(path=f"{p}#ch{i}") for i, w in enumerate(spk)})
    q = str(tmp_path / "rtest.refined.dialogues.jsonl"); open(q, "w").write(d.to_json() + "\n")
    ds = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(3.0, 40.0), seed=3, window_draws=8, max_speakers=4, shrink_speakers=True, spk_activity=True)
    assert ds.stats["cst_shrunk_speakers"] > 0
    for i, (c, t0, L) in enumerate(ds.items):
        assert len(window_speakers(ds.eps[c], t0, L)) <= 4, (t0, L)
        ds[i]                                                                                      # 활동 라벨(K, 4)까지 만든다
