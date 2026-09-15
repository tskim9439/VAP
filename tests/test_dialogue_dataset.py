"""Phase 2 DialogueWindowDataset — 가짜 tokenizer·합성 wav 로 창 선택·시퀀스·soft·활동·collate 를 검사."""
import json, numpy as np, pytest, torch
sf = pytest.importorskip("soundfile")
from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef
from vapasr.data.dialogue_dataset import DialogueWindowDataset, collate_dialogue
from vapasr.data.dialogue_tokens import PHASE2_SPECIALS
from vapasr.uslm.interleave_data import SPECIAL_TOKENS
SR = 16000

class FakeTok:
    """add_tokens / convert_tokens_to_ids / __call__ 만 흉내. 특수 토큰은 150000 부터, 텍스트는 문자 코드."""
    def __init__(self): self.v = {"<|audio_pad|>": 149999}
    def add_tokens(self, toks, special_tokens=True):
        for t in toks: self.v.setdefault(t, 150000 + len(self.v))
    def convert_tokens_to_ids(self, t): return self.v[t]
    def __call__(self, text, add_special_tokens=False, **kw): return {"input_ids": [ord(c) % 1000 + 1 for c in text]}

def make(tmp_path, n_utts_missing_tokens=False):
    dur = 60.0; st = str(tmp_path / "st.wav"); t = np.arange(int(dur * SR)) / SR
    sf.write(st, np.stack([0.3 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 660 * t)], 1).astype(np.float32), SR)
    utts = []
    for k in range(10):                                   # a: 0–2, 6–8, …  b: 3–5, 9–11, …  (서로 안 겹침, 1 s 간격의 mutual silence)
        s = k * 6.0; utts.append(Utterance("0", s, s + 2.0, "ab", "ab", tokens=[(11, s + 1.0), (12, s + 2.0)], utt_id=f"a{k}"))
        utts.append(Utterance("1", s + 3.0, s + 5.0, "cd", "cd", tokens=None if n_utts_missing_tokens and k == 4 else [(13, s + 4.0), (14, s + 5.0)], utt_id=f"b{k}"))
    d = Dialogue(conv_id="c1", corpus="test", lang="Korean", split="train", duration_s=dur, speakers=["0", "1"], utterances=utts, channels={"0": ChannelRef(path=st + "#ch0"), "1": ChannelRef(path=st + "#ch1")})
    p = str(tmp_path / "test.dialogues.jsonl"); open(p, "w").write(d.to_json() + "\n"); return p

def test_windows_start_in_silence_and_samples_are_consistent(tmp_path):
    ds = DialogueWindowDataset([make(tmp_path)], FakeTok(), window_s=(20.0, 20.0), hop_s=5.0, delays=(4,), seed=1)
    assert len(ds) > 0 and ds.stats["skipped_untranscribed"] == 0
    for cid, t0, L in ds.items: assert not any(u.start < t0 < u.end for u in ds.dlgs[cid].utterances)     # 시작은 무음
    s = ds[0]; K = s["K"]; assert K == 250 and int(s["is_audio"].sum()) == K and s["wav"].shape[0] == int(20.0 * SR)
    assert s["activity"].shape == (K, 6) and (s["activity"].sum(1) <= 2).all() and s["labels"][s["is_audio"]].eq(-100).all()
    eot = ds.sp.eot; assert (s["ids"][s["soft_pos"]] == eot).all() and all(0.0 <= w <= 1.0 for w in s["soft_w"].tolist())
    seq = ds.sequence(0, 4); assert seq["stats"].skipped_episodes == 0 and seq["info"]["alloc"].exhausted == 0

def test_truncated_episode_masks_eot(tmp_path):
    ds = DialogueWindowDataset([make(tmp_path)], FakeTok(), window_s=(7.0, 7.0), hop_s=6.0, delays=(2,), seed=0)
    eps, info = ds.window_episodes(ds.dlgs["c1"], 0.0, 7.0)          # 창 [0,7): a 0–2 완전, b 3–5 완전, a 6–8 은 7 에서 잘림
    cut = [e for e in eps if e.speaker == "0" and abs(e.start - 6.0) < 1e-6][0]
    assert cut.outcome == "truncated" and cut.p_end is None and cut.end == 7.0 and [t for _, t in cut.tokens] == [7.0]
    full = [e for e in eps if e.speaker == "0" and e.start == 0.0][0]; assert full.p_end is not None and full.outcome == "shift"

def test_unaligned_utterance_excludes_window(tmp_path):
    ds = DialogueWindowDataset([make(tmp_path, n_utts_missing_tokens=True)], FakeTok(), window_s=(20.0, 20.0), hop_s=5.0, delays=(4,))
    assert ds.stats["skipped_untranscribed"] > 0 and not any(t0 < 29.0 < t0 + L for _, t0, L in ds.items)      # b4(27–29 s) 를 덮는 창 없음

def test_collate_dialogue(tmp_path):
    ds = DialogueWindowDataset([make(tmp_path)], FakeTok(), window_s=(20.0, 25.0), hop_s=5.0, delays=(4,), seed=2)
    b = collate_dialogue([ds[0], ds[1]])
    assert b["ids"].shape[0] == 2 and b["activity"].shape[0] == 2 and b["activity"].shape[2] == 6 and b["activity_mask"].sum() == ds[0]["K"] + ds[1]["K"]
    assert b["soft_b"].shape == b["soft_pos"].shape == b["soft_w"].shape and b["wav"].shape[0] == 2
