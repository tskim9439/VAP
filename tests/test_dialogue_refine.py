"""라벨 시각 보정 — 느슨한 라벨 끝을 VAD·토큰으로 당기고, 발화 안 pause 에서 분할하며, 토큰을 배분한다."""
import numpy as np, pytest
sf = pytest.importorskip("soundfile")
from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef
from vapasr.data.dialogue_refine import channel_vad, refine_utterances
SR = 16000

def make(tmp_path):
    # 채널 0: 1.0–2.0 s 음성, 2.6–3.4 s 음성(pause 0.6 s), 나머지 무음. 라벨은 0.8–4.5 s 하나(느슨).
    t = np.arange(int(6 * SR)) / SR; x = np.zeros((len(t), 2), np.float32)
    for s, e in ((1.0, 2.0), (2.6, 3.4)): x[int(s * SR): int(e * SR), 0] = 0.3 * np.sin(2 * np.pi * 300 * t[int(s * SR): int(e * SR)])
    x += np.random.RandomState(0).randn(*x.shape).astype(np.float32) * 1e-4
    p = str(tmp_path / "st.wav"); sf.write(p, x, SR)
    toks = [(11, 1.4), (12, 1.9), (13, 3.0), (14, 3.35)]
    d = Dialogue(conv_id="c", corpus="t", lang="Korean", split="train", duration_s=6.0, speakers=["0", "1"],
                 utterances=[Utterance("0", 0.8, 4.5, "a b c d", "a b c d", tokens=toks, utt_id="u1")], channels={"0": ChannelRef(path=p + "#ch0"), "1": ChannelRef(path=p + "#ch1")})
    return d

def test_refine_tightens_and_splits(tmp_path):
    d = make(tmp_path); vad = channel_vad(d)
    assert len(vad["0"]) == 2 and vad["1"] == [] and abs(vad["0"][0][0] - 1.0) < 0.15 and abs(vad["0"][1][1] - 3.4) < 0.15
    out, st = refine_utterances(d, vad)
    assert len(out) == 2 and st["split"] == 1
    a, b = out; assert a.speaker == "0" and abs(a.start - 0.9) < 0.15 and abs(a.end - 2.1) < 0.2 and [t for t, _ in a.tokens] == [11, 12]
    assert abs(b.start - 2.5) < 0.2 and abs(b.end - 3.5) < 0.2 and [t for t, _ in b.tokens] == [13, 14] and b.utt_id.startswith("u1#") and b.text == ""

def test_refine_without_vad_uses_tokens(tmp_path):
    d = make(tmp_path); out, st = refine_utterances(d, {"0": [], "1": []})
    assert st["no_vad"] == 1 and len(out) == 1 and abs(out[0].end - 3.45) < 1e-6 and out[0].start == 0.8

def test_refine_keeps_tokenless_utterance_as_one_span(tmp_path):
    d = make(tmp_path); d.utterances[0].tokens = None; d.utterances[0].flags = ["digit"]; d.utterances[0].text = ""
    out, st = refine_utterances(d, channel_vad(d))
    assert len(out) == 1 and out[0].flags == ["digit"] and out[0].tokens is None and abs(out[0].start - 0.9) < 0.15 and abs(out[0].end - 3.5) < 0.2
