"""Phase 2 mono 혼합 — 채널 wav·조각 배치·게인·피크 제한."""
import numpy as np, pytest
sf = pytest.importorskip("soundfile")
from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef
from vapasr.data.dialogue_mix import load_channel, mix_dialogue, crop_audio
SR = 16000

def tone(f, dur, sr=SR, amp=0.3):
    t = np.arange(int(dur * sr)) / sr; return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)

def test_pieces_are_placed_at_offsets(tmp_path):
    p1 = str(tmp_path / "a_1.wav"); sf.write(p1, tone(440, 0.5), SR); p2 = str(tmp_path / "a_3.wav"); sf.write(p2, tone(440, 0.5), SR)
    x = load_channel(ChannelRef(pieces=[(p1, 1.0), (p2, 2.5)]), 4.0)
    assert len(x) == 4 * SR and np.abs(x[: SR]).max() == 0 and np.abs(x[SR: SR + 8000]).max() > 0.2 and np.abs(x[int(2.5 * SR) + 8000:]).max() == 0

def test_channel_suffix_and_stereo_split(tmp_path):
    st = str(tmp_path / "st.wav"); sf.write(st, np.stack([tone(440, 2.0), tone(880, 2.0)], 1), SR)
    a = load_channel(ChannelRef(path=st + "#ch0"), 2.0); b = load_channel(ChannelRef(path=st + "#ch1"), 2.0)
    assert len(a) == 2 * SR and float(np.abs(a - b).max()) > 0.1

def test_mix_normalizes_each_speaker_and_limits_peak(tmp_path):
    st = str(tmp_path / "st.wav"); sf.write(st, np.stack([tone(440, 2.0, amp=0.05), tone(880, 2.0, amp=0.9)], 1), SR)
    d = Dialogue(conv_id="c", corpus="t", lang="Korean", split="train", duration_s=2.0, speakers=["0", "1"],
                 utterances=[Utterance("0", 0.0, 1.0), Utterance("1", 1.0, 2.0)], channels={"0": ChannelRef(path=st + "#ch0"), "1": ChannelRef(path=st + "#ch1")})
    mono, meta = mix_dialogue(d, target_dbfs=-20.0)
    assert len(mono) == 2 * SR and float(np.abs(mono).max()) <= 0.98 + 1e-6
    r0 = float(np.sqrt(np.mean(mono[: SR] ** 2))); r1 = float(np.sqrt(np.mean(mono[SR:] ** 2)))
    assert abs(20 * np.log10(r0 / r1)) < 1.0 and meta["gains"]["0"] > meta["gains"]["1"]      # 두 화자 구간 RMS 가 같아짐
    assert len(crop_audio(mono, 0.5, 1.25)) == int(0.75 * SR)

def test_missing_channel_reported(tmp_path):
    d = Dialogue(conv_id="c", corpus="t", lang="Korean", split="train", duration_s=1.0, speakers=["0", "1"], utterances=[], channels={"0": ChannelRef(pieces=[])})
    mono, meta = mix_dialogue(d); assert meta["missing_channels"] == ["0", "1"] and float(np.abs(mono).max()) == 0

def test_empty_pieces_survive_json_roundtrip_and_mix(tmp_path):
    d = Dialogue(conv_id="c", corpus="t", lang="Korean", split="train", duration_s=1.0, speakers=["0", "1"], utterances=[], channels={"0": ChannelRef(pieces=[]), "1": ChannelRef(pieces=[])})
    d2 = Dialogue.from_json(d.to_json()); assert d2.channels["0"].pieces == [] and d2.channels["0"].path == ""
    mono, meta = mix_dialogue(d2); assert len(mono) == SR and meta["missing_channels"] == ["0", "1"]
