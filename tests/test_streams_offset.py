"""src_offset_s(긴 대화 wav 안의 발화 구간) · #chN 채널 읽기 — AI Hub 71631 stereo 경로 형식."""
import os, numpy as np, pytest
sf = pytest.importorskip("soundfile")
from vapasr.data.streams import load_utt_audio, load_seg_audio, assemble_stream, seg_audio_key, iter_utterances, SR

@pytest.fixture
def stereo_wav(tmp_path):
    sr = 48000; t = np.arange(sr * 4) / sr
    x = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)], 1).astype(np.float32) * 0.5
    p = str(tmp_path / "st.wav"); sf.write(p, x, sr); return p

def test_offset_read_matches_full_read(stereo_wav):
    full = load_utt_audio(stereo_wav + "#ch1"); cut = load_utt_audio(stereo_wav + "#ch1", 1.25, 0.5)
    ref = full[int(1.25 * SR): int(1.25 * SR) + int(0.5 * SR)]
    assert len(cut) == len(ref) == 8000 and float(np.abs(cut - ref).max()) < 0.05      # 리샘플 경계 오차만

def test_channel_suffix_selects_channel(stereo_wav):
    a = load_utt_audio(stereo_wav + "#ch0", 1.0, 0.1); b = load_utt_audio(stereo_wav + "#ch1", 1.0, 0.1)
    assert float(np.abs(a - b).max()) > 0.1

def test_assemble_uses_src_offset_and_keys_cache(stereo_wav):
    seg = dict(utt_id="u", path=stereo_wav + "#ch1", silence_before_s=0.3, offset_s=0.3, dur_s=0.5, src_offset_s=1.25, text="x")
    cache = {}; y = assemble_stream(dict(id="r1", duration_s=1.1, segments=[seg]), cache=cache)
    ref = load_seg_audio(seg); o = int(0.3 * SR)
    assert len(y) == int(1.1 * SR) and np.allclose(y[o: o + len(ref)], ref) and list(cache) == [seg_audio_key(seg)] and "@1.25" in seg_audio_key(seg)
    u = next(iter_utterances(dict(segments=[seg]))); assert u["src_offset_s"] == 1.25 and u["dur_s"] == 0.5

def test_segment_past_eof_is_clamped_or_rejected(stereo_wav):
    x = load_utt_audio(stereo_wav + "#ch0", 3.5, 1.0)                    # 4 s 파일에서 3.5–4.5 s → 0.5 s 만
    assert abs(len(x) - int(0.5 * SR)) <= 2
    with pytest.raises(ValueError): load_utt_audio(stereo_wav + "#ch0", 4.2, 1.0)
    with pytest.raises(ValueError): load_utt_audio(stereo_wav + "#ch0", 3.97, 1.0)   # 남은 오디오 0.1 s 미만
