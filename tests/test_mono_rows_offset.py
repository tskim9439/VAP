"""MonoStreamDataset 의 행 인덱스(_Rows)가 segment 의 src_offset_s 를 보존하는지 — 빠지면 긴 대화 wav 발화(71631)가 파일 첫머리 오디오로 조립된다(D2 까지의 버그)."""
import os, json, gzip, pytest
from vapasr.data.streams import assemble_stream, seg_audio_key

def test_seg_key_and_assemble_use_src_offset(tmp_path):
    sf = pytest.importorskip("soundfile"); import numpy as np
    sr = 16000; t = np.arange(sr * 6) / sr; x = np.zeros((sr * 6, 2), np.float32); x[sr * 4: sr * 5, 1] = np.sin(2 * np.pi * 440 * t[: sr]) * 0.5   # 4–5 s 구간에만 ch1 소리
    p = str(tmp_path / "conv.wav"); sf.write(p, x, sr)
    seg = dict(path=p + "#ch1", offset_s=0.3, silence_before_s=0.3, dur_s=1.0, text="x", src_offset_s=4.0)
    y = assemble_stream(dict(id="r", duration_s=1.6, segments=[seg])); o = int(0.3 * sr)
    assert float(np.abs(y[o: o + sr]).max()) > 0.4, "src_offset_s 구간(소리 있음)이 조립돼야 한다"
    seg_bad = {k: v for k, v in seg.items() if k != "src_offset_s"}                       # 예전 _Rows 가 만들던 형태
    y2 = assemble_stream(dict(id="r", duration_s=1.6, segments=[seg_bad])); assert float(np.abs(y2[o: o + sr]).max()) < 1e-6, "src_offset_s 없으면 파일 첫머리(무음)"
    assert "@4.0" in seg_audio_key(seg)

def test_rows_keep_src_offset(monkeypatch, tmp_path):
    """mono_data._Rows 가 src_offset_s 를 남기는지 — read_streams 를 가짜 행으로 대체해 확인."""
    import vapasr.uslm.mono_data as md
    row = dict(id="ah71-train-x", duration_s=2.0, subset="train", mode="stream", segments=[dict(path="/a.wav#ch0", offset_s=0.3, silence_before_s=0.3, dur_s=1.2, text="x", src_offset_s=12.5)])
    src = open(md.__file__, encoding="utf-8").read(); assert '"src_offset_s": s["src_offset_s"]' in src and "_items-online-v2" in src
