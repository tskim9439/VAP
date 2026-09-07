"""Stage 1 스트림 조립 — streams.jsonl(experiments/s1_build_manifest.py) 행 → mono 16 kHz 파형.

무음은 디지털 0 이 아니라 **인접 발화의 앞/뒤 20 ms 배경을 늘린 것**이다(마이크 노이즈 플로어 유지, 인코더가 '무음 = 완전 0' 을 배우지 않게).
길이·위치는 manifest 에 기록된 silence_before_s / offset_s / dur_s 를 그대로 따르므로, 캐시·정렬·로더가 같은 시각 축을 본다.
"""
import os, json, hashlib
from typing import List, Dict, Iterator, Optional
import numpy as np
from .kspon import read_pcm, SR

EDGE = int(0.02 * SR)   # 320 샘플

def read_streams(manifest_dir: str, mode: Optional[str] = None, subset: Optional[str] = None) -> List[dict]:
    rows = [json.loads(l) for l in open(os.path.join(manifest_dir, "streams.jsonl"), encoding="utf-8")]
    if mode: rows = [r for r in rows if r["mode"] == mode]
    if subset: rows = [r for r in rows if r["subset"] == subset]
    return rows

def load_utt_audio(path: str) -> np.ndarray:
    """flac/wav(헤더 있음) 은 soundfile, .pcm 은 raw 리더. → float32 (T,) @16 kHz"""
    if path.endswith(".pcm"): return read_pcm(path)[0]
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x[:, 0]
    if sr != SR:
        import soxr; x = soxr.resample(x, sr, SR)
    return x

def flac_duration(path: str) -> float:
    """FLAC STREAMINFO 헤더(처음 42 바이트)만 읽어 길이(초)를 구한다 — sf.info 보다 수십 배 빠르다(NFS 에서 파일당 1 회 작은 read).
    헤더가 예상과 다르면 soundfile 로 폴백."""
    with open(path, "rb") as f: d = f.read(42)
    if len(d) == 42 and d[:4] == b"fLaC" and (d[4] & 0x7F) == 0:               # 첫 메타데이터 블록 = STREAMINFO
        info = d[8:42]; sr = (info[10] << 12) | (info[11] << 4) | (info[12] >> 4)
        total = ((info[13] & 0x0F) << 32) | (info[14] << 24) | (info[15] << 16) | (info[16] << 8) | info[17]
        if sr > 0 and total > 0: return total / sr
    import soundfile as sf
    return float(sf.info(path).duration)

def _rng(seed_key: str) -> np.random.RandomState:
    return np.random.RandomState(int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16))

def silence_like(edge: np.ndarray, n: int, rng: np.random.RandomState, gain: float = 0.7) -> np.ndarray:
    """edge(20 ms 배경) 를 n 샘플로 늘린다: 무작위 순서·부호 반전으로 타일링 후 gain. edge 가 사실상 0 이면 0."""
    if n <= 0: return np.zeros(0, np.float32)
    if len(edge) < 16 or float(np.sqrt(np.mean(edge ** 2))) < 1e-5: return np.zeros(n, np.float32)
    reps = n // len(edge) + 2; tiles = [edge if rng.rand() < 0.5 else edge[::-1] for _ in range(reps)]
    tiles = [t * (1 if rng.rand() < 0.5 else -1) for t in tiles]
    return (np.concatenate(tiles)[:n] * gain).astype(np.float32)

def assemble_stream(row: dict, cache: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """→ float32 (T,), T = round(duration_s·SR). segments 의 offset_s 위치에 발화가 놓인다."""
    rng = _rng(row["id"]); T = int(round(row["duration_s"] * SR)); out = np.zeros(T, np.float32); prev_tail = None
    for seg in row["segments"]:
        x = cache[seg["path"]] if cache is not None and seg["path"] in cache else load_utt_audio(seg["path"])
        if cache is not None: cache[seg["path"]] = x
        o = int(round(seg["offset_s"] * SR)); g = int(round(seg["silence_before_s"] * SR))
        edge = prev_tail if prev_tail is not None else x[:EDGE]
        out[max(0, o - g): o] = silence_like(edge, o - max(0, o - g), rng)
        n = min(len(x), T - o); out[o: o + n] = x[:n]; prev_tail = x[-EDGE:]
        end = o + n
    if prev_tail is not None and end < T: out[end:] = silence_like(prev_tail, T - end, rng)
    return out

def iter_utterances(row: dict) -> Iterator[dict]:
    """정렬·채점용: 발화 단위 (start, end, text, path) — 스트림 시각 기준."""
    for i, seg in enumerate(row["segments"]):
        yield dict(idx=i, start=seg["offset_s"], end=round(seg["offset_s"] + seg["dur_s"], 3), text=seg["text"], path=seg["path"], utt_id=seg["utt_id"])
