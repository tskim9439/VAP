"""압축(tar) 안의 오디오를 풀지 않고 읽기 — tar 멤버 오프셋 인덱스(2026-09-07, 사용자 지시).
경로 표기: "<archive.tar[.gz]>::<member/path.wav>". gzip tar 는 무작위 접근이 불가하므로 인덱스만 만들고(순차 1 회) 읽기는 비압축 tar 만 지원한다;
gzip 이면 최초 1 회 `tar_index` 가 경고하고, 학습 전에 비압축 tar 로 재포장(`tar -xOf a.tar.gz | tar -cf a.tar`)하거나 세그먼트 캐시를 권한다.
인덱스는 <archive>.index.json 옆에 저장(쓰기 불가 디렉토리면 $MXC_DATA_ROOT/archive-index/ 아래)."""
import os, io, json, tarfile, hashlib
from typing import Dict, Tuple, Optional
import numpy as np

SEP = "::"
def is_archive_path(p: str) -> bool: return SEP in p

def _index_path(archive: str) -> str:
    p = archive + ".index.json"; d = os.path.dirname(archive)
    if os.access(d, os.W_OK): return p
    root = os.environ.get("MXC_DATA_ROOT", os.environ.get("DATA_ROOT", "/tmp")); os.makedirs(os.path.join(root, "archive-index"), exist_ok=True)
    return os.path.join(root, "archive-index", hashlib.sha1(archive.encode()).hexdigest()[:12] + "-" + os.path.basename(archive) + ".index.json")

def tar_index(archive: str) -> Dict[str, Tuple[int, int]]:
    """member → (data offset, size). 한 번 순차로 훑고 저장. 이후는 인덱스만 읽는다."""
    ip = _index_path(archive)
    if os.path.exists(ip): return {k: tuple(v) for k, v in json.load(open(ip)).items()}
    idx = {}
    with tarfile.open(archive, "r:*") as tf:            # gz 도 인덱스는 만들 수 있다(offset 은 압축 해제 스트림 기준 → 읽기는 비압축 tar 만)
        for m in tf:
            if m.isfile(): idx[m.name] = (m.offset_data, m.size)
    json.dump(idx, open(ip + ".tmp", "w")); os.replace(ip + ".tmp", ip)
    if archive.endswith((".gz", ".tgz")): print(f"!! {archive}: gzip tar 는 무작위 읽기 불가 — 인덱스만 생성. 비압축 tar 로 재포장 필요", flush=True)
    return idx

_IDX: Dict[str, Dict[str, Tuple[int, int]]] = {}
def read_member(archive: str, member: str) -> bytes:
    if archive not in _IDX: _IDX[archive] = tar_index(archive)
    off, size = _IDX[archive][member]
    with open(archive, "rb") as f: f.seek(off); return f.read(size)

def load_audio_from_archive(path: str) -> np.ndarray:
    """"a.tar::x.wav" → float32 (T,) @16 kHz (wav/flac 은 soundfile, .pcm 은 raw 16-bit)."""
    archive, member = path.split(SEP, 1); b = read_member(archive, member)
    if member.endswith(".pcm"):
        if len(b) % 2: b = b[:-1]
        return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True); x = x[:, 0]
    if sr != 16000:
        import soxr; x = soxr.resample(x, sr, 16000)
    return x
