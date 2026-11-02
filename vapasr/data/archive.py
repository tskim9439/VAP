"""압축(tar) 안의 오디오를 풀지 않고 읽기 — tar 멤버 오프셋 인덱스(2026-09-07, 사용자 지시).
경로 표기: "<archive.tar[.gz]|archive.zip>::<member/path.wav>". zip 은 zipfile 로 멤버만 읽는다(AI Hub 031/033). gzip tar 는 무작위 접근이 불가하므로 인덱스만 만들고(순차 1 회) 읽기는 비압축 tar 만 지원한다;
gzip 이면 최초 1 회 `tar_index` 가 경고하고, 학습 전에 비압축 tar 로 재포장(`tar -xOf a.tar.gz | tar -cf a.tar`)하거나 세그먼트 캐시를 권한다.
인덱스는 <archive>.index.json 옆에 저장(쓰기 불가 디렉토리면 $MXC_DATA_ROOT/archive-index/ 아래)."""
import os, io, json, tarfile, hashlib, unicodedata
from typing import Dict, List, Tuple, Optional
import numpy as np

SEP = "::"
def is_archive_path(p: str) -> bool: return SEP in p

# 공용 DB 가 옮긴 폴더: 옛 경로 → 지금 경로. 공용 폴더는 고치거나 링크를 걸지 않고 읽는 쪽에서 찾는다.
MOVED = {"/soundai/DB/raw/aihub/031.방송콘텐츠 한국어-영어 통·번역 음성 데이터": "/soundai/DB/raw/aihub/031.broadcasting_koen"}   # 2026-11-01 이름 변경, 안쪽 한글 이름은 NFD

def _rules(moved: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """(옛 폴더 NFC, 새 폴더 + "/", 옛 폴더의 ASCII 머리 — 옮긴 폴더 밖 경로를 I/O·정규화 없이 거른다)"""
    return [(unicodedata.normalize("NFC", o), n.rstrip("/") + "/", o[:next((i for i, c in enumerate(o) if not c.isascii()), len(o))]) for o, n in moved.items()]
_MOVED = _rules(MOVED)
_RESOLVED: Dict[str, str] = {}

def resolve_path(p: str) -> str:
    """옮긴 공용 폴더(MOVED) 아래 옛 경로 → 지금 있는 경로(찾으면 프로세스 안에서 기억). 그 밖 경로는 I/O 없이 그대로.
    정규형이 섞여 있어(031: 옛 manifest 는 맨 위 폴더만 NFD·안쪽 NFC, 새 폴더는 안쪽 한글 이름이 모두 NFD) 옛 폴더는 NFC 로 비교하고
    나머지는 그대로·NFD·NFC 중 있는 것을 쓴다. 옛 경로가 다시 생기면 옛 경로, 어디에도 없으면 p 그대로(원래 오류가 난다)."""
    for old, new, head in _MOVED:
        if not p.startswith(head) or p.startswith(new): continue
        if p in _RESOLVED: return _RESOLVED[p]
        parts = p.split("/"); n = old.count("/") + 1
        if unicodedata.normalize("NFC", "/".join(parts[:n])) != old: continue
        rest = "/".join(parts[n:])
        r = next((c for c in (p, new + rest, new + unicodedata.normalize("NFD", rest), new + unicodedata.normalize("NFC", rest)) if os.path.exists(c)), None)
        if r is None: return p                                        # 못 찾은 것은 기억하지 않는다(NFS 일시 오류 뒤 다시 찾게)
        if len(_RESOLVED) >= 65536: _RESOLVED.clear()
        _RESOLVED[p] = r; return r
    return p

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
_ZIP: Dict[str, "zipfile.ZipFile"] = {}
_PREBUILT: Dict[str, Dict[str, Tuple[int, int]]] = {}

def _prebuilt_index(archive: str) -> Dict[str, Tuple[int, int]]:
    """Read the speechlm ASR pipeline's identity-checked tar index, read-only.

    VAPASR_TAR_INDEX_ROOT opts in. A missing index is an error rather than a
    hidden full-tar scan during training (read_member falls back only to an
    archive's own existing <archive>.index.json).
    """
    root = os.environ["VAPASR_TAR_INDEX_ROOT"]
    if archive in _PREBUILT:
        return _PREBUILT[archive]
    from pathlib import Path
    from .selection import digest
    path = Path(archive)
    stat = path.stat()
    identity = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    source = Path(root) / (digest(identity) + ".json")
    if not source.is_file():
        raise FileNotFoundError(f"prebuilt tar index missing: {source}")
    content = json.loads(source.read_text())
    if content.get("identity") != identity:
        raise ValueError(f"prebuilt tar index identity mismatch: {source}")
    if len(_PREBUILT) >= 32:
        _PREBUILT.pop(next(iter(_PREBUILT)))
    _PREBUILT[archive] = {k: tuple(v) for k, v in content["members"].items()}
    return _PREBUILT[archive]

def read_member(archive: str, member: str) -> bytes:
    archive = resolve_path(archive)
    if archive.endswith(".zip"):                      # zip: 중앙 디렉토리로 멤버만 읽는다(풀지 않음). 프로세스당 핸들 캐시
        import zipfile
        if archive not in _ZIP: _ZIP[archive] = zipfile.ZipFile(archive)
        try: return _ZIP[archive].read(member)
        except KeyError: return _ZIP[archive].read("/" + member)   # AI Hub zip 은 멤버 이름이 "/" 로 시작
    if os.environ.get("VAPASR_TAR_INDEX_ROOT"):
        try:
            off, size = _prebuilt_index(archive)[member]
        except FileNotFoundError:                        # not a speechlm archive (e.g. voxpopuli): its own index file, never a full scan
            if not os.path.exists(_index_path(archive)): raise
            if archive not in _IDX: _IDX[archive] = tar_index(archive)
            off, size = _IDX[archive][member]
    else:
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
