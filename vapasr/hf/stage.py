"""가중치 파일을 로컬(tmpfs) 로 순차 복사한 뒤 읽는다 — NFS 위 safetensors mmap 은 페이지 폴트마다 왕복해 2.4 GB 에 수 분이 걸린다(2026-09-09).
VAPASR_STAGE_DIR(기본 /dev/shm/vapasr-stage) 가 없거나 빈 문자열이면 원본 경로를 그대로 쓴다."""
import os, shutil, glob, time
from typing import Optional

STAGE_FILES = ("*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template*", "*.nemo")

def stage_dir(src: str, dst_root: Optional[str] = None, quiet: bool = False) -> str:
    """src 디렉토리의 모델 파일을 dst_root/<src 를 평탄화한 이름>/ 로 복사(크기·mtime 같으면 건너뜀) → 복사본 경로. 실패하면 src."""
    root = os.environ.get("VAPASR_STAGE_DIR", "/dev/shm/vapasr-stage") if dst_root is None else dst_root
    if not root or not os.path.isdir(src): return src
    dst = os.path.join(root, os.path.abspath(src).strip("/").replace("/", "__"))
    try:
        os.makedirs(dst, exist_ok=True); t0 = time.time(); n = 0
        for pat in STAGE_FILES:
            for f in glob.glob(os.path.join(src, pat)):
                if not os.path.isfile(f): continue
                o = os.path.join(dst, os.path.basename(f)); st = os.stat(f)
                if os.path.exists(o) and os.path.getsize(o) == st.st_size and abs(os.stat(o).st_mtime - st.st_mtime) < 1: continue
                with open(f, "rb") as a, open(o + ".tmp", "wb") as b: shutil.copyfileobj(a, b, 64 << 20)     # 64 MB 순차 읽기
                os.utime(o + ".tmp", (st.st_atime, st.st_mtime)); os.replace(o + ".tmp", o); n += 1
        if n and not quiet: print(f"stage: {src} → {dst} ({n} 파일, {time.time() - t0:.0f}s)", flush=True)
        return dst
    except OSError as e:
        if not quiet: print(f"stage 실패({e}) → 원본에서 읽음: {src}", flush=True)
        return src
