"""vapasr.hf.stage.stage_dir — 같은 src 를 여러 프로세스가 동시에 복사해도 복사본이 원본과 같다(잠금·프로세스별 임시 이름). CPU·파일만."""
import hashlib
import os
from multiprocessing import get_context

from vapasr.hf.stage import stage_dir

FILES = ("model.safetensors", "config.json")


def _stage(args):
    src, root = args
    d = stage_dir(src, root, quiet=True)
    return d, {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() for f in FILES}


def test_concurrent_stage_same_src(tmp_path):
    src = tmp_path / "ck"; src.mkdir()
    blob = os.urandom(1 << 20) * 48                                                            # 48 MB — 복사가 겹칠 만큼 크게
    (src / "model.safetensors").write_bytes(blob); (src / "config.json").write_text('{"a": 1}')
    want = {"model.safetensors": hashlib.sha256(blob).hexdigest(), "config.json": hashlib.sha256(b'{"a": 1}').hexdigest()}
    root = tmp_path / "stage"
    with get_context("fork").Pool(6) as p: got = p.map(_stage, [(str(src), str(root))] * 6)
    dst = got[0][0]
    assert dst != str(src) and all(d == dst and h == want for d, h in got), got
    assert not [f for f in os.listdir(dst) if ".tmp" in f]                                     # 임시 파일이 남지 않는다
    assert stage_dir(str(src), str(root), quiet=True) == dst                                   # 다시 부르면 건너뛴다(크기·mtime 같음)
