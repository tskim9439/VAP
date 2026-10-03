#!/usr/bin/env python3
"""BAS TAXI → CST 공통 스키마 매니페스트(JSONL, 세션당 한 줄) + 선택적으로 16 kHz 리샘플 음성.

  python experiments/cst_build_taxi.py --root /Volumes/Samsung_T5/VAPKT-DB/TAXI/TAXI \\
      --out /Volumes/Samsung_T5/VAPKT-DB/TAXI/cst/taxi-sessions.jsonl [--resample-16k /Volumes/Samsung_T5/VAPKT-DB/TAXI/cst/wav16k]

--resample-16k 를 주면 usable 턴의 8 kHz 음성을 16 kHz(scipy resample_poly 2:1, 16-bit)로 같은 상대 경로에 쓰고,
매니페스트에 그 경로를 audio_16k 로 남긴다(대역은 전화 대역 그대로 — 고역은 복원되지 않는다).
TAXI 는 재배포 금지라 산출물은 로컬(T5)·내부 서버에만 둔다."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vapasr.cst.taxi import iter_sessions, stats


def resample_16k(src: Path, dst: Path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(str(src), dtype="float32")
    assert sr == 8000 and x.ndim == 1, (src, sr, x.shape)
    y = np.clip(resample_poly(x, 2, 1), -1.0, 1.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav"); sf.write(str(tmp), y, 16000, subtype="PCM_16"); tmp.replace(dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--resample-16k", default=None)
    a = p.parse_args()
    sessions = list(iter_sessions(a.root))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rs = Path(a.resample_16k) if a.resample_16k else None
    with out.open("w") as f:
        for s in sessions:
            if rs:
                s.audio["root_16k"] = str(rs)
                for t in s.usable_turns():
                    dst = rs / t.audio
                    if not dst.exists():
                        resample_16k(Path(a.root) / t.audio, dst)
            f.write(s.to_json() + "\n")
    print(json.dumps(stats(sessions), ensure_ascii=False))


if __name__ == "__main__":
    main()
