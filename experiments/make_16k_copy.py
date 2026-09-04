#!/usr/bin/env python
"""otoSpeech 48 kHz FLOAT → 16 kHz PCM_16 사본 (디렉토리 구조 유지). 로더의 리샘플 비용 제거.
python experiments/make_16k_copy.py /data3/tskim/corpora/turnbench/otoSpeech /data3/tskim/corpora/turnbench/otoSpeech16k [--workers 8]
"""
import os, sys, glob, shutil, argparse, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf, soxr
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()

def one(d):
    name = os.path.basename(d); out = os.path.join(a.dst, name); os.makedirs(out, exist_ok=True)
    for c in (1, 2):
        p, q = os.path.join(d, f"speaker_{c}_audio.wav"), os.path.join(out, f"speaker_{c}_audio.wav")
        if os.path.exists(q): continue
        x, sr = sf.read(p, dtype="float32")
        if sr != 16000: x = soxr.resample(x, sr, 16000)
        sf.write(q + ".tmp", np.clip(x, -1, 1), 16000, subtype="PCM_16", format="WAV"); os.replace(q + ".tmp", q)
    for f in ("metadata.json", "speaker_1_annotation_a.srt", "speaker_2_annotation_a.srt"):
        if not os.path.exists(os.path.join(out, f)): shutil.copy(os.path.join(d, f), out)
    return name

dirs = sorted(d for d in glob.glob(os.path.join(a.src, "*")) if os.path.isfile(os.path.join(d, "metadata.json")))
t0 = time.time(); n = 0
with ProcessPoolExecutor(a.workers) as ex:
    for name in ex.map(one, dirs):
        n += 1
        if n % 50 == 0: print(f"  {n}/{len(dirs)}  {time.time()-t0:.0f}s", flush=True)
print(f"done {n} dirs in {time.time()-t0:.0f}s → {a.dst}")
