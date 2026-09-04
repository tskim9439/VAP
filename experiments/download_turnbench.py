#!/usr/bin/env python
"""TurnBench test + otoSpeech 를 /data3/tskim/corpora/turnbench/ 로 내려받는다 (재개 가능). HF_TOKEN 필요."""
import os, sys, time
from huggingface_hub import snapshot_download
root = os.path.join(os.environ.get("DATA_CORPORA_DIR", "/data3/tskim/corpora"), "turnbench")
targets = {"dev": "mundo-ai/turn-benchmark-dev", "test": "mundo-ai/turn-benchmark-test", "otoSpeech": "otoearth/otoSpeech-full-duplex-turn-104h"}
for sub in (sys.argv[1:] or ["test", "otoSpeech"]):
    rid = targets[sub]; t = time.time(); print("==", rid, "→", f"{root}/{sub}", flush=True)
    p = snapshot_download(rid, repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir=f"{root}/{sub}", max_workers=8)
    print("done", p, f"{time.time()-t:.0f}s", flush=True)
