"""정렬 완료 후 MonoStreamDataset 항목 캐시(_items.json.gz)를 미리 만든다 — 학습 job 의 rank 0 가 수십만 파일을 읽으며 기다리지 않도록.
python experiments/s2_prewarm_items.py librispeech-960 kspon-full librispeech-dev librispeech-test kspon-dev kspon-eval"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from vapasr.uslm.mono_data import MonoStreamDataset
tok = AutoTokenizer.from_pretrained(os.environ["MXC_QWEN_ASR_DIR"])
for name in sys.argv[1:]:
    t = time.time(); ds = MonoStreamDataset([name], tok, mode="stream"); print(f"  {name:18s} stream {len(ds):7d} (drop {ds.dropped}, no-align {ds.no_align}) {time.time()-t:.0f}s | {ds.align_dirs[name]}", flush=True)
    t = time.time(); du = MonoStreamDataset([name], tok, mode="utt"); print(f"  {name:18s} utt    {len(du):7d} {time.time()-t:.0f}s", flush=True)
