#!/usr/bin/env python
"""USLM fine-tune run 들의 평가 이력 표 (WER/CER %)."""
import os, glob, json
root = os.path.join(os.environ.get("CKPT_EXP_DIR", "/tmp"), "uslm")
for d in sorted(glob.glob(os.path.join(root, "u05-asr-*"))):
    p = os.path.join(d, "results.json")
    if not os.path.exists(p): continue
    r = json.load(open(p)); a = r.get("args", {})
    print(f"{os.path.basename(d)}  (lr {a.get('lr')}/{a.get('lr_adapter')}, init {'distill' if a.get('init_adapter') else 'random'})")
    for h in r["hist"]:
        cells = [f"{k}:{(v.get('wer', v.get('cer')) * 100):.1f}%" for k, v in h.items() if isinstance(v, dict)]
        print(f"   step {h['step']:5d}  " + "  ".join(cells))
