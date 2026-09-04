#!/usr/bin/env python
import json, os
b = json.load(open(os.path.join(os.environ.get("CKPT_EXP_DIR", "/tmp"), "uslm", "baselines.json")))
for k, v in sorted(b.items()): print(f"{k:42s} {v.get('wer', v.get('cer')):.4f}  n={v['n']}")
