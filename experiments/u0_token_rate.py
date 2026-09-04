#!/usr/bin/env python
"""U0 — Qwen3 tokenizer 기준 토큰율. 발화 단위 tok/s 와 80 ms 프레임당 토큰 수(균등 가정) 분포 → chunk 당 M 예산.
python experiments/u0_token_rate.py  → $DATA_LOG_DIR/u0-token-rate.json
"""
import os, sys, json, glob, re, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.corpora import parse_srt, SPEECH_LABELS
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B")
def _f(x): return float(str(x).replace(",", ""))
rows = {"ko": [], "en": []}
# AI Hub (KO): 라벨 JSON 발화
for js in sorted(glob.glob("/data3/tskim/corpora/aihub/adult-val-labels/**/*.json", recursive=True))[:1500] + sorted(glob.glob("/data3/tskim/corpora/aihub/adult-train-labels/**/*.json", recursive=True))[:1500]:
    d = json.load(open(js, encoding="utf-8"))
    for u in d.get("Conversation", []):
        s, e, t = _f(u["StartTime"]), _f(u["EndTime"]), u.get("Text", "").strip()
        if e - s >= 0.3 and t: rows["ko"].append((len(tok.encode(t, add_special_tokens=False)), e - s, t))
# otoSpeech (EN): SRT speech 범주
for d in sorted(glob.glob("/data3/tskim/corpora/turnbench/otoSpeech/*/"))[:420]:
    for c in (1, 2):
        p = os.path.join(d, f"speaker_{c}_annotation_a.srt")
        if not os.path.exists(p): continue
        for s, e, lab, t in parse_srt(p):
            if lab in SPEECH_LABELS and e - s >= 0.3 and t.strip(): rows["en"].append((len(tok.encode(t.strip(), add_special_tokens=False)), e - s, t.strip()))
out = {}
for lang, r in rows.items():
    n = np.array([x[0] for x in r]); dur = np.array([x[1] for x in r]); rate = n / dur; per_frame = rate * 0.08
    q = lambda v, p: float(np.percentile(v, p))
    out[lang] = dict(utterances=len(r), tokens=int(n.sum()), hours=float(dur.sum() / 3600),
                     tok_per_s=dict(mean=float(rate.mean()), p50=q(rate, 50), p90=q(rate, 90), p99=q(rate, 99), max=float(rate.max())),
                     tok_per_80ms=dict(mean=float(per_frame.mean()), p50=q(per_frame, 50), p90=q(per_frame, 90), p99=q(per_frame, 99)),
                     frac_rate_gt=dict(**{f">{k}tok/s(M={k/12.5:.0f})": float((rate > k).mean()) for k in (12.5, 25, 37.5, 50)}),
                     token_weighted_frac_gt=dict(**{f">{k}": float(n[rate > k].sum() / n.sum()) for k in (12.5, 25, 37.5, 50)}),
                     chars_per_token=float(sum(len(x[2]) for x in r) / max(1, n.sum())))
    fast = sorted(r, key=lambda x: -x[0] / x[1])[:5]; out[lang]["fastest_examples"] = [dict(tok=a, dur=round(b, 2), rate=round(a / b, 1), text=c[:60]) for a, b, c in fast]
print(json.dumps(out, indent=1, ensure_ascii=False))
p = os.path.join(os.environ.get("DATA_LOG_DIR", "/tmp"), "u0-token-rate.json"); json.dump(out, open(p, "w"), indent=1, ensure_ascii=False); print("saved", p)
