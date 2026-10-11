"""FLEURS English (en-de, en-ko rows) gain-normalized variant: every en_us test wav scaled to peak 0.5 (gain only, no other processing).
New files only: audio/fleurs-peak05/en_us/<file>.wav, manifest-max30s-fleurs-en-peak05.jsonl (dataset/id suffixed '-peak05')."""
import json, os
from pathlib import Path
import numpy as np, soundfile as sf
A = Path("/soundai/users/tskim/VAPKT-data/data/evaluation/ast-fleurs-covost2-v1")
out_dir = A / "audio" / "fleurs-peak05" / "en_us"; out_dir.mkdir(parents=True, exist_ok=True)
dst = A / "manifest-max30s-fleurs-en-peak05.jsonl"; assert not dst.exists(), dst
rows, gains = [], {}
for l in open(A / "manifest-max30s-fleurs.jsonl"):
    r = json.loads(l)
    if r["lang"] != "English": continue
    src = Path(r["path"]); p = out_dir / src.name
    if src.name not in gains:
        x, sr = sf.read(str(src), dtype="float32"); assert sr == 16000 and x.ndim == 1
        g = 0.5 / max(float(np.abs(x).max()), 1e-6); gains[src.name] = g
        if not p.exists():
            tmp = out_dir / (src.name + ".tmp.wav"); sf.write(str(tmp), np.clip(x * g, -1, 1), 16000, subtype="PCM_16"); os.replace(tmp, p)
    ds = r["dataset"] + "-peak05"
    rows.append(dict(r, id=ds + "/" + r["id"].split("/", 1)[1], dataset=ds, path=str(p), gain=gains[src.name], source_path=str(src)))
with open(dst, "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
g = np.array(list(gains.values()))
print(f"rows {len(rows)} files {len(gains)} gain median {np.median(g):.1f}x p10 {np.percentile(g, 10):.1f}x p90 {np.percentile(g, 90):.1f}x")
