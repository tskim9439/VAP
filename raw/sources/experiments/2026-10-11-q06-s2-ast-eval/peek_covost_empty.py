import glob, json, collections, statistics as st
import numpy as np, soundfile as sf
R = "/soundai/users/tskim/VAPKT-data/results/ast-covost2-q06-s2-ast"
rows = [json.loads(l) for f in sorted(glob.glob(f"{R}/predictions-rank*.jsonl")) for l in open(f)]
d4 = [r for r in rows if r["dataset"] == "covost2-en-de" and str(r["delta"]) == "4"]
silent = [r for r in d4 if not r["token_ids"]]; tonly = [r for r in d4 if r["token_ids"] and not r["translation_hypothesis"]]
print(f"en-de δ4: n={len(d4)} silent={len(silent)} transcript-only={len(tonly)}")
def rms(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = x.mean(1); return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12), float(np.abs(x).max())
for lab, rs in (("silent", silent[:200]), ("ok", [r for r in d4 if r["translation_hypothesis"]][:200])):
    v = [rms(r["path"]) for r in rs]
    print(f"  {lab:7s} n={len(v)} RMS median {st.median(a for a, _ in v):.1f} dBFS, peak median {st.median(b for _, b in v):.3f}, frac RMS<-50dB {np.mean([a < -50 for a, _ in v]):.2f}, dur median {st.median(r['audio_s'] for r in rs):.1f}s")
allv = [rms(r["path"])[0] for r in d4[::20]]
print(f"  all en (every 20th, n={len(allv)}): frac RMS<-50 dBFS {np.mean([a < -50 for a in allv]):.3f}")
