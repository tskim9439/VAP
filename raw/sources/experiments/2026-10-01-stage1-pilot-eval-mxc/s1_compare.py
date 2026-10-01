"""Stage 1 파일럿(q17-s1·q06-s1) vs 기준선(E2-final·v035-d8·Qwen 오프라인·Nemotron RNN-T) — single-turn-vpaa-kspon-v1.
주 지표: 영어 WER(score_en micro) + AA-WER, 한국어 cer_nospace. 짝 bootstrap 2,000 회: VoxPopuli 는 세션 군집, Kspon 은 발화 단위."""
import json, glob, numpy as np, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.aa_wer import aa_summary
R = "/soundai/users/tskim/VAPKT-data/results"
MAN = {json.loads(l)["id"]: json.loads(l) for l in open("/soundai/users/tskim/VAPKT-data/data/evaluation/single-turn-vpaa-kspon-v1/manifest.jsonl")}
def load(*dirs):
    out = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{R}/{d}/predictions*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                if r["id"] not in MAN: continue
                m = r["metrics"][r.get("primary_metric") or ("wer" if r["lang"] == "English" else "cer_nospace")]
                out[(r["dataset"], str(r["delta"]), r["id"])] = (m["errors"], m["n_ref"], r)
    return out
M = {"q17-s1": load("single-turn-vpaa-kspon-q17-s1"), "q06-s1": load("single-turn-vpaa-kspon-q06-s1"),
     "E2": load("single-turn-E2-final-v1", "single-turn-vpaa-E2-final"), "v035-d8": load("single-turn-semcommit-v035-d8-v1", "single-turn-vpaa-semcommit-v035-d8"),
     "qwen1.7B-off": load("single-turn-qwen3-asr-1.7b-offline", "single-turn-vpaa-qwen3-asr-1.7b-offline"),
     "qwen0.6B-off": load("single-turn-qwen3-asr-0.6b-offline", "single-turn-vpaa-qwen3-asr-0.6b-offline")}
for name, f in (("rnnt56_0", "predictions-ctx0.jsonl"), ("rnnt56_3", "predictions-ctx3.jsonl")):
    o = {}
    for d in ("single-turn-nemotron-rnnt", "single-turn-vpaa-nemotron-rnnt"):
        for l in open(f"{R}/{d}/{f}"):
            r = json.loads(l)
            if r["id"] in MAN:
                m = r["metrics"]["wer" if r["lang"] == "English" else "cer_nospace"]; o[(r["dataset"], "x", r["id"])] = (m["errors"], m["n_ref"], dict(r, reference=MAN[r["id"]]["reference"]))
    M[name] = o
DS = ["voxpopuli-aa-test", "kspon-eval_clean", "kspon-eval_other"]
def rate(m, ds, dl):
    v = [x for (d, de, i), x in m.items() if d == ds and de == dl]
    if not v: return None, None
    e = sum(x[0] for x in v); n = sum(x[1] for x in v)
    aa = aa_summary([x[2] for x in v])["wer_duration_weighted"] if ds.startswith("vox") else None
    return 100 * e / n, (None if aa is None else 100 * aa), len(v)
print("## 오류율(%) — 영어: WER / AA-WER, 한국어: CER(공백 제외)")
for ds in DS:
    print(f"\n### {ds}")
    for name, m in M.items():
        dls = sorted({de for (d, de, i) in m if d == ds}, key=lambda s: (not s.isdigit(), s))
        cells = []
        for dl in dls:
            r = rate(m, ds, dl)
            if r[0] is None: continue
            cells.append(f"{dl}: {r[0]:.2f}" + (f"/{r[1]:.2f}" if r[1] is not None else "") + f" (n={r[2]})")
        if cells: print(f"{name:14s} " + " | ".join(cells))
rng = np.random.default_rng(7); B = 2000
print("\n## 짝 bootstrap (Δ = 새 모델 − 기준, %p, 95 % CI)")
for new in ("q17-s1", "q06-s1"):
    for base, pairs in (("E2", ("2", "4", "6")), ("v035-d8", ("2", "4", "6")), ("q06-s1", ("2", "4", "6", "final")), ("qwen1.7B-off", ("final",)), ("rnnt56_0", ("2", "4", "6", "final"))):
        if new == base or (new == "q06-s1" and base == "q06-s1"): continue
        for ds in DS:
            for dl in pairs:
                bdl = "offline" if base.startswith("qwen") else ("x" if base.startswith("rnnt") else dl)
                ids = sorted(i for (d, de, i) in M[new] if d == ds and de == dl and (d, bdl, i) in M[base])
                if not ids: continue
                key = (lambda i: MAN[i]["session"]) if ds.startswith("vox") else (lambda i: i)
                gs = sorted({key(i) for i in ids}); gi = {g: k for k, g in enumerate(gs)}; E = np.zeros((len(gs), 3))
                for i in ids:
                    k = gi[key(i)]; E[k, 0] += M[base][(ds, bdl, i)][0]; E[k, 1] += M[new][(ds, dl, i)][0]; E[k, 2] += M[new][(ds, dl, i)][1]
                pa, pb = E[:, 0].sum() / E[:, 2].sum(), E[:, 1].sum() / E[:, 2].sum()
                idx = rng.integers(0, len(gs), (B, len(gs))); Sx = E[idx].sum(1); diff = (Sx[:, 1] - Sx[:, 0]) / Sx[:, 2] * 100
                lo, hi = np.percentile(diff, [2.5, 97.5]); sig = "유의" if lo > 0 or hi < 0 else "n.s."
                print(f"{new} vs {base:12s} {ds:18s} {dl:>5s}(기준 {bdl}): {100*pb:.2f} vs {100*pa:.2f}  Δ {100*(pb-pa):+.2f} [{lo:+.2f},{hi:+.2f}] {100*(pb/pa-1):+.1f}% {sig}")
