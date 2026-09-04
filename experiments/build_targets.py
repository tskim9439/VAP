#!/usr/bin/env python
"""코퍼스 → 대화별 npz (VAD@50Hz, 유도 이벤트, 메타) + manifest.jsonl. 라벨은 로드 시 파생 (targets.vap_labels 등).

python experiments/build_targets.py --corpus aihub --root <wav root> --label-root <json root> --out $DATA_MANIFEST_DIR/aihub-vs02 [--limit N] [--qc 50]
python experiments/build_targets.py --corpus otoSpeech --root /data3/tskim/corpora/turnbench/otoSpeech --out $DATA_MANIFEST_DIR/otoSpeech [--vad-from label|energy]
"""
import os, sys, json, time, argparse, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data import iter_corpus, vad_frames, pool_vad, vap_labels, time_to_next_onset, derive_events, EVENT_TYPES
from vapasr.data.vad import energy_vad, frames_to_segments
from vapasr.data.corpora import SPEECH_LABELS

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True); ap.add_argument("--root", required=True); ap.add_argument("--label-root", default=None)
ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--qc", type=int, default=0)
ap.add_argument("--vad-from", default="label"); ap.add_argument("--frame-hz", type=int, default=50)
ap.add_argument("--only", default=None, help="쉼표로 구분한 conversation id — 이것만 (재)생성, manifest 는 append")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); qc_dir = os.path.join(a.out, "qc"); os.makedirs(qc_dir, exist_ok=True)
FH = a.frame_hz; only = set(a.only.split(",")) if a.only else None
man = open(os.path.join(a.out, "manifest.jsonl"), "a" if only else "w"); stats = dict(n=0, hours=0.0, events={k: 0 for k in EVENT_TYPES}, vad_agree=[], speech_ratio=[])
t0 = time.time(); qc_pool = []
for conv in iter_corpus(a.corpus, a.root, a.limit, load_audio=True, label_root=a.label_root or a.root, vad_from=a.vad_from, only=only):
    if only and conv.id not in only: continue
    va = vad_frames(conv.vad, FH, conv.duration)
    ev = derive_events(va, FH); ttn = time_to_next_onset(va, FH)
    # 에너지 VAD vs 라벨 VAD 일치도 (라벨이 있는 코퍼스에서만) — 프레임 단위 accuracy
    if conv.source != "aihub" and conv.audio is not None:
        e = np.stack([energy_vad(conv.audio[c], 16000, 1000 / FH) for c in (0, 1)], 1).astype(np.float32)
        n = min(len(e), len(va)); stats["vad_agree"].append(float((e[:n] == va[:n]).mean()))
    path = os.path.join(a.out, f"{conv.id}.npz")
    np.savez_compressed(path, vad=va.astype(np.uint8), tau_bin=ttn["bin"].astype(np.int8), censored=ttn["censored"],
                        events=np.array([(t, s, EVENT_TYPES.index(k)) for t, s, k in ev], dtype=np.float32).reshape(-1, 3),
                        frame_hz=FH, duration=conv.duration)
    man.write(json.dumps(dict(id=conv.id, source=conv.source, npz=path, duration=conv.duration, vad_source=conv.vad_source,
                              n_utts=len(conv.utterances), meta={k: v for k, v in conv.meta.items() if isinstance(v, (str, int, float, list))}), ensure_ascii=False) + "\n")
    stats["n"] += 1; stats["hours"] += conv.duration / 3600; stats["speech_ratio"].append(float(va.mean()))
    for _, _, k in ev: stats["events"][k] += 1
    if a.qc and len(qc_pool) < a.qc * 3 and conv.audio is not None:
        qc_pool.append((conv, va, ev))
    if stats["n"] % 20 == 0: print(f"  {stats['n']} conv, {stats['hours']:.1f} h, {time.time()-t0:.0f}s", flush=True)
man.close()

# ── QC: 무작위 20 s 창 — 파형 + VAD + 이벤트 오버레이 ──
if a.qc and qc_pool:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    random.seed(0); picks = random.sample(qc_pool, min(a.qc, len(qc_pool)))
    for k, (conv, va, ev) in enumerate(picks):
        L = 20.0; s0 = random.uniform(0, max(0, conv.duration - L)); s1 = s0 + L
        fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
        col = {"SHIFT": "red", "HOLD": "gray", "INTERRUPT": "purple", "BACKCHANNEL": "green"}
        for c in (0, 1):
            ax = axes[c]; i0, i1 = int(s0 * 16000), int(s1 * 16000); x = conv.audio[c, i0:i1]
            ax.plot(np.linspace(s0, s1, len(x)), x, lw=0.3, color="k"); ax.set_ylim(-1, 1)
            f0, f1 = int(s0 * FH), min(int(s1 * FH), len(va)); ax.fill_between(np.arange(f0, f1) / FH, -1, 1, where=va[f0:f1, c] > 0, color="tab:blue", alpha=0.15, step="post")
            for u in conv.utterances:
                if u.speaker == c and u.end > s0 and u.start < s1 and u.label:
                    ax.axvspan(max(u.start, s0), min(u.end, s1), ymin=0.92, ymax=1.0, color="orange" if u.label in SPEECH_LABELS or conv.source == "aihub" else "brown", alpha=0.6)
            for t, sp, kind in ev:
                if sp == c and s0 <= t <= s1: ax.axvline(t, color=col[kind], lw=1.5, alpha=0.9); ax.text(t, 0.7, kind[:3], color=col[kind], fontsize=7, rotation=90)
            ax.set_ylabel(f"spk {c}")
        axes[0].set_title(f"{conv.id} [{conv.source}, vad={conv.vad_source}] {s0:.1f}-{s1:.1f}s   파랑=VAD 주황=라벨구간(위)  빨강=SHIFT 회색=HOLD 보라=INT 초록=BC")
        fig.tight_layout(); fig.savefig(os.path.join(qc_dir, f"qc-{k:02d}-{conv.id}.png"), dpi=90); plt.close(fig)
    print(f"QC PNG {len(picks)}개 → {qc_dir}")

def q(v): return dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)), p10=float(np.percentile(v, 10))) if v else None
stats["vad_agree"] = q(stats["vad_agree"]); stats["speech_ratio"] = q(stats["speech_ratio"]); stats["sec"] = time.time() - t0
print(json.dumps(stats, indent=1, ensure_ascii=False)); json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1, ensure_ascii=False)
