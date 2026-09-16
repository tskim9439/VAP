#!/usr/bin/env python
"""Phase 2 Q0→D1 — 보정본(refined.dialogues.jsonl)에서 학습 창 목록과 창 통계를 만들고 32 창 overfit 셋을 고른다 (task-phase2-data-prep 순서 9, 정본 §13 D1).

  python experiments/p2_build_windows.py --dir <phase2 dir> --tokenizer $MXC_QWEN_ASR_DIR [--hop 10] [--sample 1000] [--overfit 32]
출력(<dir>/windows/): <corpus>.windows.jsonl (conv_id, t0, L — 재현 가능한 창 목록), <corpus>.stats.json (창 수·시간·제외 사유·마스크 비율·토큰/태그/EOT 밀도·화자 수),
      overfit32.windows.jsonl (코퍼스 배분·마스크 0·화자 2명 이상 우선) + overfit32.txt (창별 요약). GPU 불필요(토크나이저만).
창 통계는 창마다 시퀀스를 만들어야 하므로 코퍼스당 --sample 개를 무작위로 뽑아 잰다(마스크 비율·토큰 밀도)."""
import os, sys, json, argparse, random, collections, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from vapasr.data.dialogue_dataset import DialogueWindowDataset
from vapasr.data.dialogue import CHUNK_S
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--tokenizer", default=os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B"))
ap.add_argument("--corpora", default="aihub71631,aihub134-1,aihub134-2,otoSpeech,ami,notsofar,icsi"); ap.add_argument("--hop", type=float, default=10.0); ap.add_argument("--window", type=float, nargs=2, default=(20.0, 40.0))
ap.add_argument("--sample", type=int, default=1000); ap.add_argument("--overfit", type=int, default=32); ap.add_argument("--delay", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); out = os.path.join(a.dir, "windows"); os.makedirs(out, exist_ok=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer)
OVERFIT_QUOTA = {"aihub71631": 6, "aihub134-1": 5, "aihub134-2": 4, "otoSpeech": 5, "ami": 4, "notsofar": 4, "icsi": 4}
rng = random.Random(a.seed); cands = {}; summary = {}
for c in a.corpora.split(","):
    path = os.path.join(a.dir, f"{c}.refined.dialogues.jsonl")
    if not os.path.exists(path): print(f"!! {path} 없음"); continue
    t0 = time.time(); ds = DialogueWindowDataset([path], tok, window_s=tuple(a.window), hop_s=a.hop, delays=(a.delay,), seed=a.seed)
    with open(os.path.join(out, f"{c}.windows.jsonl"), "w") as f:
        for cid, t0w, L in ds.items: f.write(json.dumps(dict(conv=cid, t0=t0w, L=L)) + "\n")
    hours = sum(L for _, _, L in ds.items) / 3600; dlg_hours = sum(d.duration_s for d in ds.dlgs.values()) / 3600
    idx = list(range(len(ds))); rng.shuffle(idx); idx = idx[: a.sample]; mask_frac = []; toks = []; tags = []; eots = []; softs = []; nspk = collections.Counter(); reass = 0; rows = []
    for i in idx:
        s = ds.sequence(i, a.delay); K = s["K"]; mk = len(s["masked_chunks"]); st = s["stats"]; info = s["info"]
        mask_frac.append(mk / K); toks.append(st.text); tags.append(st.tags); eots.append(st.eot); softs.append(len(s["soft_pos"])); n = len({e.speaker for e in ds.window_episodes(ds.dlgs[s["cid"]], s["t0"], s["L"])[0]}); nspk[n] += 1; reass += info["alloc"].reassigned
        rows.append(dict(i=i, conv=s["cid"], t0=s["t0"], L=s["L"], K=K, mask=mk / K, text=st.text, spk=n, soft=len(s["soft_pos"]), reass=info["alloc"].reassigned))
    mf = np.array(mask_frac) if mask_frac else np.zeros(1)
    summary[c] = dict(dialogues=len(ds.dlgs), dialogue_hours=round(dlg_hours, 1), windows=len(ds), window_hours=round(hours, 1), yield_ratio=round(hours / max(1e-6, dlg_hours), 3), hop_s=a.hop, window_s=list(a.window),
                      skipped=dict(no_start=ds.stats["no_start"], sparse=ds.stats["skipped_sparse"], untranscribed=ds.stats["skipped_untranscribed"], with_mask=ds.stats["windows_with_mask"]),
                      sampled=len(idx), mask_frac=dict(mean=round(float(mf.mean()), 3), p50=round(float(np.median(mf)), 3), p90=round(float(np.percentile(mf, 90)), 3), share_zero=round(float((mf == 0).mean()), 3), share_gt_half=round(float((mf > 0.5).mean()), 3)),
                      text_tokens_per_window=dict(mean=round(float(np.mean(toks)), 1), p10=float(np.percentile(toks, 10)), p90=float(np.percentile(toks, 90))) if toks else None,
                      tags_per_window=round(float(np.mean(tags)), 1) if tags else None, eot_per_window=round(float(np.mean(eots)), 1) if eots else None, soft_eot_per_window=round(float(np.mean(softs)), 1) if softs else None,
                      speakers_per_window=dict(sorted(nspk.items())), reassigned_per_sampled_window=round(reass / max(1, len(idx)), 3), sec=round(time.time() - t0, 1))
    json.dump(summary[c], open(os.path.join(out, f"{c}.stats.json"), "w"), indent=1, ensure_ascii=False); print(c, json.dumps(summary[c], ensure_ascii=False), flush=True)
    cands[c] = [r for r in rows if r["mask"] == 0 and r["text"] >= 40]
# overfit 셋: 코퍼스 배분, 마스크 0, 텍스트 40 토큰 이상, 화자 수 다양성(2명 이상 우선, 회의는 3명 이상 우선)
chosen = []
for c, q in OVERFIT_QUOTA.items():
    pool = cands.get(c, []); pref = [r for r in pool if r["spk"] >= (3 if c in ("ami", "notsofar", "icsi") else 2)] or pool
    rng.shuffle(pref); chosen += [dict(r, corpus=c) for r in pref[:q]]
chosen = chosen[: a.overfit]
with open(os.path.join(out, f"overfit{a.overfit}.windows.jsonl"), "w") as f:
    for r in chosen: f.write(json.dumps(dict(corpus=r["corpus"], conv=r["conv"], t0=r["t0"], L=r["L"])) + "\n")
with open(os.path.join(out, f"overfit{a.overfit}.txt"), "w", encoding="utf-8") as f:
    for r in chosen: f.write(f"{r['corpus']:11s} {r['conv']:60s} t0={r['t0']:8.1f} L={r['L']:5.1f} K={r['K']:4d} spk={r['spk']} text={r['text']:4d} soft={r['soft']:3d} reass={r['reass']}\n")
print(f"overfit{a.overfit}: {len(chosen)} windows", collections.Counter(r["corpus"] for r in chosen))
json.dump(summary, open(os.path.join(out, "all.stats.json"), "w"), indent=1, ensure_ascii=False)
