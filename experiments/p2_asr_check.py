#!/usr/bin/env python
"""Phase 2 Q0 — 전사 라벨 품질 점검: 발화마다 깨끗한 화자 채널을 Qwen3-ASR 로 인식해 라벨(TN target)과 CER(KO)/WER(EN) 을 잰다.
  python experiments/p2_asr_check.py --dialogues <dir>/aihub71631.dialogues.jsonl [--limit-utts 400] --out <dir>/asrcheck/aihub71631.json
보고: 발화별 오류율 분포(중앙값·p90·≥0.3 비율), 최악 20개(라벨 vs ASR), 발화 길이별. ASR 자체 오류도 섞이므로 코퍼스 간 상대 비교와 이상 발화 식별용이다."""
import os, sys, json, argparse, random, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--dialogues", required=True); ap.add_argument("--out", required=True); ap.add_argument("--limit-utts", type=int, default=400)
ap.add_argument("--gpu", default="0"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min-dur", type=float, default=0.3); ap.add_argument("--batch", type=int, default=64, help="배치 발화 수. 실측(2026-09-15, H200): 32–256 에서 peak 11 GB 로 일정(모델이 내부 배치 고정)"); ap.add_argument("--shard", default=None); ap.add_argument("--mem-frac", type=float, default=0.8, help="프로세스당 GPU 메모리 상한 비율(워커 W 개면 0.8/W)"); a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
import numpy as np, torch
from qwen_asr import Qwen3ASRModel
from vapasr.data.dialogue import Dialogue
from vapasr.data.streams import load_utt_audio, SR
from vapasr.data.textnorm import score_ko, score_en
QWEN = os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B")

def ed(a, b):
    if not a: return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
def err(lang, ref, hyp):
    if lang == "Korean": r, h = score_ko(ref, False), score_ko(hyp, False); return ed(list(r), list(h)) / max(1, len(r))
    r, h = score_en(ref).split(), score_en(hyp).split(); return ed(r, h) / max(1, len(r))

def utt_audio(d, u):
    ref = d.channels[u.speaker]
    if ref.pieces is not None:
        p = min(ref.pieces, key=lambda x: abs(x[1] - u.start)); return load_utt_audio(p[0])
    return load_utt_audio(ref.path, u.start, u.end - u.start)

dlgs = [Dialogue.from_json(l) for l in open(a.dialogues, encoding="utf-8")]
cands = [(d, u) for d in dlgs for u in d.utterances if u.text and (u.end - u.start) >= a.min_dur]
random.Random(a.seed).shuffle(cands); cands = cands[: a.limit_utts]
if a.shard: k, n = (int(x) for x in a.shard.split("/")); cands = cands[k::n]
torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0); TOTAL = torch.cuda.get_device_properties(0).total_memory
model = Qwen3ASRModel.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="cuda"); BASE = torch.cuda.memory_allocated()
print(f"  GPU total {TOTAL/1e9:.0f} GB, cap {a.mem_frac:.0%}, weights {BASE/1e9:.1f} GB", flush=True)
def transcribe(auds, langs):
    global B
    try: return model.transcribe(audio=auds, language=langs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); B = max(8, B // 2)
        if len(auds) == 1: raise
        h = len(auds) // 2; return transcribe(auds[:h], langs[:h]) + transcribe(auds[h:], langs[h:])
rows = []; B = int(a.batch); import time as _t; T0 = _t.time(); i = 0; pre = {}
while i < len(cands):
    batch = cands[i: i + B]; i += len(batch); auds = []
    for d, u in batch:
        try: auds.append((utt_audio(d, u), SR))
        except Exception: auds.append(None)
    keep = [(x, au) for x, au in zip(batch, auds) if au is not None]
    for (d, u), _ in keep:
        if id(u) in pre: r = pre[id(u)]; hyp = r.text if hasattr(r, "text") else str(r); e = err(d.lang, u.text, hyp); rows.append(dict(conv=d.conv_id, utt=u.utt_id, dur=round(u.end - u.start, 2), ref=u.text, hyp=hyp, err=round(e, 3), cps=round(len(u.text.replace(" ", "")) / max(0.1, u.end - u.start), 2)))
    keep = [(x, au) for x, au in keep if id(x[1]) not in pre]
    if not keep: continue
    res = transcribe([au for _, au in keep], [d.lang for (d, _), _ in keep])
    for ((d, u), _), r in zip(keep, res):
        hyp = r.text if hasattr(r, "text") else str(r); e = err(d.lang, u.text, hyp)
        rows.append(dict(conv=d.conv_id, utt=u.utt_id, dur=round(u.end - u.start, 2), ref=u.text, hyp=hyp, err=round(e, 3), cps=round(len(u.text.replace(" ", "")) / max(0.1, u.end - u.start), 2)))
    print(f"  {len(rows)}/{len(cands)}", flush=True)
errs = np.array([r["err"] for r in rows]); durs = np.array([r["dur"] for r in rows])
def q(v, p): return float(np.percentile(v, p)) if len(v) else None
by_dur = {}
for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 1e9)):
    m = (durs >= lo) & (durs < hi)
    if m.any(): by_dur[f"{lo}-{hi if hi < 1e9 else 'inf'}s"] = dict(n=int(m.sum()), med=q(errs[m], 50), p90=q(errs[m], 90), ge30=float((errs[m] >= 0.3).mean()))
if a.shard: a.out = a.out.replace(".json", f".shard{a.shard.replace('/', 'of')}.json")      # 샤드별 파일명(요약·발화별 모두) — 워커들이 같은 파일을 덮어쓰던 버그 수정(2026-09-16)
# 발화별 결과 전량(jsonl) — refine 의 --asr-flags 입력(sbatch 가 샤드를 <corpus>.utts.jsonl 로 합친다)
with open(a.out.replace(".json", ".utts.jsonl"), "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
rep = dict(corpus=os.path.basename(a.dialogues).split(".")[0], lang=dlgs[0].lang if dlgs else None, n=len(rows), err_median=q(errs, 50), err_p90=q(errs, 90), share_ge_0_3=float((errs >= 0.3).mean()) if len(errs) else None,
           share_ge_0_5=float((errs >= 0.5).mean()) if len(errs) else None, by_dur=by_dur, worst=sorted(rows, key=lambda r: -r["err"])[:20], model=QWEN)
rep["sec"] = round(_t.time() - T0, 1); rep["batch_final"] = B; rep["gpu_total_gb"] = round(TOTAL / 1e9, 1); rep["utts_per_s"] = round(len(rows) / max(1e-6, rep["sec"]), 1); rep["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(rep, open(a.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "worst"}, ensure_ascii=False))
for r in rep["worst"][:8]: print(f"  [{r['err']:.2f}] {r['dur']}s REF: {r['ref'][:70]} || ASR: {r['hyp'][:70]}")
