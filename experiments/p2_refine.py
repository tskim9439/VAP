#!/usr/bin/env python
"""Phase 2 Q0 — 정렬 결과를 합치고 화자 채널 VAD 로 발화 경계를 보정해 `<corpus>.refined.dialogues.jsonl` 을 만든다 (tokens 내장, 이후 단계는 align_dir 불필요).
  python experiments/p2_refine.py --dialogues <dir>/<corpus>.dialogues.jsonl --align <dir>/align-asr-tn-v1/<corpus> [--no-vad] [--limit N]
보고: 발화 수, 분할 수, 당겨진 시간 합, VAD 없는 발화 수, 보정 전후 발화 길이 중앙값. GPU 불필요(채널 오디오를 한 번 읽는다)."""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from vapasr.data.dialogue import Dialogue
from vapasr.data.dialogue_dataset import load_align_parts
from vapasr.data.dialogue_refine import channel_vad, refine_utterances
ap = argparse.ArgumentParser(); ap.add_argument("--dialogues", required=True); ap.add_argument("--align", required=True); ap.add_argument("--no-vad", action="store_true"); ap.add_argument("--limit", type=int); ap.add_argument("--out")
ap.add_argument("--workers", type=int, default=0, help="대화 병렬 처리 프로세스 수(0 = min(16, CPU/2)). 채널 VAD 가 대화당 수 초라 순차로는 1,500 대화에 2 시간")
ap.add_argument("--min-cps", type=float, default=1.5, help="KO 글자/초(EN 은 /2.2) 하한 — 1 s 이상 발화에서 이보다 성기면 라벨 결손 의심(예: 71631 의 3.9 s 짜리 '이')")
ap.add_argument("--asr-flags", help="p2_asr_check 의 *.utts.jsonl — err ≥ --asr-thr 면 전사 감독 제외(asr_disagree; 시각·활동·ONSET/EOT 는 유지, Dataset 이 그 구간 손실을 마스크)"); ap.add_argument("--asr-thr", type=float, default=0.2)
a = ap.parse_args()
asr = {}
if a.asr_flags:
    for line in open(a.asr_flags, encoding="utf-8"):
        r = json.loads(line); asr[(r["conv"], r["utt"])] = r
aligned, failed = load_align_parts(a.align); outp = a.out or a.dialogues.replace(".dialogues.jsonl", ".refined.dialogues.jsonl"); tmp = outp + ".tmp"
tot = dict(dialogues=0, utts=0, split=0, tightened_s=0.0, no_vad=0, unaligned=0); before = []; after = []; t0 = time.time()

def one(line):
    """대화 한 줄 → (refined json line, 통계). 워커 프로세스에서 실행(fork: aligned/asr 를 공유)."""
    d = Dialogue.from_json(line); toks = aligned.get(d.conv_id, {}); st = dict(unaligned=0, implausible=0, asr_disagree=0, asr_unchecked=0); bef = []
    for u in d.utterances:
        if u.utt_id in toks: u.tokens = toks[u.utt_id]
        elif u.text: st["unaligned"] += 1
        bef.append(u.end - u.start); dur = u.end - u.start; cps = len(u.text.replace(" ", "")) / max(0.1, dur)
        if u.text and dur >= 1.0 and cps < (a.min_cps if d.lang == "Korean" else a.min_cps * 2.2):
            u.flags = (u.flags or []) + ["implausible_rate"]; u.text = ""; u.tokens = None; st["implausible"] += 1
        r = asr.get((d.conv_id, u.utt_id))
        if r and u.text and r["err"] >= a.asr_thr:
            u.flags = (u.flags or []) + ["asr_disagree"]; u.text = ""; u.tokens = None; st["asr_disagree"] += 1
        elif a.asr_flags and u.text and r is None: st["asr_unchecked"] += 1
    vad = {s: [] for s in d.speakers} if a.no_vad else channel_vad(d)
    d.utterances, rs = refine_utterances(d, vad); d.meta["refined"] = dict(vad=not a.no_vad, **rs)
    st.update({k: rs[k] for k in ("utts", "split", "tightened_s", "no_vad")}); return d.to_json(), st, bef, [u.end - u.start for u in d.utterances]

lines = [l for n, l in enumerate(open(a.dialogues, encoding="utf-8")) if not a.limit or n < a.limit]
W = a.workers or max(1, min(16, (os.cpu_count() or 2) // 2))
import multiprocessing as mp
with open(tmp, "w", encoding="utf-8") as f:
    it = mp.get_context("fork").Pool(W).imap(one, lines, chunksize=2) if W > 1 else map(one, lines)
    for js, st, bef, aft in it:
        f.write(js + "\n"); tot["dialogues"] += 1; before += bef; after += aft
        for k, v in st.items(): tot[k] = tot.get(k, 0) + v
        if tot["dialogues"] % 50 == 0: print(f"  {tot['dialogues']}/{len(lines)} dialogues {time.time()-t0:.0f}s (workers {W})", flush=True)
os.replace(tmp, outp)
rep = dict(tot, tightened_s=round(tot["tightened_s"], 1), dur_median_before=round(float(np.median(before)), 2) if before else None, dur_median_after=round(float(np.median(after)), 2) if after else None, out=outp, workers=W, sec=round(time.time() - t0, 1))
json.dump(rep, open(outp.replace(".jsonl", ".stats.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(rep, ensure_ascii=False))
