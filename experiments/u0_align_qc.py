#!/usr/bin/env python
"""U0 정렬 QC — 실패 발화(같은 종료 시각 뭉침 / 비현실적 토큰율)를 찾고, 필터 전후 interleave 통계를 비교한다.
python experiments/u0_align_qc.py [--dirs otoSpeech,aihub-ts01-5] [--max-same 8] [--max-rate 25] [--M 4] [--delay 2]
"""
import os, sys, json, glob, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.interleave import build_interleaved, Specials
from vapasr.uslm.interleave_data import bad_utterance, _read_jsonl
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(os.environ.get("DATA_MANIFEST_DIR", "/data3/tskim/manifests"), "align"))
ap.add_argument("--dirs", default="otoSpeech,aihub-ts01-5,aihub-vs02"); ap.add_argument("--max-same", type=int, default=8); ap.add_argument("--max-rate", type=float, default=25.0)
ap.add_argument("--M", type=int, default=4); ap.add_argument("--delay", type=int, default=2); a = ap.parse_args()
sp = Specials(next_audio=-1, empty_audio=-2, spk=(-3, -4)); out = {}
for d in a.dirs.split(","):
    files = sorted(glob.glob(os.path.join(a.root, d, "*.jsonl")))
    if not files: print(f"== {d}: 없음"); continue
    reasons = collections.Counter(); n_utt = n_tok = bad_tok = 0; worst = []
    for f in files:
        for u in _read_jsonl(f):
            n_utt += 1; n_tok += len(u["tokens"]); r = bad_utterance(u, a.max_same, a.max_rate)
            if r:
                reasons[r] += 1; bad_tok += len(u["tokens"])
                worst.append((len(u["tokens"]), os.path.basename(f), round(u["start"], 1), round(u["end"] - u["start"], 1), u["text"][:40]))
    worst.sort(reverse=True)
    print(f"== {d}: 발화 {n_utt}, 토큰 {n_tok} | 불량 발화 {sum(reasons.values())} ({100*sum(reasons.values())/max(1,n_utt):.2f} %), 불량 토큰 {bad_tok} ({100*bad_tok/max(1,n_tok):.2f} %) {dict(reasons)}", flush=True)
    for w in worst[:3]: print(f"   최악: 토큰 {w[0]}개 {w[1]} t={w[2]}s dur={w[3]}s '{w[4]}'")
    res = {}
    for filt in (False, True):
        tot = dict(chunks=0, tokens=0, ov=0, mb=0)
        for f in files:
            streams = [[], []]; dur = 0.0
            for u in _read_jsonl(f):
                dur = max(dur, u["end"])
                if filt and bad_utterance(u, a.max_same, a.max_rate): continue
                for t in u["tokens"]: streams[u["speaker"]].append((t["id"], t["end_time"]))
            for s in streams: s.sort(key=lambda x: x[1])
            _, st = build_interleaved(streams, dur + 1.0, sp, delay_frames=a.delay, max_per_chunk=a.M)
            tot["chunks"] += st.chunks; tot["tokens"] += st.tokens; tot["ov"] += st.overflow_tokens; tot["mb"] = max(tot["mb"], st.max_backlog)
        tag = "필터후" if filt else "필터전"
        ovp = 100 * tot["ov"] / max(1, tot["tokens"]); mbs = tot["mb"] * 0.08
        print("   %s M=%d: 토큰 %d 이월 %.2f %% max backlog %d chunk (%.1f s)" % (tag, a.M, tot["tokens"], ovp, tot["mb"], mbs), flush=True)
        res[tag] = dict(tokens=tot["tokens"], overflow_frac=ovp / 100, max_backlog=tot["mb"])
    out[d] = dict(utts=n_utt, tokens=n_tok, bad_utts=sum(reasons.values()), bad_tokens=bad_tok, reasons=dict(reasons), interleave=res)
json.dump(out, open(os.path.join(a.root, "align-qc.json"), "w"), indent=1, ensure_ascii=False)
print("saved", os.path.join(a.root, "align-qc.json"))
