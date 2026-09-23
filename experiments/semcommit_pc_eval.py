#!/usr/bin/env python3
"""Semantic commit 의 LLM 라벨 독립 EN 교차 점검 — semcommit_eval.py 스트림 jsonl 의 <SEM_END> 방출을 구두점 전사(words.jsonl 의 pnc_text, LibriSpeech-PC)의
문장 끝(. ? !)·절 경계(, ; :)에 대어 설정별 P_pc(커밋 중 문장 끝 비율)·R_pc(문장 끝 중 커밋된 비율)를 낸다(vapasr/hf/commit_metrics.pc_commit_counts).
pnc_text 가 없는 스트림은 건너뛴다(개수 보고). GPU 불필요.

  python experiments/semcommit_pc_eval.py --words words-ls-test.jsonl --streams eval/r1-d4-ls-test.streams.jsonl eval/oracle-d4-ls-test.streams.jsonl --out pc.json
"""
import argparse, collections, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.hf.commit_metrics import SEM_END_ID, pc_commit_counts, ref_chunk


def score_file(words: dict, path: str) -> dict:
    """스트림 jsonl 하나 → {설정 이름: 합산 카운트 + P_pc/R_pc}."""
    agg, ids, skipped = collections.defaultdict(collections.Counter), collections.defaultdict(set), set()
    for line in open(path):
        r = json.loads(line); w = words.get(r["id"])
        if w is None or not w.get("pnc_text"): skipped.add(r["id"]); continue
        cfg = r["config"]["name"] if isinstance(r["config"], dict) else str(r["config"]); K = int(r["K"]); delta = int(r["delta"])
        ws = sorted(w["words"], key=lambda x: int(x["i"]))
        ev = [e for e in r["hyp"]["events"] if int(e["id"]) == SEM_END_ID]
        ids[cfg].add(r["id"]); agg[cfg].update(pc_commit_counts(ws, w["pnc_text"], r["hyp"]["words"], ev, [ref_chunk(float(x["end_time"]), delta, K) for x in ws], K))
    out = {}
    for cfg, c in agg.items():
        out[cfg] = dict(c, streams=len(ids[cfg]), P_pc=round(c["SENT"] / c["commits"], 4) if c["commits"] else None,
                        R_pc=round(c["sent_hit"] / c["n_sent"], 4) if c["n_sent"] else None)
    return dict(configs=out, skipped_no_pc=len(skipped))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--words", required=True); ap.add_argument("--streams", nargs="+", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    words = {r["id"]: r for r in map(json.loads, open(a.words))}
    res = {}
    for p in a.streams:
        res[p] = s = score_file(words, p)
        print(f"== {Path(p).name} (pnc_text 없는 스트림 {s['skipped_no_pc']} 건너뜀)")
        for cfg, c in sorted(s["configs"].items()):
            f = lambda x: "    -" if x is None else f"{x:5.2f}"
            print(f"  {cfg:11s} commits {c['commits']:4d}  SENT {c['SENT']:4d}  CLAUSE {c['CLAUSE']:4d}  NONE {c['NONE']:4d}  P_pc {f(c['P_pc'])}  R_pc {f(c['R_pc'])}  (문장 끝 {c['n_sent']})")
    if a.out: json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False); print("→", a.out)


if __name__ == "__main__":
    main()
