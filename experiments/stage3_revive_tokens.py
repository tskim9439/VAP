#!/usr/bin/env python
"""Stage 3 격리 발화 되살리기 재료(cst_dialogue 모듈 설명 'revive') — 정제(p2_refine)는 REVIVABLE 사유(asr_disagree)로 격리한 발화의
text·tokens 를 지웠지만 utt_id·구간은 남겼다(토큰 없는 발화는 나누지 않는다). 그 발화의 정렬 토큰을 정렬 결과(p2_align parts)에서 다시 뽑는다.
  python experiments/stage3_revive_tokens.py --data <phase2> --corpora aihub71631,aihub134-1,aihub134-2,kec,ztc,nsc --out <phase2>/revive-asr-v1
정렬 폴더: 한국어 <data>/align-asr-tn-v1/<corpus>, 독일어 <data>/align-asr-tn-de-v1/<corpus>(--align corpus=폴더 로 바꾼다).
산출: <out>/<corpus>.revive.jsonl(줄 = {"conv_id", "utts": {utt_id: [[tid, t], …]}}) · <out>/<corpus>.stats.json. 정제 파일은 고치지 않는다. GPU 불필요."""
import argparse, json, os, sys, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.cst_dialogue import REVIVABLE
from vapasr.data.dialogue import Dialogue
from vapasr.data.dialogue_dataset import load_align_parts

DE = {"kec", "ztc", "nsc"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--corpora", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--align", action="append", default=[], help="corpus=정렬 폴더(parts/*.jsonl) — 기본 <data>/align-asr-tn-v1 | align-asr-tn-de-v1/<corpus>")
    ap.add_argument("--extra-align", action="append", default=[], help="더 합칠 정렬 루트(<루트>/<corpus>/parts) — 기본 정렬에 없는 발화만 채운다"
                    "(v2: 정렬 대상이 아니었던 격리 발화를 revive_extra_input.py → p2_align.py 로 따로 정렬)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); over = dict(x.split("=", 1) for x in a.align)
    for c in [x for x in a.corpora.split(",") if x]:
        t0 = time.time(); al = over.get(c) or os.path.join(a.data, "align-asr-tn-de-v1" if c in DE else "align-asr-tn-v1", c)
        aligned, _ = load_align_parts(al); st = Counter(); dst = os.path.join(a.out, f"{c}.revive.jsonl"); dur = 0.0; extra = set()
        for root in a.extra_align:                                                               # 기본 정렬에 없는 발화만 채운다
            if not os.path.isdir(os.path.join(root, c, "parts")): continue
            for cid, m in load_align_parts(os.path.join(root, c))[0].items():
                have = aligned.setdefault(cid, {}); extra.update((cid, u) for u in m if u not in have); have.update({u: t for u, t in m.items() if u not in have})
        with open(dst + ".tmp", "w", encoding="utf-8") as f:
            for line in open(os.path.join(a.data, f"{c}.refined.dialogues.jsonl"), encoding="utf-8"):
                d = Dialogue.from_json(line); toks = aligned.get(d.conv_id, {}); out = {}; miss = set(d.meta.get("missing", []))
                for i, u in enumerate(d.utterances, 1):
                    if not u.flags or u.tokens: continue
                    st["flagged"] += 1
                    if i in miss: st["missing_audio"] += 1; continue                                  # 결손 조각(134-1·134-2): 라벨만 있고 오디오가 없다
                    if not set(u.flags) <= REVIVABLE: st["other_reason"] += 1; continue
                    t = toks.get(u.utt_id)
                    if not t: st["no_align"] += 1; continue
                    out[u.utt_id] = [[int(i), round(float(e), 3)] for i, e in t]; st["revived"] += 1; st["tokens"] += len(t); dur += u.end - u.start
                    st["from_extra_align"] += (d.conv_id, u.utt_id) in extra
                    st["tok_outside_0.3s"] += sum(1 for _, e in t if not u.start - 0.3 <= e <= u.end + 0.3)
                if out: f.write(json.dumps({"conv_id": d.conv_id, "utts": out}, ensure_ascii=False) + "\n"); st["dialogues"] += 1
        os.replace(dst + ".tmp", dst)
        rep = dict(corpus=c, align=al, extra_align=a.extra_align, aligned_dialogues=len(aligned), **st, revived_h=round(dur / 3600, 1), sec=round(time.time() - t0, 1))
        json.dump(rep, open(os.path.join(a.out, f"{c}.stats.json"), "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
