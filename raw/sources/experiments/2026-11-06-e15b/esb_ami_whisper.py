#!/usr/bin/env python
"""ESB AMI 단일 화자 평가(slurm/submit-eval-esb-ami.sh 산출)를 OpenASR 리더보드와 같은 Whisper EnglishTextNormalizer 로 다시 맨다(2026-11-06).
WER = 모든 발화의 편집 거리 합 / 참조 단어 합(정규화 뒤 참조가 빈 발화는 따로 센다). 삭제·대치·삽입도 나눠 적는다. 읽기만 한다.
  python esb_ami_whisper.py <결과 폴더> [<결과 폴더> …]
"""
import collections, glob, json, os, sys


def main():
    from whisper_normalizer.english import EnglishTextNormalizer
    from rapidfuzz.distance import Levenshtein
    N = EnglishTextNormalizer()
    for d in sys.argv[1:]:
        rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(d, "predictions-rank*.jsonl"))) for l in open(p, encoding="utf-8")]
        agg = collections.defaultdict(collections.Counter)
        for r in rows:
            ref = N(r["raw_reference"]).split(); hyp = N(r.get("hypothesis") or "").split(); g = agg[str(r.get("delta"))]
            g["utts"] += 1
            if not ref: g["empty_ref"] += 1; g["ins_on_empty"] += len(hyp); continue
            for op in Levenshtein.editops(ref, hyp): g[op.tag] += 1
            g["n"] += len(ref)
        for dl, g in sorted(agg.items()):
            n = max(1, g["n"]); e = g["replace"] + g["delete"] + g["insert"]
            print(f"{os.path.basename(d.rstrip('/'))} δ{dl}: Whisper WER {100 * e / n:.2f} %(대치 {100 * g['replace'] / n:.2f} · 삭제 {100 * g['delete'] / n:.2f} · "
                  f"삽입 {100 * g['insert'] / n:.2f}) · 발화 {g['utts']}(참조 빈 발화 {g['empty_ref']}, 그 삽입 {g['ins_on_empty']}) · 참조 단어 {g['n']}")


if __name__ == "__main__":
    main()
