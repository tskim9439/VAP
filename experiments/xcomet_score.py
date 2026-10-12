#!/usr/bin/env python3
"""스트리밍 번역 예측(simulst_common.py 형식, predictions-*.jsonl)을 xCOMET 으로 채점한다 (2026-10-12).

- 입력 삼중항: src = 평가 manifest 의 원문 전사(raw_reference), mt = translation_hypothesis, ref = translation_reference.
  같은 삼중항은 한 번만 채점한다(시스템·설정 사이에 같은 출력이 많다). 채점한 삼중항은 --cache(jsonl)에 쌓아 다음 실행에서 다시 쓴다.
- 모델: Unbabel/XCOMET-XL(또는 -XXL)의 checkpoints/model.ckpt. 오프라인이므로 hparams.yaml 의 pretrained_model(facebook/xlm-roberta-xl)의
  config·토크나이저가 HF 캐시(HF_HUB_CACHE)에 있어야 한다(local_files_only=True).
- 출력: (system, setting, dataset) 마다 평균 점수(×100), 문장 수, 빈 번역 수.

usage(comet overlay env, GPU 한 장):
  python experiments/xcomet_score.py --checkpoint CKPT --pred-dir DIR [DIR ...] --manifest M [M ...] --cache CACHE.jsonl --out OUT.json
       [--keep q06-s3-ast/d4 seamless-streaming ...]   (system 또는 system/setting 앞부분이 맞는 그룹만)
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulst_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pred-dir", nargs="+", required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    src_of = {r["id"]: r["raw_reference"] for r in C.load_rows(a.manifest)}
    rows = []
    for d in a.pred_dir:
        for p in sorted(glob.glob(os.path.join(d, "predictions-*.jsonl"))):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                name = f"{r['system']}/{r['setting']}"
                if a.keep and not any(name == k or name.startswith(k + "/") or r["system"] == k for k in a.keep): continue
                rows.append(r)
    key = lambda r: (src_of[r["id"]], r["translation_hypothesis"].strip(), r["translation_reference"])
    cache = {}
    if os.path.exists(a.cache):
        for line in open(a.cache, encoding="utf-8"):
            c = json.loads(line); cache[(c["src"], c["mt"], c["ref"])] = c["score"]
    todo = sorted({key(r) for r in rows} - set(cache))
    print(f"예측 {len(rows)} 행 · 삼중항 {len({key(r) for r in rows})} · 새로 채점 {len(todo)}", flush=True)
    if todo:
        from comet import load_from_checkpoint
        model = load_from_checkpoint(a.checkpoint, local_files_only=True)
        step = 4096                                                                 # 조금씩 채점해 캐시에 쌓는다(중단돼도 이어서)
        with open(a.cache, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), step):
                part = todo[i:i + step]
                out = model.predict([dict(src=s, mt=m, ref=t) for s, m, t in part], batch_size=a.batch_size, gpus=1, progress_bar=False)
                for (s, m, t), sc in zip(part, out.scores):
                    cache[(s, m, t)] = float(sc); f.write(json.dumps(dict(src=s, mt=m, ref=t, score=float(sc)), ensure_ascii=False) + "\n")
                f.flush(); print(f"PROGRESS {min(i + step, len(todo))}/{len(todo)}", flush=True)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["system"], r["setting"], r["dataset"])].append((cache[key(r)], not r["translation_hypothesis"].strip()))
    rep = {f"{s}/{st}/{ds}": dict(xcomet=round(100 * sum(x for x, _ in v) / len(v), 2), n=len(v), empty_translations=sum(e for _, e in v))
           for (s, st, ds), v in sorted(groups.items())}
    meta = dict(checkpoint=a.checkpoint, manifests=a.manifest, pred_dirs=a.pred_dir, keep=a.keep, triples=len(cache))
    json.dump(dict(meta=meta, groups=rep), open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in rep.items():
        print(f"{k:58s} n={v['n']:5d} xCOMET {v['xcomet']:6.2f} empty={v['empty_translations']}", flush=True)


if __name__ == "__main__":
    main()
