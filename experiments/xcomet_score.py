#!/usr/bin/env python3
"""스트리밍 번역 예측(simulst_common.py 형식, predictions-*.jsonl)을 xCOMET 으로 채점한다 (2026-10-12).

- 입력 삼중항: src = 평가 manifest 의 원문 전사(raw_reference), mt = translation_hypothesis, ref = translation_reference.
  같은 삼중항은 한 번만 채점한다(시스템·설정 사이에 같은 출력이 많다). 채점한 삼중항은 --cache(jsonl)에 쌓아 다음 실행에서 다시 쓴다.
- 빈 번역(mt 가 공백뿐)은 모델에 넣지 않고 0 점으로 둔다: XCOMET-XL 은 빈 mt 에 오류 구간을 찾지 못해 1.0 을 준다(2026-10-12 확인 —
  독·영·한 모두 빈 mt 1.0, '.' 0.5, 한 단어 약 0.5). 빈 번역 수는 그룹마다 따로 낸다.
- 모델: Unbabel/XCOMET-XL(또는 -XXL)의 checkpoints/model.ckpt. 오프라인이므로 hparams.yaml 의 pretrained_model(facebook/xlm-roberta-xl)의
  config·토크나이저가 HF 캐시(HF_HUB_CACHE)에 있어야 한다(local_files_only=True).
- 출력: (system, setting, dataset) 마다 평균 점수(×100), 문장 수, 빈 번역 수.

usage(comet overlay env, GPU 한 장):
  python experiments/xcomet_score.py --checkpoint CKPT --pred-dir DIR [DIR ...] --manifest M [M ...] --cache CACHE.jsonl [다른 캐시 ...] --out OUT.json
       [--keep q06-s3-ast/d4 seamless-streaming ...]   (system 또는 system/setting 앞부분이 맞는 그룹만)
       [--shard R N]   (아직 없는 삼중항의 R 번째 1/N 만 채점해 첫 --cache 에 쓴다; GPU 여러 장에 나눌 때. 끝나면 모든 캐시를 주고 다시 돌려 집계)
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
    ap.add_argument("--cache", nargs="+", required=True, help="읽을 캐시들(jsonl); 새 점수는 첫 파일에 덧붙인다")
    ap.add_argument("--shard", type=int, nargs=2, default=None, metavar=("R", "N"))
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
    for cp in a.cache:
        if os.path.exists(cp):
            for line in open(cp, encoding="utf-8"):
                c = json.loads(line); cache[(c["src"], c["mt"], c["ref"])] = c["score"]
    todo = sorted(k for k in {key(r) for r in rows} - set(cache) if k[1])                 # 빈 mt 는 채점하지 않는다(0 점)
    if a.shard: todo = todo[a.shard[0]::a.shard[1]]
    print(f"예측 {len(rows)} 행 · 삼중항 {len({key(r) for r in rows})} · 새로 채점 {len(todo)}", flush=True)
    if todo:
        from comet import load_from_checkpoint
        model = load_from_checkpoint(a.checkpoint, local_files_only=True)
        step = 4096                                                                 # 조금씩 채점해 캐시에 쌓는다(중단돼도 이어서)
        with open(a.cache[0], "a", encoding="utf-8") as f:
            for i in range(0, len(todo), step):
                part = todo[i:i + step]
                out = model.predict([dict(src=s, mt=m, ref=t) for s, m, t in part], batch_size=a.batch_size, gpus=1, progress_bar=False)
                for (s, m, t), sc in zip(part, out.scores):
                    cache[(s, m, t)] = float(sc); f.write(json.dumps(dict(src=s, mt=m, ref=t, score=float(sc)), ensure_ascii=False) + "\n")
                f.flush(); print(f"PROGRESS {min(i + step, len(todo))}/{len(todo)}", flush=True)
    if a.shard:
        print(f"shard {a.shard[0]}/{a.shard[1]} 끝 — 모든 캐시를 주고 --shard 없이 다시 돌려 집계한다", flush=True); return
    missing = [r["id"] for r in rows if key(r)[1] and key(r) not in cache]
    assert not missing, f"채점 안 된 행 {len(missing)} 개(예: {missing[:3]})"
    groups = defaultdict(list)
    for r in rows:
        k = key(r)
        groups[(r["system"], r["setting"], r["dataset"])].append((cache[k] if k[1] else 0.0, not k[1]))
    rep = {f"{s}/{st}/{ds}": dict(xcomet=round(100 * sum(x for x, _ in v) / len(v), 2), n=len(v), empty_translations=sum(e for _, e in v))
           for (s, st, ds), v in sorted(groups.items())}
    meta = dict(checkpoint=a.checkpoint, manifests=a.manifest, pred_dirs=a.pred_dir, keep=a.keep, triples=len(cache), empty_mt_score=0.0)
    json.dump(dict(meta=meta, groups=rep), open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in rep.items():
        print(f"{k:58s} n={v['n']:5d} xCOMET {v['xcomet']:6.2f} empty={v['empty_translations']}", flush=True)


if __name__ == "__main__":
    main()
