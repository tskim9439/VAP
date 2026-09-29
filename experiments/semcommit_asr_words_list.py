#!/usr/bin/env python3
"""SEM 중립 ASR words 목록(semcommit_train --asr-words @<목록>) 만들기 — 라벨이 없는 words.jsonl 경로를 한 줄에 하나씩.

  라벨링 파트:  <labels-root>/parts/<part>/words.jsonl(.ok 있음) 중 DONE.json 이 없는 파트(라벨링 전·중단 파트). 라벨이 끝난 파트는 스냅숏 목록
                (semcommit_collect_done.py)이 이미 라벨과 함께 넣으므로 뺀다. speechlm 파트 words.jsonl 은 main·short 풀이 섞여 있어 데이터셋이 길이로 나눈다.
  ASR 전용 루트: <asr-root>/parts/<part>/words.jsonl(.ok 있음) — semcommit_build_en_parts.py --part-prefix asr 로 만든 E2 코퍼스 등.
.ok 가 없는 파트(쓰는 중)는 건너뛴다. 출력은 원자적으로 쓰고(tmp → rename) 요약(파트 수·행 수·시간)을 표준 출력과 <out>.summary.json 에 남긴다.

  python experiments/semcommit_asr_words_list.py --labels-root .../labels/speechlm-all19-v035 --asr-root .../semcommit-work/asr-words-v1 \\
      --out .../semcommit-work/asr-words-v1/lists/asr-words-20260929.list
"""
import argparse, json, os, sys
from pathlib import Path


def scan(root: Path, need_not_done: bool):
    parts = root / "parts"
    if not parts.is_dir(): return [], dict(parts=0, rows=0, hours=0.0, skipped_done=0, skipped_no_ok=0)
    out, st = [], dict(parts=0, rows=0, hours=0.0, skipped_done=0, skipped_no_ok=0)
    for d in sorted(p for p in parts.iterdir() if p.is_dir()):
        w, ok = d / "words.jsonl", d / "words.jsonl.ok"
        if not (w.is_file() and ok.is_file()): st["skipped_no_ok"] += 1; continue
        if need_not_done and (d / "DONE.json").exists(): st["skipped_done"] += 1; continue
        try: meta = json.loads(ok.read_text())
        except (OSError, ValueError): meta = {}
        c = meta.get("counts") or {}
        st["parts"] += 1; st["rows"] += int(meta.get("rows") or sum(v for k, v in c.items() if k.startswith("kept")) or 0); st["hours"] += float(meta.get("hours") or 0.0)
        out.append(str(w))
    st["hours"] = round(st["hours"], 2)
    return out, st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-root", default=None, help="라벨링 루트(DONE.json 없는 파트만)"); ap.add_argument("--asr-root", action="append", default=[], help="ASR 전용 루트(반복 가능)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    paths, summary = [], {}
    if a.labels_root:
        p, st = scan(Path(a.labels_root), True); paths += p; summary["labels_root_not_done"] = dict(root=a.labels_root, **st)
    for r in a.asr_root:
        p, st = scan(Path(r), False); paths += p; summary[f"asr_root:{r}"] = st
    if not paths: sys.exit("목록이 비었다 — 경로를 확인")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text("".join(x + "\n" for x in paths)); os.replace(tmp, out)
    summary["files"] = len(paths)
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
