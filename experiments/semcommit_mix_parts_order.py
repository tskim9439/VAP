#!/usr/bin/env python3
"""Language-balanced parts order for the part labeling worker (PARTS_ORDER=...): English and Korean parts interleaved so that at any
prefix of the order the main-pool hours of the two languages stay as close as possible — a snapshot of finished parts
(semcommit_collect_done.py) is then a training set with little language skew.

Balance unit: main-pool hours (8–30 s streams; training draws most steps from it — the short pool is capped by --short-step-fraction).
  - finished parts (DONE.json) count as already done: the language behind is served first (catch-up), then the two alternate.
  - English part hours are exact (en-parts/<set>.tsv from semcommit_build_en_parts.py, all main pool).
  - Korean part main hours are not known before labeling: estimated per source DB from its finished parts (mean main h/part),
    else the mean over all finished Korean parts.
Inside a language the order is kept: Korean = the QC split's DB round-robin order; English = sets interleaved in proportion to
their hours (smooth weighted round-robin), so a partial run samples every set. Parts already done are left out.
Writes <out>.tsv (part \t source \t rows — the worker's 3-column format) and <out>.summary.json (the projected balance curve).

  python experiments/semcommit_mix_parts_order.py --ko-order <QC_SPLIT>/parts-order.tsv --en-parts <OUT>/en-parts \\
      --labels-root <OUT> --source-map <W>/claude-code-v034-20260925/part-source-map.json --out <OUT>/orders/mixed-20260928
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def done_parts(labels_root):
    """{part: (lang, main_hours)} for finished parts (main hours from the main pool words file)."""
    out = {}
    for f in sorted((Path(labels_root) / "parts").glob("*/DONE.json")):
        d = json.loads(f.read_text()); part = f.parent.name
        main = (d.get("files") or {}).get("main")
        h, lang = 0.0, "English" if part.startswith("en-") else "Korean"
        if main:
            for line in open(main["words"], encoding="utf-8"):
                h += json.loads(line)["duration_s"] / 3600
        out[part] = (lang, h)
    return out


def weighted_interleave(queues, weight):
    """Smooth weighted round-robin over {name: [items]} by weight[name] (total hours) — each emitted item advances its queue."""
    q = {k: list(v) for k, v in queues.items() if v}
    sent = defaultdict(float); out = []
    while q:
        k = min(q, key=lambda n: (sent[n] + 1e-9) / weight[n])
        item = q[k].pop(0); out.append(item); sent[k] += item[3]
        if not q[k]:
            del q[k]
    return out


def mix(ko_order, en_dir, labels_root, source_map=None):
    done = done_parts(labels_root)
    src = json.loads(Path(source_map).read_text()) if source_map else {}
    per_src = defaultdict(list)
    for part, (lang, h) in done.items():
        if lang == "Korean":
            per_src[(src.get(part) or {}).get("source")].append(h)
    all_ko = [h for v in per_src.values() for h in v]
    mean_all = sum(all_ko) / len(all_ko) if all_ko else 1.0
    est = {s: sum(v) / len(v) for s, v in per_src.items() if v}
    ko = []
    for line in Path(ko_order).read_text().splitlines():
        if not line.strip():
            continue
        part, source, npass = line.split("\t")[:3]
        if part not in done:
            ko.append((part, source, npass, est.get(source, mean_all)))
    en_q, en_w = {}, {}
    for f in sorted(Path(en_dir).glob("*.tsv")):
        rows = [l.split("\t") for l in f.read_text().splitlines() if l.strip()]
        items = [(p, s, n, float(h)) for p, s, n, h in rows if p not in done]
        if items:
            en_q[f.stem] = items; en_w[f.stem] = sum(x[3] for x in items)
    en = weighted_interleave(en_q, en_w)
    cum = {"English": sum(h for l, h in done.values() if l == "English"), "Korean": sum(h for l, h in done.values() if l == "Korean")}
    start = dict(cum)
    order, curve, i, j, last_en = [], [], 0, 0, 0
    while i < len(en) or j < len(ko):
        take_en = j >= len(ko) or (i < len(en) and cum["English"] <= cum["Korean"])
        item = en[i] if take_en else ko[j]
        lang = "English" if take_en else "Korean"
        i, j = (i + 1, j) if take_en else (i, j + 1)
        cum[lang] += item[3]; order.append(item); last_en = len(order) if take_en else last_en
        if len(order) % 100 == 0 or (take_en and i == len(en)):
            curve.append(dict(parts=len(order), en_h=round(cum["English"], 1), ko_h=round(cum["Korean"], 1)))
    summary = dict(done_parts=len(done), done_main_hours={k: round(v, 1) for k, v in start.items()}, remaining_parts=len(order),
                   en_parts=len(en), ko_parts=len(ko), en_main_hours=round(sum(x[3] for x in en), 1),
                   ko_main_hours_est=round(sum(x[3] for x in ko), 1), ko_est_main_h_per_part={k or "?": round(v, 3) for k, v in est.items()},
                   en_sets={k: dict(parts=len(v), hours=round(en_w[k], 1)) for k, v in en_q.items()},
                   balanced_until_part=last_en, curve=curve)
    return order, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ko-order", required=True, help="QC split parts-order.tsv (part, source, npass)")
    p.add_argument("--en-parts", required=True, help="en-parts directory of semcommit_build_en_parts.py (<set>.tsv)")
    p.add_argument("--labels-root", required=True, help="labels root with parts/*/DONE.json")
    p.add_argument("--source-map", default=None, help="part-source-map.json (Korean part → source DB) for per-DB estimates")
    p.add_argument("--out", required=True, help="output prefix: <out>.tsv and <out>.summary.json (must not exist)")
    a = p.parse_args(argv)
    tsv, js = Path(a.out + ".tsv"), Path(a.out + ".summary.json")
    if tsv.exists() or js.exists():
        raise SystemExit(f"refusing to overwrite {tsv}")
    order, summary = mix(a.ko_order, a.en_parts, a.labels_root, a.source_map)
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text("".join(f"{p}\t{s}\t{n}\n" for p, s, n, _ in order))
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("curve", "ko_est_main_h_per_part")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
