#!/usr/bin/env python3
"""Make a small, deterministic, audio-backed review bundle for speechlm QC.

All copied audio is decoded from the immutable source and checked against the
alignment waveform hash.  The bundle is evidence, not a signed approval.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection_audio import decode


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select(review_sample, per_cell):
    counts = Counter()
    selected = []
    for item in rows(review_sample):
        cell = tuple(item["review_cell"])
        if counts[cell] >= per_cell:
            continue
        counts[cell] += 1
        selected.append(item)
    return selected


def _html_card(item):
    esc = lambda value: html.escape(str(value), quote=True)
    flags = ", ".join(item["acoustic_flags"]) or "없음"
    return (f'<article><h2>{esc(item["source"])} · {esc(item["review_cell"][1])}</h2>'
            f'<p>key: <code>{esc(item["key"])}</code> · {item["duration_s"]:.2f}s · '
            f'RMS {item["rms"]:.4f} · clipping {item["clipping_ratio"]:.4f} · {esc(flags)}</p>'
            f'<audio controls preload="none" src="{esc(item["audio_file"])}"></audio>'
            f'<p><b>Qwen 출력:</b> {esc(item["target_text"])}</p>'
            f'<p><b>원전사(참고):</b> {esc(item["original_transcript"])}</p>'
            f'<p>정렬 상태: {esc(item["qc_state"])} · 단어 0 길이: {item["zero_duration_words"]}</p></article>')


def build(qc, audit, results, out, per_cell=5):
    if out.exists():
        raise FileExistsError(out)
    q, a = json.loads((qc / "summary.json").read_text()), json.loads((audit / "summary.json").read_text())
    if not q.get("complete") or not a.get("complete") or a.get("qc_summary_sha256") != sha256(qc / "summary.json"):
        raise ValueError("QC/approval audit is incomplete or inconsistent")
    sample_path = audit / "review-sample.jsonl"
    if sha256(sample_path) != a["review_sample_sha256"]:
        raise ValueError("review sample checksum mismatch")
    selected = select(sample_path, per_cell)
    if not selected:
        raise ValueError("empty review sample")
    by_shard = defaultdict(dict)
    for item in selected:
        shard = item["source_shard"]
        key = item["key"]
        if key in by_shard[shard]:
            raise ValueError(f"duplicate sample key: {key}")
        by_shard[shard][key] = item
    out.mkdir(parents=True)
    (out / "audio").mkdir()
    cards = []
    import soundfile as sf
    for shard in sorted(by_shard):
        wanted = by_shard[shard]
        part = results / shard
        records = {row["key"]: row for row in rows(part / "records.jsonl") if row["key"] in wanted}
        aligned = {row["key"]: row for row in rows(part / "aligned.jsonl") if row["key"] in wanted}
        if set(records) != set(wanted) or set(aligned) != set(wanted):
            raise ValueError(f"review source join failed: {shard}")
        for key in sorted(wanted):
            candidate, source, record = wanted[key], aligned[key], records[key]
            waveform, info = decode(dict(audio=source["audio"], duration_s=source["duration_s"]))
            if info["sha256"] != source["waveform_sha256"] or info["sha256"] != candidate["waveform_sha256"]:
                raise ValueError(f"review audio hash mismatch: {key}")
            acoustic_flags = list(info["review"])
            if info["rms"] < 0.003:
                acoustic_flags.append("very_low_rms")
            zero = sum(float(w["start_s"]) == float(w["end_s"]) for w in source["words"])
            audio_file = f"audio/{len(cards):05d}.wav"
            sf.write(out / audio_file, waveform, 16000, subtype="PCM_16")
            cards.append(dict(key=key, source=source["source"], review_cell=candidate["review_cell"],
                              source_shard=shard, qc_state=candidate["state"], duration_s=source["duration_s"],
                              target_text=source["target_text"],
                              original_transcript=(record.get("source_manifest") or {}).get("original_transcript", ""),
                              audio_file=audio_file, waveform_sha256=info["sha256"],
                              rms=info["rms"], clipping_ratio=info["clipping_ratio"],
                              acoustic_flags=acoustic_flags, zero_duration_words=zero))
            if len(cards) % 20 == 0:
                print(f"REVIEW_BUNDLE {len(cards)}/{len(selected)}", flush=True)
    with (out / "review-items.jsonl").open("x", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    html_body = "\n".join(_html_card(card) for card in cards)
    (out / "index.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>speechlm 승인 표본</title>'
        '<style>body{font:16px sans-serif;max-width:1050px;margin:2rem auto;background:#f8f9fb}'
        'article{background:white;border:1px solid #ccd;padding:1rem;margin:1rem 0;border-radius:8px}'
        'audio{width:100%}code{overflow-wrap:anywhere}p{line-height:1.5}</style>'
        '<h1>speechlm 승인 표본 — 판정 전</h1><p>오디오와 Qwen 전사·원전사를 비교한다. '
        '이 페이지 자체는 학습 승인 기록이 아니다.</p>' + html_body + '</html>', encoding="utf-8")
    report = dict(schema="speechlm-semcommit-review-bundle-v1", complete=True, training_eligible=False,
                  approval_audit_sha256=sha256(audit / "summary.json"), sample_rows=len(cards),
                  sample_by_cell={f"{k[0]}:{k[1]}": v for k, v in Counter(tuple(c["review_cell"]) for c in cards).items()},
                  acoustic_flagged=sum(bool(c["acoustic_flags"]) for c in cards),
                  review_items_sha256=sha256(out / "review-items.jsonl"),
                  html_sha256=sha256(out / "index.html"))
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.per_cell <= 20:
        parser.error("per-cell must be 1..20")
    build(args.qc, args.audit, args.results, args.out, args.per_cell)


if __name__ == "__main__":
    main()
