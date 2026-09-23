#!/usr/bin/env python3
"""Build an auditable Arrow manifest from selected SpeechLM catalog rows.

The source Arrow and tar trees are read-only. Original transcripts are copied
verbatim; text normalization and teacher transcription are deliberately not
performed here.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc


SCHEMA = pa.schema(
    [
        ("manifest_version", pa.string()),
        ("catalog_id", pa.string()),
        ("corpus", pa.string()),
        ("data_id", pa.string()),
        ("split", pa.string()),
        ("language", pa.string()),
        ("audio_tar", pa.string()),
        ("audio_member", pa.string()),
        ("source_audio_path", pa.string()),
        ("original_transcript", pa.string()),
        ("sample_rate", pa.int32()),
        ("num_channels", pa.int8()),
        ("num_frames", pa.int64()),
        ("duration_sec", pa.float64()),
        ("speaker_id", pa.string()),
        ("speaker_separation_basis", pa.string()),
        ("source_arrow", pa.string()),
        ("source_row_index", pa.int64()),
    ]
)


def parse_literal(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def task_meta(add_info: dict[str, Any]) -> dict[str, Any]:
    nested = add_info.get("task_meta")
    return nested if isinstance(nested, dict) else add_info


_STRING_PATTERNS: dict[str, re.Pattern[str]] = {}
_NUMBER_PATTERNS: dict[str, re.Pattern[str]] = {}


def literal_string_field(value: Any, key: str) -> str | None:
    """Extract one quoted Python-literal field without parsing the full dict."""
    if not isinstance(value, str):
        return None
    pattern = _STRING_PATTERNS.get(key)
    if pattern is None:
        pattern = re.compile(
            rf"['\"]{re.escape(key)}['\"]\s*:\s*(?P<value>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
        )
        _STRING_PATTERNS[key] = pattern
    match = pattern.search(value)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group("value"))
    except (SyntaxError, ValueError):
        return None
    return str(parsed)


def literal_number_field(value: Any, key: str) -> float | None:
    if not isinstance(value, str):
        return None
    pattern = _NUMBER_PATTERNS.get(key)
    if pattern is None:
        pattern = re.compile(rf"['\"]{re.escape(key)}['\"]\s*:\s*([-+0-9.eE]+)")
        _NUMBER_PATTERNS[key] = pattern
    match = pattern.search(value)
    return float(match.group(1)) if match else None


def member_from(row: dict[str, Any]) -> str:
    value = literal_string_field(row.get("add_info"), "tar_path")
    if value:
        return str(value).lstrip("/")
    return (literal_string_field(row.get("content_meta"), "file_name") or "").lstrip("/")


def transcript_from(row: dict[str, Any]) -> str:
    for item in row.get("conversations") or []:
        if item.get("from") == "assistant":
            return str(item.get("value", ""))
    return ""


def split_from(path: str) -> str:
    low = path.lower()
    if any(x in low for x in ("/validation/", "/valid/", "/dev/")):
        return "validation"
    if any(x in low for x in ("/test/", "/evaluation/", "/eval/")):
        return "test"
    return "train"


def speaker_from(catalog_id: str, path: str) -> str | None:
    if catalog_id == "SLM-SPEECH-000035":
        match = re.search(r"/audio/([^/]+)/", path)
        return match.group(1) if match else None
    if catalog_id in {"SLM-SPEECH-000029", "SLM-SPEECH-000031", "SLM-SPEECH-000033"}:
        match = re.search(r"/(S\d{5,})/", path, re.IGNORECASE)
        return match.group(1) if match else None
    return None


class DurationCache:
    def __init__(self, tar_root: Path, max_items: int = 4096):
        self.tar_root = tar_root
        self.max_items = max_items
        self.cache: OrderedDict[tuple[str, str], dict[str, float]] = OrderedDict()

    def get(self, catalog_id: str, member: str) -> float | None:
        prefix = member.split("/", 1)[0]
        key = (catalog_id, prefix)
        values = self.cache.get(key)
        if values is None:
            path = self.tar_root / catalog_id / f"{prefix}.duration.json"
            if not path.exists():
                return None
            with path.open(encoding="utf-8") as handle:
                values = json.load(handle)
            self.cache[key] = values
            self.cache.move_to_end(key)
            while len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
        return float(values[member]) if member in values else None


def iter_rows(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    offset = 0
    wanted = ("data_id", "add_info", "content_meta", "conversations")
    with pa.memory_map(path, "r") as source:
        reader = ipc.open_stream(source)
        for batch in reader:
            indices = [batch.schema.get_field_index(name) for name in wanted]
            for values in zip(*(batch.column(i).to_pylist() for i in indices)):
                yield offset, dict(zip(wanted, values))
                offset += 1


class ShardWriter:
    def __init__(self, output: Path, rows_per_shard: int):
        self.output = output
        self.rows_per_shard = rows_per_shard
        self.rows: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        index = len(self.paths)
        path = self.output / f"data-{index:05d}.arrow"
        table = pa.Table.from_pylist(self.rows, schema=SCHEMA)
        options = ipc.IpcWriteOptions(compression="zstd")
        with pa.OSFile(str(path), "wb") as sink, ipc.new_stream(sink, SCHEMA, options=options) as writer:
            writer.write_table(table, max_chunksize=10_000)
        self.paths.append(path.name)
        self.rows.clear()

    def close(self) -> list[str]:
        self.flush()
        return self.paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalogs", type=Path, default=Path(__file__).with_name("speechlm_single_speaker_catalogs.json"))
    parser.add_argument("--rows-per-shard", type=int, default=250_000)
    parser.add_argument("--limit-per-catalog", type=int)
    parser.add_argument("--catalog-id", action="append", help="build only this accepted catalog (repeatable)")
    parser.add_argument("--source-arrow", type=Path, help="process one data-*.arrow source shard")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.catalogs.read_text(encoding="utf-8"))
    accepted = config["accepted"]
    if args.catalog_id:
        missing = sorted(set(args.catalog_id) - set(accepted))
        if missing:
            raise SystemExit(f"catalog is not accepted: {', '.join(missing)}")
        accepted = {key: accepted[key] for key in args.catalog_id}
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(args.output, args.rows_per_shard)
    durations = DurationCache(args.tar_root)
    counts: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "seconds": 0.0, "missing_duration": 0, "missing_tar": 0})

    for catalog_id, spec in accepted.items():
        source_dir = args.source_root / catalog_id / "train"
        if args.source_arrow:
            candidate = args.source_arrow.resolve()
            if candidate.parent != source_dir.resolve() or not candidate.name.startswith("data-") or candidate.suffix != ".arrow":
                raise RuntimeError(f"source Arrow is outside selected catalog: {candidate}")
            arrow_files = [str(candidate)]
        else:
            arrow_files = sorted(glob.glob(str(source_dir / "data-*.arrow")))
        if not arrow_files:
            raise RuntimeError(f"no source Arrow files: {source_dir}")
        kept = 0
        for arrow_path in arrow_files:
            for row_index, row in iter_rows(arrow_path):
                add_info = row.get("add_info")
                member = member_from(row)
                source_path = literal_string_field(add_info, "audio_path") or literal_string_field(add_info, "id") or ""
                required = spec.get("path_contains")
                if required and required not in (source_path or member):
                    continue
                if not member:
                    continue
                prefix = member.split("/", 1)[0]
                audio_tar = args.tar_root / catalog_id / f"{prefix}.tar"
                duration = literal_number_field(add_info, "wav_seconds")
                if duration is None:
                    duration = durations.get(catalog_id, member)
                if duration is None:
                    counts[catalog_id]["missing_duration"] += 1
                else:
                    duration = float(duration)
                if not audio_tar.exists():
                    counts[catalog_id]["missing_tar"] += 1
                sample_rate = int(spec["sample_rate"])
                writer.add(
                    {
                        "manifest_version": config["schema_version"],
                        "catalog_id": catalog_id,
                        "corpus": spec["name"],
                        "data_id": str(row.get("data_id") or ""),
                        "split": split_from(source_path or member),
                        "language": spec["lang"],
                        "audio_tar": str(audio_tar),
                        "audio_member": member,
                        "source_audio_path": source_path,
                        "original_transcript": transcript_from(row),
                        "sample_rate": sample_rate,
                        "num_channels": 1,
                        "num_frames": round(duration * sample_rate) if duration is not None else None,
                        "duration_sec": duration,
                        "speaker_id": speaker_from(catalog_id, source_path or member),
                        "speaker_separation_basis": spec["basis"],
                        "source_arrow": arrow_path,
                        "source_row_index": row_index,
                    }
                )
                counts[catalog_id]["rows"] += 1
                counts[catalog_id]["seconds"] += duration or 0.0
                kept += 1
                if args.limit_per_catalog and kept >= args.limit_per_catalog:
                    break
            if args.limit_per_catalog and kept >= args.limit_per_catalog:
                break
        print(json.dumps({"catalog_id": catalog_id, **counts[catalog_id]}, ensure_ascii=False), flush=True)

    shards = writer.close()
    summary = {
        "schema_version": config["schema_version"],
        "source_root": str(args.source_root),
        "tar_root": str(args.tar_root),
        "shards": shards,
        "total_rows": sum(x["rows"] for x in counts.values()),
        "total_hours": sum(x["seconds"] for x in counts.values()) / 3600,
        "catalogs": counts,
        "selection": config,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "dataset_info.json").write_text(
        json.dumps({"features": {field.name: str(field.type) for field in SCHEMA}, "splits": {"train": {"num_examples": summary["total_rows"]}}}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"done": True, "rows": summary["total_rows"], "hours": summary["total_hours"], "shards": len(shards)}), flush=True)


if __name__ == "__main__":
    main()
