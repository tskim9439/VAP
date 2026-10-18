"""Export the existing forced-alignment cache as a portable JSONL manifest.

The source cache is one large gzip-compressed JSON object. Reading its items
incrementally avoids loading a full LibriSpeech/KsponSpeech split into RAM.
"""

import argparse
import gzip
import json
from pathlib import Path


KSPON_REMOTE_ROOT = (
    "/soundai/databricks_build_managed/"
    "1baf7241-0193-4ef8-a79a-b892a4cc792f/KsponSpeech/"
)


def iter_items(cache_path):
    decoder = json.JSONDecoder()
    with gzip.open(cache_path, "rt", encoding="utf-8") as source:
        buffer = ""
        while '"items"' not in buffer:
            chunk = source.read(1024 * 1024)
            if not chunk:
                raise ValueError("alignment cache has no items array")
            buffer += chunk
        start = buffer.find("[", buffer.index('"items"'))
        if start < 0:
            raise ValueError("alignment cache has no items array opening bracket")
        position = start + 1

        while True:
            while position < len(buffer) and buffer[position] in " \r\n\t,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                item, position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                buffer = buffer[position:]
                position = 0
                chunk = source.read(1024 * 1024)
                if not chunk:
                    raise ValueError("alignment cache ended mid-item")
                buffer += chunk
                continue
            yield item
            if position > 1024 * 1024:
                buffer = buffer[position:]
                position = 0


def export_split(root, cache_root, split, local_kspon_root, no_atomic=False):
    cache = cache_root / split / "_items-online-v2.json.gz"
    output = root / split / "aligned-manifest.jsonl.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output if no_atomic else output.with_suffix(output.suffix + ".partial")
    if no_atomic and output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    count = 0
    with gzip.open(temporary, "xt" if no_atomic else "wt", encoding="utf-8", compresslevel=1) as target:
        for item in iter_items(cache):
            if not isinstance(item, dict) or "id" not in item or "tokens" not in item:
                raise ValueError("unexpected alignment item structure")
            for segment in item.get("segments", []):
                source_path = segment.get("path", "")
                if source_path.startswith(KSPON_REMOTE_ROOT):
                    relative = source_path[len(KSPON_REMOTE_ROOT) :]
                    segment["local_audio_relpath"] = relative
                    segment["local_audio_path"] = str(local_kspon_root / relative)
            target.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            target.write("\n")
            count += 1
    if not no_atomic:
        temporary.replace(output)
    print("{}: {} aligned records -> {}".format(split, count, output), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path,
                        help="source align-asr-tn-v1 directory; defaults to ROOT/align-asr-tn-v1")
    parser.add_argument("--local-kspon-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--no-atomic", action="store_true", help="write a new output directly, without renaming on remote storage")
    args = parser.parse_args()
    cache_root = args.cache_root or args.root / "align-asr-tn-v1"
    for split in args.splits:
        export_split(args.root, cache_root, split, args.local_kspon_root, args.no_atomic)


if __name__ == "__main__":
    main()
