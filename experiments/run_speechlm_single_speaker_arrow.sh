#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/soundai/DB/speechlm/speechlm_v1_encoder_alignment_2607}
TAR_ROOT=${TAR_ROOT:-/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/DI_LAB_CATALOG}
OUTPUT=${OUTPUT:-/soundai/users/tskim/VAPKT-data/data/speechlm-asr-single-speaker-v1-20260922}
PYTHON_BIN=${PYTHON_BIN:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}
WORKERS=${WORKERS:-48}
RESUME=${RESUME:-0}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG=$SCRIPT_DIR/speechlm_single_speaker_catalogs.json
BUILDER=$SCRIPT_DIR/build_speechlm_single_speaker_arrow.py

if [[ -e "$OUTPUT" && "$RESUME" != 1 ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/catalogs" "$OUTPUT/logs"
cp "$CONFIG" "$OUTPUT/selection-policy.json"

mapfile -t catalog_ids < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import json, sys
for key in json.load(open(sys.argv[1], encoding="utf-8"))["accepted"]:
    print(key)
PY
)

work_list=$OUTPUT/source-shards.tsv
: > "$work_list"
for cid in "${catalog_ids[@]}"; do
  catalog_out=$OUTPUT/catalogs/$cid
  if [[ -f "$catalog_out/summary.json" ]]; then
    continue
  fi
  mkdir -p "$catalog_out/_parts"
  part=0
  for source_arrow in "$SOURCE_ROOT/$cid/train"/data-*.arrow; do
    part_out=$(printf '%s/_parts/%05d' "$catalog_out" "$part")
    if [[ ! -f "$part_out/summary.json" ]]; then
      if [[ -d "$part_out" ]]; then
        find "$part_out" -type f -delete
        find "$part_out" -depth -type d -empty -delete
      fi
      printf '%s\t%05d\t%s\n' "$cid" "$part" "$source_arrow" >> "$work_list"
    fi
    part=$((part + 1))
  done
done

export SOURCE_ROOT TAR_ROOT OUTPUT PYTHON_BIN CONFIG BUILDER
cat "$work_list" | xargs -P "$WORKERS" -n 3 bash -c '
  cid="$1"
  part="$2"
  source_arrow="$3"
  "$PYTHON_BIN" -u "$BUILDER" \
    --source-root "$SOURCE_ROOT" \
    --tar-root "$TAR_ROOT" \
    --catalogs "$CONFIG" \
    --catalog-id "$cid" \
    --source-arrow "$source_arrow" \
    --output "$OUTPUT/catalogs/$cid/_parts/$part" \
    --rows-per-shard 250000 \
    >"$OUTPUT/logs/$cid-$part.log" 2>&1
' _

"$PYTHON_BIN" - "$OUTPUT" "$CONFIG" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
for catalog_id in config["accepted"]:
    catalog = root / "catalogs" / catalog_id
    if (catalog / "summary.json").exists():
        continue
    parts = []
    for path in sorted((catalog / "_parts").glob("*/summary.json")):
        parts.append(json.loads(path.read_text(encoding="utf-8")))
    expected = len(list(Path(parts[0]["source_root"]).joinpath(catalog_id, "train").glob("data-*.arrow"))) if parts else 0
    if len(parts) != expected:
        raise SystemExit(f"incomplete source shards for {catalog_id}: {len(parts)}/{expected}")
    shards = []
    for part_index, part_dir in enumerate(sorted((catalog / "_parts").iterdir())):
        for shard_index, source in enumerate(sorted(part_dir.glob("data-*.arrow"))):
            target = catalog / f"data-p{part_index:05d}-s{shard_index:02d}.arrow"
            source.replace(target)
            shards.append(target.name)
    stats = {"rows": 0, "seconds": 0.0, "missing_duration": 0, "missing_tar": 0}
    for part in parts:
        value = part["catalogs"][catalog_id]
        for key in stats:
            stats[key] += value[key]
    catalog_summary = {
        "schema_version": config["schema_version"],
        "source_root": parts[0]["source_root"],
        "tar_root": parts[0]["tar_root"],
        "shards": shards,
        "total_rows": stats["rows"],
        "total_hours": stats["seconds"] / 3600,
        "catalogs": {catalog_id: stats},
        "selection": config,
    }
    (catalog / "summary.json").write_text(json.dumps(catalog_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

items = []
for path in sorted((root / "catalogs").glob("*/summary.json")):
    items.append(json.loads(path.read_text(encoding="utf-8")))
summary = {
    "manifest_version": "vapasr-speechlm-single-speaker-v1",
    "layout": "catalogs/<catalog_id>/data-*.arrow",
    "catalog_count": len(items),
    "total_rows": sum(item["total_rows"] for item in items),
    "total_hours": sum(item["total_hours"] for item in items),
    "catalogs": {next(iter(item["catalogs"])): next(iter(item["catalogs"].values())) for item in items},
}
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY
