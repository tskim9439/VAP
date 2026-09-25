#!/usr/bin/env bash
# Invoked by semcommit-main-label-apex-n2.sbatch: one visible GPU per task.
set -euo pipefail
cd "$PROJECT"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "Slurm must bind exactly one visible GPU per task: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
}

spec() {
  case "$1" in
    qwen38) MODEL=$QWEN; KIND=qwen38; NAME=qwen3.8-27b ;;
    exaone4) MODEL=$EXAONE; KIND=exaone4; NAME=exaone4-32b ;;
    *) echo "invalid model kind: $1" >&2; exit 2 ;;
  esac
}

rank=${SLURM_PROCID:?}
count=0
while IFS=$'\t' read -r pool words; do
  [[ -n "$pool" && -n "$words" ]] || continue
  [[ "$pool" == short || "$pool" == long ]] || exit 2
  [[ -f "$words" ]] || exit 2
  name=$(basename "$words" .jsonl)
  dest="$OUT/$pool/$name"
  mkdir -p "$dest"
  "$PY" - "$words" "$pool" "$APPROVAL_SUMMARY" <<'PY'
import json, sys
words, pool, approval = sys.argv[1:]
approved = json.load(open(approval))
fp = approved.get("approval_fingerprint") or approved.get("fingerprint")
if not fp: raise SystemExit("approval summary has no fingerprint")
n = 0
for line in open(words, encoding="utf-8"):
    if not line.strip(): continue
    r = json.loads(line); n += 1
    if (r.get("set") != "speechlm-partial" or r.get("training_eligible") is not True
            or r.get("candidate_only") is not False or r.get("approval_fingerprint") != fp
            or (r["duration_s"] < 8) != (pool == "short")):
        raise SystemExit(f"unapproved or wrong pool: {r.get('id')}")
if not n: raise SystemExit("empty words shard")
print(f"APPROVED_WORDS {pool} {n} {words}", flush=True)
PY
  spec "$A_KIND"
  "$PY" -u experiments/semcommit_teacher.py stageA --words "$words" --model "$MODEL" --kind "$KIND" --judge-name "$NAME" \
    --out "$dest/A.jsonl" --gpu 0 --batch-size 8
  "$PY" -u experiments/semcommit_teacher.py stageB --words "$words" --stageA "$dest/A.jsonl" \
    --model "$EXAONE" --kind exaone4 --judge-name exaone4-32b --out "$dest/B.exaone4-32b.jsonl" \
    --extra-candidates "$EXTRA_CANDIDATES" --gpu 0 --batch-size 16
  "$PY" -u experiments/semcommit_teacher.py stageB --words "$words" --stageA "$dest/A.jsonl" \
    --model "$QWEN" --kind qwen38 --judge-name qwen3.8-27b --out "$dest/B.qwen3.8-27b.jsonl" \
    --extra-candidates "$EXTRA_CANDIDATES" --gpu 0 --batch-size 16
  spec "$C_KIND"
  "$PY" -u experiments/semcommit_teacher.py stageC --words "$words" --stageA "$dest/A.jsonl" \
    --model "$MODEL" --kind "$KIND" --judge-name "$NAME" --out "$dest/C.jsonl" \
    --extra-candidates "$EXTRA_CANDIDATES" --gpu 0 --batch-size 16
  "$PY" -u experiments/semcommit_teacher.py grade --recipe v0.3 --turn-end none \
    --extra-candidates "$EXTRA_CANDIDATES" \
    --thresholds "$THRESHOLDS" --words "$words" --stageA "$dest/A.jsonl" \
    --stageB "$dest/B.exaone4-32b.jsonl" "$dest/B.qwen3.8-27b.jsonl" --stageC "$dest/C.jsonl" \
    --judges-en exaone4-32b,qwen3.8-27b --judges-ko exaone4-32b,qwen3.8-27b \
    --out "$dest/labels.jsonl" --stats "$dest/labels.stats.json"
  count=$((count + 1))
  echo "RANK_DONE rank=$rank count=$count pool=$pool words=$words"
  if [[ "${MAX_SHARDS_PER_RANK:-0}" -gt 0 && "$count" -ge "$MAX_SHARDS_PER_RANK" ]]; then break; fi
done < <(awk -F '\t' -v r="$rank" -v n="${SLURM_NTASKS:?}" '((NR-1)%n)==r {print $0}' "$WORD_INDEX")
echo "RANK_COMPLETE rank=$rank shards=$count"
