#!/usr/bin/env bash
# Part-unit semcommit labeling worker (invoked by semcommit-part-label-apex.sbatch; one visible GPU per task).
# Each alignment part is finished end to end — candidate words → QC-pass approval → Stage A → B(EXAONE) → B(Qwen) → C → grade
# → pools + DONE.json — before the next one, so every finished part is usable for training right away
# (experiments/semcommit_collect_done.py). Parts come from parts-order.tsv (DB round-robin) and are dealt out by rank.
# A part with DONE.json is skipped; a failed part prints PART_FAILED and is retried by the next submission (teacher stages
# resume their own rows). Never deletes: retries write into attempt-scoped directories.
# Part locks make concurrent runs safe (a SLURM job next to direct GPU workers, see slurm/semcommit-direct-label.sh): the worker
# holding a part keeps <part>/lock-<owner> fresh (touched every 5 min); a part with a foreign lock fresher than LOCK_STALE_S
# (default 1800 s) is skipped (PART_BUSY), a stale lock (dead run) is taken over. Lock files are never removed.
# Outside SLURM set WORKER_ID/NWORKERS/RUN_ID; ORDER=reverse walks parts-order.tsv from the end.
set -uo pipefail
cd "$PROJECT"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "exactly one visible GPU per worker is required: ${CUDA_VISIBLE_DEVICES:-unset}" >&2; exit 2; }

rank=${SLURM_PROCID:-${WORKER_ID:?SLURM_PROCID or WORKER_ID}}; ntasks=${SLURM_NTASKS:-${NWORKERS:?SLURM_NTASKS or NWORKERS}}
run_id=${SLURM_JOB_ID:-${RUN_ID:?SLURM_JOB_ID or RUN_ID}}
attempt="j${run_id}-r${SLURM_RESTART_COUNT:-0}-t$rank"
owner="${run_id}-r${SLURM_RESTART_COUNT:-0}-t${rank}-$(hostname -s)-$$"
LOCK_STALE_S=${LOCK_STALE_S:-1800}

fresh_foreign_lock() {   # prints a foreign lock fresher than LOCK_STALE_S, if any
  local dest=$1 mine=$2 f now; now=$(date +%s)
  for f in "$dest"/lock-*; do
    [[ -e $f && $f != "$mine" ]] || continue
    (( now - $(stat -c %Y "$f") < LOCK_STALE_S )) && { echo "$f"; return 0; }
  done
  return 1
}

claim_part() {   # 0 = this worker owns the part now
  local dest=$1 mine=$1/lock-$owner f
  fresh_foreign_lock "$dest" "$mine" >/dev/null && return 1      # someone is working on it
  touch "$mine"; sleep 3
  for f in $(fresh_foreign_lock_all "$dest" "$mine"); do          # simultaneous claim: smallest lock name wins
    [[ $f < $mine ]] && return 1
  done
  return 0
}

fresh_foreign_lock_all() {
  local dest=$1 mine=$2 f now; now=$(date +%s)
  for f in "$dest"/lock-*; do
    [[ -e $f && $f != "$mine" ]] || continue
    (( now - $(stat -c %Y "$f") < LOCK_STALE_S )) && echo "$f"
  done
}
T="$PY -u experiments/semcommit_teacher.py"
X="--extra-candidates $EXTRA_CANDIDATES"

spec() {
  case "$1" in
    qwen38) MODEL=$QWEN; KIND=qwen38; NAME=qwen3.8-27b ;;
    exaone4) MODEL=$EXAONE; KIND=exaone4; NAME=exaone4-32b ;;
    *) echo "invalid model kind: $1" >&2; exit 2 ;;
  esac
}

words_dir() {   # the directory whose words.jsonl is complete (words.jsonl.ok), or empty
  local dest=$1 d
  [[ -f $dest/words.jsonl.ok ]] && { echo "$dest"; return; }
  for d in "$dest"/attempt-*; do [[ -f $d/words.jsonl.ok ]] && { echo "$d"; return; }; done
}

run_part() {
  local part=$1 dest=$OUT/parts/$1 wd
  mkdir -p "$dest"
  wd=$(words_dir "$dest")
  if [[ -z $wd ]]; then
    if [[ -e $dest/words.jsonl ]]; then wd=$dest/attempt-$attempt; mkdir -p "$wd"; else wd=$dest; fi
    "$PY" -u experiments/semcommit_build_speechlm_candidates.py --results "$RESULTS" --out-dir "$dest/cand-$attempt" \
      --part "$part" --tokenizer "$QWEN_ASR" || return 1
    [[ -f $dest/cand-$attempt/$part.words.jsonl ]] || { echo "no candidate output for $part" >&2; return 1; }
    "$PY" -u experiments/semcommit_approve_speechlm_words.py --training-eligibility "$APPROVAL_SUMMARY" --qc-split "$QC_SPLIT" \
      --part "$part" --candidates "$dest/cand-$attempt/$part.words.jsonl" --out-words "$wd/words.jsonl" || return 1
  fi
  local w=$wd/words.jsonl
  if [[ -s $w ]]; then
    spec "$A_KIND"
    $T stageA --words "$w" --model "$MODEL" --kind "$KIND" --judge-name "$NAME" --out "$wd/A.jsonl" --gpu 0 --batch-size 8 || return 1
    $T stageB --words "$w" --stageA "$wd/A.jsonl" --model "$EXAONE" --kind exaone4 --judge-name exaone4-32b \
      --out "$wd/B.exaone4-32b.jsonl" $X --gpu 0 --batch-size 16 || return 1
    $T stageB --words "$w" --stageA "$wd/A.jsonl" --model "$QWEN" --kind qwen38 --judge-name qwen3.8-27b \
      --out "$wd/B.qwen3.8-27b.jsonl" $X --gpu 0 --batch-size 16 || return 1
    spec "$C_KIND"
    $T stageC --words "$w" --stageA "$wd/A.jsonl" --model "$MODEL" --kind "$KIND" --judge-name "$NAME" \
      --out "$wd/C.jsonl" $X --gpu 0 --batch-size 16 || return 1
    $T grade --recipe v0.3 --turn-end none $X --thresholds "$THRESHOLDS" --words "$w" --stageA "$wd/A.jsonl" \
      --stageB "$wd/B.exaone4-32b.jsonl" "$wd/B.qwen3.8-27b.jsonl" --stageC "$wd/C.jsonl" \
      --judges-en exaone4-32b,qwen3.8-27b --judges-ko exaone4-32b,qwen3.8-27b \
      --out "$wd/labels.jsonl" --stats "$wd/labels.stats.json" || return 1
    "$PY" -u experiments/semcommit_part_finalize.py --part "$part" --words "$w" --labels "$wd/labels.jsonl" --stats "$wd/labels.stats.json" \
      --pool-dir "$wd/pools-$attempt" --done "$dest/DONE.json" \
      --meta "{\"job\": \"${SLURM_JOB_ID:-}\", \"words_dir\": \"$wd\", \"thresholds\": \"$THRESHOLDS\"}" || return 1
  else
    "$PY" -u experiments/semcommit_part_finalize.py --part "$part" --words "$w" --pool-dir "$wd/pools-$attempt" --done "$dest/DONE.json" \
      --meta "{\"job\": \"${SLURM_JOB_ID:-}\", \"words_dir\": \"$wd\", \"empty\": true}" || return 1
  fi
}

done_n=0; failed_n=0
while IFS=$'\t' read -r part source npass; do
  [[ -n $part ]] || continue
  [[ -f $OUT/parts/$part/DONE.json ]] && continue
  mkdir -p "$OUT/parts/$part"
  claim_part "$OUT/parts/$part" || { echo "PART_BUSY rank=$rank part=$part"; continue; }
  [[ -f $OUT/parts/$part/DONE.json ]] && continue                 # finished while we were claiming
  ( while sleep 300; do touch "$OUT/parts/$part/lock-$owner"; done ) & hb=$!
  if run_part "$part"; then
    done_n=$((done_n + 1)); echo "PART_DONE rank=$rank part=$part source=$source"
  else
    failed_n=$((failed_n + 1)); echo "PART_FAILED rank=$rank part=$part source=$source"
  fi
  kill "$hb" 2>/dev/null; wait "$hb" 2>/dev/null
  if [[ "${MAX_PARTS_PER_RANK:-0}" -gt 0 && "$done_n" -ge "$MAX_PARTS_PER_RANK" ]]; then break; fi
done < <( { if [[ ${ORDER:-forward} == reverse ]]; then tac "$QC_SPLIT/parts-order.tsv"; else cat "$QC_SPLIT/parts-order.tsv"; fi; } |
          awk -F '\t' -v r="$rank" -v n="$ntasks" '((NR-1)%n)==r {print $0}')
echo "RANK_COMPLETE rank=$rank done=$done_n failed=$failed_n"
