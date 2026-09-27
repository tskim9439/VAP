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
# PARTS_ORDER (default $QC_SPLIT/parts-order.tsv): another part list, e.g. the language-balanced EN+KO order of
# experiments/semcommit_mix_parts_order.py. English parts (en-*) come with words.jsonl(.ok) prebuilt by semcommit_build_en_parts.py.
# Groups (GROUP_PARTS, default 8): the worker claims up to GROUP_PARTS parts, builds their words, then runs every model stage of
# the whole group in ONE teacher process (semcommit_teacher.py multi) ordered by model — Stage A, B and C with the A/C judge,
# then B with the other judge — so each 27–32B judge is loaded once per group instead of once per stage per part (the NFS load
# was ≈ 9–11 min × 4 per part, 64 % of a small part's time). Grade + finalize (DONE.json) then run per part; a part whose
# job failed is PART_FAILED while the rest of the group still finishes. Job lists/results: $OUT/multi/ (never removed).
set -uo pipefail
cd "$PROJECT"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "exactly one visible GPU per worker is required: ${CUDA_VISIBLE_DEVICES:-unset}" >&2; exit 2; }

rank=${SLURM_PROCID:-${WORKER_ID:?SLURM_PROCID or WORKER_ID}}; ntasks=${SLURM_NTASKS:-${NWORKERS:?SLURM_NTASKS or NWORKERS}}
run_id=${SLURM_JOB_ID:-${RUN_ID:?SLURM_JOB_ID or RUN_ID}}
attempt="j${run_id}-r${SLURM_RESTART_COUNT:-0}-t$rank"
owner="${run_id}-r${SLURM_RESTART_COUNT:-0}-t${rank}-$(hostname -s)-$$"
LOCK_STALE_S=${LOCK_STALE_S:-1800}
BS_A=${BS_A:-16}; BS_BC=${BS_BC:-32}   # H200 143 GB: 27–32B bf16 ≈ 54–64 GB; OOM halves the batch automatically (LLM.max_batch)
GROUP_PARTS=${GROUP_PARTS:-8}
PARTS_ORDER=${PARTS_ORDER:-$QC_SPLIT/parts-order.tsv}
export BS_A BS_BC

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

words_dir() {   # the directory whose words.jsonl is complete (words.jsonl.ok), or empty
  local dest=$1 d
  [[ -f $dest/words.jsonl.ok ]] && { echo "$dest"; return; }
  for d in "$dest"/attempt-*; do [[ -f $d/words.jsonl.ok ]] && { echo "$d"; return; }; done
}

prepare_part() {   # builds the part's words.jsonl (candidates → QC-pass approval); prints the words dir
  local part=$1 dest=$OUT/parts/$1 wd
  mkdir -p "$dest"
  wd=$(words_dir "$dest")
  if [[ -z $wd && $part == en-* ]]; then echo "English part $part has no prebuilt words.jsonl.ok (semcommit_build_en_parts.py)" >&2; return 1; fi
  if [[ -z $wd ]]; then
    if [[ -e $dest/words.jsonl ]]; then wd=$dest/attempt-$attempt; mkdir -p "$wd"; else wd=$dest; fi
    "$PY" -u experiments/semcommit_build_speechlm_candidates.py --results "$RESULTS" --out-dir "$dest/cand-$attempt" \
      --part "$part" --tokenizer "$QWEN_ASR" >&2 || return 1
    [[ -f $dest/cand-$attempt/$part.words.jsonl ]] || { echo "no candidate output for $part" >&2; return 1; }
    "$PY" -u experiments/semcommit_approve_speechlm_words.py --training-eligibility "$APPROVAL_SUMMARY" --qc-split "$QC_SPLIT" \
      --part "$part" --candidates "$dest/cand-$attempt/$part.words.jsonl" --out-words "$wd/words.jsonl" >&2 || return 1
  fi
  echo "$wd"
}

finish_part() {   # grade + pools + DONE.json once the part's model stages are complete
  local part=$1 wd=$2 dest=$OUT/parts/$1 w=$2/words.jsonl
  $T grade --recipe v0.3 --turn-end none $X --thresholds "$THRESHOLDS" --words "$w" --stageA "$wd/A.jsonl" \
    --stageB "$wd/B.exaone4-32b.jsonl" "$wd/B.qwen3.8-27b.jsonl" --stageC "$wd/C.jsonl" \
    --judges-en exaone4-32b,qwen3.8-27b --judges-ko exaone4-32b,qwen3.8-27b \
    --out "$wd/labels.jsonl" --stats "$wd/labels.stats.json" || return 1
  "$PY" -u experiments/semcommit_part_finalize.py --part "$part" --words "$w" --labels "$wd/labels.jsonl" --stats "$wd/labels.stats.json" \
    --pool-dir "$wd/pools-$attempt" --done "$dest/DONE.json" \
    --meta "{\"job\": \"${SLURM_JOB_ID:-$run_id}\", \"words_dir\": \"$wd\", \"thresholds\": \"$THRESHOLDS\"}" || return 1
}

finish_empty() {
  local part=$1 wd=$2
  "$PY" -u experiments/semcommit_part_finalize.py --part "$part" --words "$wd/words.jsonl" --pool-dir "$wd/pools-$attempt" \
    --done "$OUT/parts/$part/DONE.json" --meta "{\"job\": \"${SLURM_JOB_ID:-$run_id}\", \"words_dir\": \"$wd\", \"empty\": true}"
}

report() {   # report <ok 0|1> <part>
  local part=$2 src=${SOURCE[$2]:-}
  if [[ $1 == 0 ]]; then done_n=$((done_n + 1)); echo "PART_DONE rank=$rank part=$part source=$src"
  else failed_n=$((failed_n + 1)); echo "PART_FAILED rank=$rank part=$part source=$src"; fi
}

run_group() {   # run_group part... — all model stages of the group in one teacher process (models loaded once each)
  local part wd tag jobs res failed; local -a active=() pairs=()
  for part in "$@"; do
    if ! wd=$(prepare_part "$part"); then report 1 "$part"; continue; fi
    if [[ -s $wd/words.jsonl ]]; then active+=("$part"); pairs+=("$part=$wd")
    elif finish_empty "$part" "$wd"; then report 0 "$part"; else report 1 "$part"; fi
  done
  (( ${#active[@]} )) || return 0
  mkdir -p "$OUT/multi"; tag="$attempt-$(date +%Y%m%d%H%M%S)-${active[0]}"
  jobs=$OUT/multi/$tag.jobs.jsonl; res=$OUT/multi/$tag.result.json
  "$PY" - "$jobs" "${pairs[@]}" <<'PY' || { for part in "${active[@]}"; do report 1 "$part"; done; return 0; }
import json, os, sys
model = {"qwen38": (os.environ["QWEN"], "qwen3.8-27b"), "exaone4": (os.environ["EXAONE"], "exaone4-32b")}
a_kind, c_kind, extra = os.environ["A_KIND"], os.environ["C_KIND"], os.environ["EXTRA_CANDIDATES"]
bs_a, bs_bc = os.environ["BS_A"], os.environ["BS_BC"]
parts = [x.split("=", 1) for x in sys.argv[2:]]
def job(part, wd, cmd, kind, out, bs):
    path, name = model[kind]
    argv = [cmd, "--words", f"{wd}/words.jsonl", "--model", path, "--kind", kind, "--judge-name", name, "--out", f"{wd}/{out}",
            "--gpu", "0", "--batch-size", bs]
    if cmd != "stageA":
        argv += ["--stageA", f"{wd}/A.jsonl", "--extra-candidates", extra]
    return dict(tag=part, argv=argv)
later = [("stageB", "exaone4", "B.exaone4-32b.jsonl"), ("stageB", "qwen38", "B.qwen3.8-27b.jsonl"), ("stageC", c_kind, "C.jsonl")]
later.sort(key=lambda x: x[1] != a_kind)            # the A judge's other stages first (still resident), then the other judge
J = [job(p, wd, "stageA", a_kind, "A.jsonl", bs_a) for p, wd in parts]
J += [job(p, wd, cmd, kind, out, bs_bc) for cmd, kind, out in later for p, wd in parts]
open(sys.argv[1], "w").write("".join(json.dumps(j) + "\n" for j in J))
PY
  echo "GROUP_START rank=$rank parts=${active[*]} jobs=$jobs"
  $T multi --jobs "$jobs" --result "$res" < /dev/null
  failed=$("$PY" -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["failed_tags"]))' "$res" 2>/dev/null) || failed="${active[*]}"
  for part in "${pairs[@]}"; do
    wd=${part#*=}; part=${part%%=*}
    if [[ " $failed " == *" $part "* ]]; then report 1 "$part"
    elif finish_part "$part" "$wd"; then report 0 "$part"; else report 1 "$part"; fi
  done
}

done_n=0; failed_n=0; declare -A SOURCE; group=()
flush_group() {   # heartbeat on every claimed lock while the group runs
  (( ${#group[@]} )) || return 0
  local p; ( while sleep 300; do for p in "${group[@]}"; do touch "$OUT/parts/$p/lock-$owner"; done; done ) & local hb=$!
  run_group "${group[@]}"
  kill "$hb" 2>/dev/null; wait "$hb" 2>/dev/null; group=()
}
while IFS=$'\t' read -r part source npass; do
  [[ -n $part ]] || continue
  [[ -f $OUT/parts/$part/DONE.json ]] && continue
  mkdir -p "$OUT/parts/$part"
  claim_part "$OUT/parts/$part" || { echo "PART_BUSY rank=$rank part=$part"; continue; }
  [[ -f $OUT/parts/$part/DONE.json ]] && continue                 # finished while we were claiming
  SOURCE[$part]=$source; group+=("$part")
  (( ${#group[@]} >= GROUP_PARTS )) && flush_group
  if [[ "${MAX_PARTS_PER_RANK:-0}" -gt 0 && "$done_n" -ge "$MAX_PARTS_PER_RANK" ]]; then break; fi
done < <( { if [[ ${ORDER:-forward} == reverse ]]; then tac "$PARTS_ORDER"; else cat "$PARTS_ORDER"; fi; } |
          awk -F '\t' -v r="$rank" -v n="$ntasks" '((NR-1)%n)==r {print $0}')
flush_group
echo "RANK_COMPLETE rank=$rank done=$done_n failed=$failed_n"
