#!/usr/bin/env bash
# speechlm 파트 words.jsonl 만 만들기(라벨링 없음, GPU 불필요) — semcommit-part-label-worker.sh 의 prepare_part 와 같은 두 단계
# (후보 semcommit_build_speechlm_candidates.py → QC 통과 승인 semcommit_approve_speechlm_words.py), 같은 출력 위치(<OUT>/parts/<part>/words.jsonl(.ok)).
# 나중에 라벨링 워커가 그 파트를 잡으면 words.jsonl.ok 를 보고 이 단계를 건너뛴다. 만든 words 는 라벨 전에도 SEM 중립 ASR 데이터로 쓸 수 있다
# (experiments/semcommit_asr_words_list.py → semcommit_train --asr-words).
# 파트 잠금은 라벨링 워커와 같다(<part>/lock-<owner>, LOCK_STALE_S 안의 남의 잠금이면 건너뜀, 잠금 파일은 지우지 않는다). 아무것도 지우지 않는다.
# 실행: SLURM(여러 task: SLURM_PROCID/SLURM_NTASKS) 또는 직접(WORKER_ID/NWORKERS/RUN_ID). CPU 작업이라 GPU 를 잡지 않는다.
#   MAIN_ONLY=1(기본): 마지막 라벨링(76481)과 같은 범위 — 새 한국어 파트는 8 s 이상 행만(승인 --main-only). 0 이면 short 행도.
set -uo pipefail
PROJECT=${PROJECT:-/soundai/users/tskim/VAPKT}
PY=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/semcommit-teacher/bin/python}
QWEN_ASR=${QWEN_ASR:-/soundai/Model/Qwen3-ASR-0.6B}
D=/soundai/users/tskim/VAPKT-data/data; W=$D/semcommit-work; G=$W/gate/speechlm-all19-g1-punctcoord-v035-20260926
RESULTS=${RESULTS:-$D/speechlm-asr-align-v1-full-b128-20260923/results}
QC_SPLIT=${QC_SPLIT:-$W/qc-pass-split-v035-20260926}
APPROVAL_SUMMARY=${APPROVAL_SUMMARY:-$G/decision/training-eligibility-summary.json}
OUT=${OUT:-$W/labels/speechlm-all19-v035}
PARTS_ORDER=${PARTS_ORDER:-$OUT/orders/mixed-20260928b.tsv}
MAIN_ONLY=${MAIN_ONLY:-1}; [[ $MAIN_ONLY == 0 ]] && MAIN_ONLY=""
LOCK_STALE_S=${LOCK_STALE_S:-1800}
cd "$PROJECT"
test -x "$PY" && test -d "$RESULTS" && test -f "$PARTS_ORDER" && test -f "$APPROVAL_SUMMARY" && test -d "$QWEN_ASR" || { echo "입력 경로 확인 실패" >&2; exit 2; }
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=

rank=${SLURM_PROCID:-${WORKER_ID:?SLURM_PROCID or WORKER_ID}}; ntasks=${SLURM_NTASKS:-${NWORKERS:?SLURM_NTASKS or NWORKERS}}
run_id=${SLURM_JOB_ID:-${RUN_ID:?SLURM_JOB_ID or RUN_ID}}
attempt="w${run_id}-r${SLURM_RESTART_COUNT:-0}-t$rank"
owner="${run_id}-r${SLURM_RESTART_COUNT:-0}-t${rank}-$(hostname -s)-$$"

fresh_foreign_lock() { local dest=$1 mine=$2 f now; now=$(date +%s)
  for f in "$dest"/lock-*; do [[ -e $f && $f != "$mine" ]] || continue; (( now - $(stat -c %Y "$f") < LOCK_STALE_S )) && { echo "$f"; return 0; }; done; return 1; }
words_ok() { local dest=$1 d; [[ -f $dest/words.jsonl.ok ]] && return 0; for d in "$dest"/attempt-*; do [[ -f $d/words.jsonl.ok ]] && return 0; done; return 1; }

built=0; skipped=0; failed=0; busy=0; i=0
while IFS=$'\t' read -r part source _; do
  [[ -n $part && $part != \#* ]] || continue
  (( i++ % ntasks == rank )) || continue
  [[ $part == en-* ]] && { skipped=$((skipped + 1)); continue; }                                   # 영어 파트는 semcommit_build_en_parts.py 가 미리 만든다
  dest=$OUT/parts/$part
  if words_ok "$dest" || [[ -f $dest/DONE.json ]]; then skipped=$((skipped + 1)); continue; fi
  mkdir -p "$dest"; mine=$dest/lock-$owner
  if fresh_foreign_lock "$dest" "$mine" >/dev/null; then busy=$((busy + 1)); echo "PART_BUSY rank=$rank part=$part"; continue; fi
  touch "$mine"
  if [[ -e $dest/words.jsonl ]]; then wd=$dest/attempt-$attempt; mkdir -p "$wd"; else wd=$dest; fi      # 미완성 words.jsonl 은 그대로 두고 attempt 에
  t0=$(date +%s)
  if "$PY" -u experiments/semcommit_build_speechlm_candidates.py --results "$RESULTS" --out-dir "$dest/cand-$attempt" --part "$part" --tokenizer "$QWEN_ASR" \
     && [[ -f $dest/cand-$attempt/$part.words.jsonl ]] \
     && "$PY" -u experiments/semcommit_approve_speechlm_words.py --training-eligibility "$APPROVAL_SUMMARY" --qc-split "$QC_SPLIT" \
          --part "$part" --candidates "$dest/cand-$attempt/$part.words.jsonl" --out-words "$wd/words.jsonl" ${MAIN_ONLY:+--main-only}; then
    built=$((built + 1)); echo "WORDS_DONE rank=$rank part=$part source=$source sec=$(( $(date +%s) - t0 ))"
  else failed=$((failed + 1)); echo "WORDS_FAILED rank=$rank part=$part source=$source"; fi
  touch "$mine"
done < "$PARTS_ORDER"
echo "[$(date '+%F %T')] words-only rank=$rank built=$built skipped=$skipped busy=$busy failed=$failed"
