#!/usr/bin/env bash
# semcommit v0.3.5 전량 라벨링(19 개 DB, 파트 단위) 제출 — 스케줄러 셸에서 `bash slurm/submit-semcommit-label-v035-apex-n2.sh`.
# 기본 2 노드 × 8 H200, job 이름 SA_SFT_FullDuplex, 24 시간. 끝나지 않은 파트가 남으면 같은 명령을 다시 제출하면 이어서 한다
# (DONE.json 파트는 건너뜀). 파트는 작업 번호로 고정 배분되므로 두 job 을 동시에 돌리지 않는다 — 이어 붙일 때는 --after.
#   옵션: --nodes N(기본 2) --time HH:MM:SS(기본 24:00:00) --after JOBID(그 job 이 끝난 뒤 시작, afterany)
#   예: 6 노드 3 시간 → 끝나면 2 노드로 이어서
#     bash slurm/submit-semcommit-label-v035-apex-n2.sh --nodes 6 --time 03:00:00
#     bash slurm/submit-semcommit-label-v035-apex-n2.sh --after <위 job id>
# 입력 경로는 이 스크립트의 지역 변수이고 sbatch --export 로 job 에만 전달된다(제출 셸 환경에는 남지 않는다).
# 중간 학습: 끝난 파트만 모아 목록을 만든다 —
#   python experiments/semcommit_collect_done.py --labels-root <OUT> --order <QC_SPLIT>/parts-order.tsv --out-dir <OUT>/snapshots/<날짜>
#   → semcommit_train.py --train-words @main-words.list --train-labels @main-labels.list \
#                        --short-train-words @short-words.list --short-train-labels @short-labels.list
set -euo pipefail
cd "$(dirname "$0")/.."

nodes=2; time_limit=24:00:00; after=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) nodes=$2; shift 2 ;;
    --time) time_limit=$2; shift 2 ;;
    --after) after=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1 (--nodes N --time HH:MM:SS --after JOBID)" >&2; exit 2 ;;
  esac
done
[[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo "--nodes 는 양의 정수" >&2; exit 2; }
[[ -z $after || $after =~ ^[0-9]+$ ]] || { echo "--after 는 job id" >&2; exit 2; }

D=/soundai/users/tskim/VAPKT-data/data
W=$D/semcommit-work
G=$W/gate/speechlm-all19-g1-punctcoord-v035-20260926

results=$D/speechlm-asr-align-v1-full-b128-20260923/results
qc_split=$W/qc-pass-split-v035-20260926
approval_summary=$G/decision/training-eligibility-summary.json
gate_accepted=$G/decision/gate-accepted.json
thresholds=$G/thresholds.json
out=$W/labels/speechlm-all19-v035

for f in "$qc_split/parts-order.tsv" "$approval_summary" "$gate_accepted" "$thresholds"; do
  [[ -f "$f" ]] || { echo "없음: $f" >&2; exit 1; }
done
[[ -d "$results" ]] || { echo "없음: $results" >&2; exit 1; }
if [[ -d "$out/parts" ]]; then
  echo "이어서 제출: 끝난 파트 $(find "$out/parts" -maxdepth 2 -name DONE.json | wc -l) / $(wc -l < "$qc_split/parts-order.tsv")" >&2
fi

sbatch --job-name=SA_SFT_FullDuplex --partition=apex ${after:+--dependency=afterany:$after} \
  --nodes="$nodes" --ntasks=$((nodes * 8)) --ntasks-per-node=8 --gres=gpu:8 --time="$time_limit" \
  --export="ALL,RESULTS=$results,QC_SPLIT=$qc_split,APPROVAL_SUMMARY=$approval_summary,GATE_ACCEPTED=$gate_accepted,THRESHOLDS=$thresholds,A_KIND=qwen38,C_KIND=qwen38,OUT=$out" \
  slurm/semcommit-part-label-apex.sbatch
