#!/usr/bin/env bash
# semcommit v0.3.5 전량 라벨링(19 개 DB, 파트 단위) 제출 — 스케줄러 셸에서 `bash slurm/submit-semcommit-label-v035-apex-n2.sh`.
# 2 노드 × 8 H200, job 이름 SA_SFT_FullDuplex, 24 시간. 끝나지 않은 파트가 남으면 같은 명령을 다시 제출하면 이어서 한다
# (DONE.json 파트는 건너뜀). 두 개를 동시에 돌리지 않는다.
# 입력 경로는 이 스크립트의 지역 변수이고 sbatch --export 로 job 에만 전달된다(제출 셸 환경에는 남지 않는다).
# 중간 학습: 끝난 파트만 모아 목록을 만든다 —
#   python experiments/semcommit_collect_done.py --labels-root <OUT> --order <QC_SPLIT>/parts-order.tsv --out-dir <OUT>/snapshots/<날짜>
#   → semcommit_train.py --train-words @main-words.list --train-labels @main-labels.list \
#                        --short-train-words @short-words.list --short-train-labels @short-labels.list
set -euo pipefail
cd "$(dirname "$0")/.."

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

sbatch --job-name=SA_SFT_FullDuplex --partition=apex \
  --nodes=2 --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --time=24:00:00 \
  --export="ALL,RESULTS=$results,QC_SPLIT=$qc_split,APPROVAL_SUMMARY=$approval_summary,GATE_ACCEPTED=$gate_accepted,THRESHOLDS=$thresholds,A_KIND=qwen38,C_KIND=qwen38,OUT=$out" \
  slurm/semcommit-part-label-apex.sbatch
