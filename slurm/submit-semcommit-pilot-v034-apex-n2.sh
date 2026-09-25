#!/usr/bin/env bash
# semcommit v0.3.4 파일럿 라벨링 제출 — 스케줄러 셸에서 `bash slurm/submit-semcommit-pilot-v034-apex-n2.sh` 한 번.
# 2 노드 × 8 H200(작업 16 개, 작업당 GPU 1 장), job 이름 SA_SFT_FullDuplex.
# 입력 경로는 이 스크립트의 지역 변수이고 sbatch --export 로 job 에만 전달된다(제출 셸의 환경에는 남지 않는다).
# 입력: 관문 통과(v0.3.4, qwen) + 승인 단어 25 shard(AI Hub 000004·000027 각 2 파트, 9,761 행).
set -euo pipefail
cd "$(dirname "$0")/.."

W=/soundai/users/tskim/VAPKT-data/data/semcommit-work
G=$W/gate/speechlm-all19-g1-punctcoord-v034-20260925
P=$W/pilot-v034-20260925

word_index=$P/approved/index.tsv
approval_summary=$G/decision/training-eligibility-summary.json
gate_accepted=$G/decision/gate-accepted.json
thresholds=$G/thresholds.json
out=$P/labels

for f in "$word_index" "$approval_summary" "$gate_accepted" "$thresholds"; do
  [[ -f "$f" ]] || { echo "없음: $f" >&2; exit 1; }
done
[[ -e "$out" ]] && echo "주의: $out 이 이미 있다 — 끝난 행은 건너뛰고 이어서 채운다" >&2

sbatch --job-name=SA_SFT_FullDuplex --partition=apex \
  --nodes=2 --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --time=08:00:00 \
  --export="ALL,WORD_INDEX=$word_index,APPROVAL_SUMMARY=$approval_summary,GATE_ACCEPTED=$gate_accepted,THRESHOLDS=$thresholds,A_KIND=qwen38,C_KIND=qwen38,OUT=$out" \
  slurm/semcommit-main-label-apex-n2.sbatch
