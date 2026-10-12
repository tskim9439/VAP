#!/usr/bin/env bash
# q06-s3-ast/final 의 한 스트림 실시간성 측정 제출(2026-10-12) — 스케줄러 셸에서 `bash slurm/submit-stream-rtf.sh [--partition P]`.
# 결과: results/stream-rtf-q06-s3-ast-node/{rows,summary}-{float32,graph-stream-float32,graph-stream-bfloat16}.json(l). 같은 폴더에 다시 내면 덮어쓴다.
set -euo pipefail
cd "$(dirname "$0")/.."
partition=hpc
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) partition=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
CK=/soundai/Model/VAPASR/semcommit-q06-s3-ast/final
OUT=/soundai/users/tskim/VAPKT-data/results/stream-rtf-q06-s3-ast-node
[[ -f $CK/config.json ]] || { echo "체크포인트 없음: $CK" >&2; exit 1; }
echo "제출: $CK → $OUT · $partition" >&2
sbatch --partition="$partition" --export="ALL,CK=$CK,OUT=$OUT" slurm/stream-rtf.sbatch
