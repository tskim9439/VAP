#!/usr/bin/env bash
# Common Voice 4 영어(CoVoST 2 en_de train) GPU 단계 제출 — 스케줄러 셸에서 `bash slurm/submit-cv-en-align.sh [옵션]`.
#   옵션: --nodes N(기본 1) --partition P(기본 hpc) --time HH:MM:SS(기본 12:00:00) --job-name NAME(기본 SA_EVAL_Fullduplex)
#         --out DIR(기본 /soundai/users/tskim/VAPKT-data/data/cv4-en-v1) --after JOBID
# 끝나지 않은 shard 가 남으면 같은 명령을 다시 제출하면 이어서 한다(완료 shard 는 건너뜀). 같은 OUT 의 job 둘을 동시에 돌리지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
nodes=1; partition=hpc; time_limit=12:00:00; job_name=SA_EVAL_Fullduplex; after=""
out=/soundai/users/tskim/VAPKT-data/data/cv4-en-v1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) nodes=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;;
    --time) time_limit=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --after) after=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo "--nodes 는 양의 정수" >&2; exit 2; }
[[ -f "$out/prefilter/prefilter.parquet" ]] || { echo "없음: $out/prefilter/prefilter.parquet (cv_en_align.py prefilter 먼저)" >&2; exit 1; }
done_n=0
[[ -d "$out/results" ]] && done_n=$(ls "$out/results" | grep -c -- '-b[0-9]' || true)
echo "제출: 완료 shard ${done_n} · 노드 ${nodes} · ${partition} · OUT=${out}" >&2
sbatch --job-name="$job_name" --partition="$partition" ${after:+--dependency=afterany:$after} \
  --nodes="$nodes" --ntasks=$((nodes * 8)) --ntasks-per-node=8 --gres=gpu:8 --time="$time_limit" \
  --export="ALL,OUT=$out" slurm/cv-en-align.sbatch
