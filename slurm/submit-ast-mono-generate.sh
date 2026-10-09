#!/usr/bin/env bash
# 단조 AST 번역 전체 생성(4 방향, 단어마다 stable prefix) 제출 — 스케줄러 셸에서 `bash slurm/submit-ast-mono-generate.sh [옵션]`.
#   옵션: --nodes N(기본 4) --partition P(기본 hpc) --time HH:MM:SS(기본 24:00:00) --job-name NAME(기본 SA_EVAL_Fullduplex)
#         --root DIR(기본 /soundai/users/tskim/VAPKT-data/data/ast-mono-v1) --after JOBID
# 먼저 prepare(CPU)가 끝나 ROOT/units/manifest.json 이 있어야 한다. 노드마다 GPU 8 장에 vLLM 엔진을 하나씩 띄운다.
# 끝나지 않은 작업이 남으면 같은 명령을 다시 제출하면 이어서 한다(완료 항목은 건너뜀). 같은 ROOT 의 job 둘을 동시에 돌리지 않는다
# (노드 수가 다르면 작업 배분이 달라 같은 항목을 두 번 처리할 수 있다 — 결과는 같은 파일로 바뀔 뿐이지만 GPU 낭비).
set -euo pipefail
cd "$(dirname "$0")/.."
nodes=4; partition=hpc; time_limit=24:00:00; job_name=SA_EVAL_Fullduplex; after=""
root=/soundai/users/tskim/VAPKT-data/data/ast-mono-v1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) nodes=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;;
    --time) time_limit=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;
    --root) root=$2; shift 2 ;;
    --after) after=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo "--nodes 는 양의 정수" >&2; exit 2; }
[[ -f "$root/units/manifest.json" ]] || { echo "없음: $root/units/manifest.json (ast_mono_generate.py prepare 먼저)" >&2; exit 1; }
[[ -f /soundai/users/tskim/VAPKT-data/conda/envs/vllm-0.30.tar ]] || { echo "없음: vLLM env tar (scripts/setup-vllm-env.sh pack)" >&2; exit 1; }
done_n=0
if [[ -d "$root/gen" ]]; then done_n=$(find "$root/gen" -name 'u-*.jsonl.gz' | wc -l); fi
echo "제출: 완료 항목 ${done_n} · 노드 ${nodes} · ${partition} · ROOT=${root}" >&2
sbatch --job-name="$job_name" --partition="$partition" ${after:+--dependency=afterany:$after} \
  --nodes="$nodes" --ntasks=$((nodes * 8)) --ntasks-per-node=8 --gres=gpu:8 --time="$time_limit" \
  --export="ALL,ROOT=$root" slurm/ast-mono-generate.sbatch
