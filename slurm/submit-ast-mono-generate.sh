#!/usr/bin/env bash
# 단조 AST 번역 전체 생성(4 방향, 단어마다 stable prefix) 제출 — 스케줄러 셸에서 `bash slurm/submit-ast-mono-generate.sh [옵션]`.
#   옵션: --nodes N(기본 2) --partition P(기본 apex) --job-name NAME(기본 SA_SFT_FullDuplex) --time HH:MM:SS(기본: 파티션 기본값)
#         --root DIR(기본 /soundai/users/tskim/VAPKT-data/data/ast-mono-v1) --after JOBID
#         --prompt v2|v3(기본 v3) --quant bf16|fp8(기본 bf16) --mem-util F(기본 0.95) --max-seqs N(기본 1024)
#         결과 폴더는 ROOT/gen-<prompt>-<quant> (설정마다 따로 — 다른 설정의 결과와 섞이지 않는다)
# 기본값(v3·bf16·0.95·1024)은 2026-10-09 시험에서 고른 것: v2 bf16 대비 en-de 샤드 2.98 배 빠르고 관문 통과 +2 %p
# (fp8 는 3.57 배지만 관문 통과가 1.3 %p 낮다 — wiki/outputs/output-ast-mono-generation-20261009.md).
# 먼저 prepare(CPU)가 끝나 ROOT/units/manifest.json 이 있어야 한다. 노드마다 GPU 8 장에 vLLM 엔진을 하나씩 띄운다.
# 끝나지 않은 작업이 남으면 같은 명령을 다시 제출하면 이어서 한다(완료 항목은 건너뜀). 같은 결과 폴더의 job 둘을 동시에 돌리지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
nodes=2; partition=apex; time_limit=""; job_name=SA_SFT_FullDuplex; after=""
root=/soundai/users/tskim/VAPKT-data/data/ast-mono-v1
prompt=v3; quant=bf16; mem_util=0.95; max_seqs=1024
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) nodes=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;;
    --time) time_limit=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;
    --root) root=$2; shift 2 ;;
    --after) after=$2; shift 2 ;;
    --prompt) prompt=$2; shift 2 ;;
    --quant) quant=$2; shift 2 ;;
    --mem-util) mem_util=$2; shift 2 ;;
    --max-seqs) max_seqs=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo "--nodes 는 양의 정수" >&2; exit 2; }
[[ $prompt == v2 || $prompt == v3 ]] || { echo "--prompt 는 v2 또는 v3" >&2; exit 2; }
[[ $quant == fp8 || $quant == bf16 ]] || { echo "--quant 는 fp8 또는 bf16" >&2; exit 2; }
[[ -f "$root/units/manifest.json" ]] || { echo "없음: $root/units/manifest.json (ast_mono_generate.py prepare 먼저)" >&2; exit 1; }
[[ -f /soundai/users/tskim/VAPKT-data/conda/envs/vllm-0.30.tar ]] || { echo "없음: vLLM env tar (scripts/setup-vllm-env.sh pack)" >&2; exit 1; }
gen_dir="gen-${prompt}-${quant}"; q=""; [[ $quant == fp8 ]] && q=fp8
done_n=0
if [[ -d "$root/$gen_dir" ]]; then done_n=$(find "$root/$gen_dir" -name 'u-*.jsonl.gz' | wc -l); fi
echo "제출: ${gen_dir} 완료 항목 ${done_n}/3474 · 노드 ${nodes} · ${partition} · ${job_name}" >&2
sbatch --job-name="$job_name" --partition="$partition" ${after:+--dependency=afterany:$after} ${time_limit:+--time="$time_limit"} \
  --nodes="$nodes" --ntasks=$((nodes * 8)) --ntasks-per-node=8 --gres=gpu:8 \
  --export="ALL,ROOT=$root,PROMPT=$prompt,QUANT=$q,GEN_DIR=$gen_dir,MEM_UTIL=$mem_util,MAX_SEQS=$max_seqs" slurm/ast-mono-generate.sbatch
