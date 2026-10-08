#!/usr/bin/env bash
# YODAS-Granary German GPU stage 제출 — 스케줄러 셸에서 `bash slurm/submit-yodas-de-align.sh [옵션]`.
#   옵션: --nodes N(기본 5) --partition P(기본 hpc) --time HH:MM:SS(기본 24:00:00) --job-name NAME(기본 SA_EVAL_Fullduplex)
#         --out DIR(기본 /soundai/users/tskim/VAPKT-data/data/yodas-de-v1) --after JOBID
#         --long  30–40 s 발화(사전 점검에서 dur_long 만 걸린 행) 처리 — shard 폴더 <shard>-dur_long, 본 처리와 동시에 돌려도 된다
# 끝나지 않은 shard 가 남으면 같은 명령을 다시 제출하면 이어서 한다(완료 shard 는 건너뜀). 같은 처리(본/--long)의 job 둘을 동시에 돌리지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
nodes=5; partition=hpc; time_limit=24:00:00; job_name=SA_EVAL_Fullduplex; after=""; long=0
out=/soundai/users/tskim/VAPKT-data/data/yodas-de-v1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) nodes=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;;
    --time) time_limit=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --after) after=$2; shift 2 ;;
    --long) long=1; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo "--nodes 는 양의 정수" >&2; exit 2; }
[[ -f "$out/prefilter/prefilter.parquet" ]] || { echo "없음: $out/prefilter/prefilter.parquet (yodas_de_prepare.py prefilter 먼저)" >&2; exit 1; }
if (( long )); then pass_env=",SELECT_FLAGS=dur_long,MAX_DUR=40"; pass_name="long(30–40 s)"; else pass_env=""; pass_name="본(1–30 s)"; fi
done_n=0  # results/ 는 첫 실행 전에는 없다(set -e·pipefail 에서 ls 실패로 조용히 끝나지 않게)
if [[ -d "$out/results" ]]; then
  if (( long )); then done_n=$(ls "$out/results" | grep -c -- '-dur_long$' || true); else done_n=$(ls "$out/results" | grep -vc -- '-dur_long$' || true); fi
fi
echo "제출: ${pass_name} 처리 · 완료 shard ${done_n} · 노드 ${nodes} · ${partition} · OUT=${out}" >&2
sbatch --job-name="$job_name" --partition="$partition" ${after:+--dependency=afterany:$after} \
  --nodes="$nodes" --ntasks=$((nodes * 8)) --ntasks-per-node=8 --gres=gpu:8 --time="$time_limit" \
  --export="ALL,OUT=$out${pass_env}" slurm/yodas-de-align.sbatch
