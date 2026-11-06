#!/usr/bin/env bash
# ESB AMI 단일 화자 받아쓰기 평가 제출(2026-11-06, 사용자 "1번 진단부터 진행하자" — wiki output-ssa-papers-review-20261106 5 절).
#   OpenASR 리더보드 AMI 셋(IHM 개별 헤드셋 12,643 발화)이다. 혼자 말 받아쓰기 격차(표준 정규화로 Parakeet + N3D 대비 2.0 / 3.2 %p)가
#   기반 받아쓰기 탓인지 가린다. 논문 보고(arXiv 2609.10265 표 2): Nemotron Speech 스트리밍 11.58 %, SSA 11.62 %(1.04 s).
#   쓰기(스케줄러 셸): bash slurm/submit-eval-esb-ami.sh <e13|e15b-ck4250|s4> [--partition P(기본 hpc)] [--deltas "10"] [--time T] [--after JOBID]
#   실행기 slurm/eval-single-turn-queue.sbatch(노드 하나, GPU 8 장, δ 만 — FINAL=0). 매니페스트 VAPKT-data/data/evaluation/single-turn-esb-ami-v1(prepare 는 따로).
#   산출 results/single-turn-esb-ami-<태그>. Whisper 정규화 WER 은 raw/sources/experiments/2026-11-06-e15b/esb_ami_whisper.py 로 다시 맨다. 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
m=${1:?"모드: e13 | e15b-ck4250 | s4"}; shift
partition=hpc; deltas="10"; time_limit=6:00:00; after=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) partition=$2; shift 2 ;;  --deltas) deltas=$2; shift 2 ;;  --time) time_limit=$2; shift 2 ;;  --after) after=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
M=/soundai/Model/VAPASR
case $m in
  e13) CK=$M/stage3-q17-e13/final ;;                          # 지금 최선(E13 · N3D 의 LLM)
  e15b-ck4250) CK=$M/stage3-q17-e15b/checkpoint-4250 ;;      # E15b 중단 시점(인코더 위층 학습의 단일 화자 영향)
  s4) CK=$M/semcommit-q17-s4-ast/final ;;                     # Stage 2(AST 3 차) — 회의 학습 전 기반
  *) echo "모드는 e13 | e15b-ck4250 | s4: $m" >&2; exit 2 ;;
esac
D=/soundai/users/tskim/VAPKT-data; MAN=$D/data/evaluation/single-turn-esb-ami-v1/manifest.jsonl; RES=$D/results
[[ -f $MAN ]] || { echo "매니페스트 없음: $MAN (eval_single_turn_asr.py prepare --esb-ami-list 먼저)" >&2; exit 1; }
[[ -f $CK/config.json ]] || { echo "체크포인트 없음: $CK" >&2; exit 1; }
Q=$RES/evalq-esb-ami-$m; mkdir -p "$Q"; T=$Q/tasks.txt
[[ -f $T ]] || for r in 0 1 2 3 4 5 6 7; do echo "$MAN $RES/single-turn-esb-ami-$m 8 $r"; done > "$T"
echo "제출: $CK · δ [$deltas] · 작업 $(wc -l < "$T") 개 · $partition · $time_limit" >&2
sbatch --partition="$partition" --time="$time_limit" ${after:+--dependency=afterany:$after} \
  --export="ALL,CK=$CK,TASKS=$T,DELTAS=$deltas,FINAL=0" slurm/eval-single-turn-queue.sbatch
