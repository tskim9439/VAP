#!/usr/bin/env bash
# q06-s3-ast/final 전체 평가 제출(2026-10-12) — 스케줄러 셸에서 `bash slurm/submit-eval-q06-s3-ast.sh [--partition P] [--after JOBID]`.
# s2(wiki output-q06-s2-ast-eval-20261011)와 같은 평가셋·설정. CoVoST 2 는 베이스라인 비교와 같은 고정 부분집합(7 행마다 1 행, 4,148 행).
# 작업 목록은 results/evalq-q06-s3-ast/tasks.txt(없을 때만 만든다). 다시 제출하면 끝난 rank 는 이어서/건너뛴다.
set -euo pipefail
cd "$(dirname "$0")/.."
partition=hpc; after=""; time_limit=12:00:00
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) partition=$2; shift 2 ;;
    --after) after=$2; shift 2 ;;
    --time) time_limit=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
CK=/soundai/Model/VAPASR/semcommit-q06-s3-ast/final
D=/soundai/users/tskim/VAPKT-data; EV=$D/data/evaluation; A=$EV/ast-fleurs-covost2-v1; RES=$D/results
Q=$RES/evalq-q06-s3-ast; mkdir -p "$Q"; T=$Q/tasks.txt
if [[ ! -f $T ]]; then
  while read -r man out world; do
    for ((r = 0; r < world; r++)); do echo "$man $RES/$out $world $r"; done
  done > "$T" <<LIST
$A/manifest-max30s-fleurs.jsonl ast-fleurs-q06-s3-ast 8
$A/manifest-max30s-covost2-sub7.jsonl ast-covost2-sub7-q06-s3-ast 8
$A/manifest-max30s-fleurs-en-peak05.jsonl ast-fleurs-en-peak05-q06-s3-ast 4
$EV/single-turn-vpaa-kspon-v1/manifest.jsonl single-turn-vpaa-kspon-q06-s3-ast 8
$EV/single-turn-yodas-de-dev-v1/manifest.jsonl single-turn-yodas-de-q06-s3-ast 8
$EV/single-turn-librispeech-v1/manifest-test-clean.jsonl single-turn-librispeech-clean-q06-s3-ast 4
$EV/single-turn-librispeech-v1/manifest-test-other.jsonl single-turn-librispeech-other-q06-s3-ast 4
LIST
fi
[[ -f $CK/config.json ]] || { echo "체크포인트 없음: $CK" >&2; exit 1; }
for m in $(awk '{print $1}' "$T" | sort -u); do [[ -f $m ]] || { echo "manifest 없음: $m" >&2; exit 1; }; done
echo "제출: $CK · 작업 $(wc -l < "$T") 개 · $partition" >&2
sbatch --partition="$partition" --time="$time_limit" ${after:+--dependency=afterany:$after} \
  --export="ALL,CK=$CK,TASKS=$T" slurm/eval-single-turn-queue.sbatch
