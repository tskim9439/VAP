#!/usr/bin/env bash
# 단일 화자 AST·ASR 전체 평가 제출(2026-10-23) — slurm/submit-eval-q06-s4-ast.sh 를 체크포인트·태그 인자로 일반화했다(q06-s3·s4-ast 평가와 같은 7 셋·설정).
#   bash slurm/submit-eval-ast.sh --ck <체크포인트 폴더> --tag <이름> [--afterok JOBID] [--partition apex] [--time 12:00:00] [--dry-run]
#   셋(몫): FLEURS 최대 30 s(8) · CoVoST 2 고정 부분집합 sub7(8) · FLEURS 영어 작은 소리 peak05(4) · vpaa-kspon(8) · yodas-de dev(8) · LibriSpeech clean·other(4·4)
#   설정은 slurm/eval-single-turn-queue.sbatch 그대로(δ4·6·8·10 + final, batch 64, 꼬리 무음 1 s, --verify, fp32). 산출 results/<셋>-<태그>/,
#   목록 results/evalq-<태그>/tasks.txt(없을 때만 만든다 — 다시 내면 끝난 몫은 이어서/건너뛴다). --afterok 면 학습 job 이 성공한 뒤에만 시작하고
#   체크포인트 검사는 job 시작 때 sbatch 의 test -f 가 한다. 노드 제외는 1.7B 체인과 같다(hpc-52·4 + 50·51·101·102). 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
ck=""; tag=""; afterok=""; partition=apex; time_limit=12:00:00; dry=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ck) ck=$2; shift 2 ;;  --tag) tag=$2; shift 2 ;;  --afterok) afterok=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;;  --time) time_limit=$2; shift 2 ;;  --dry-run) dry=1; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ -n $ck && $tag =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--ck 와 --tag(영숫자·._-)가 필요하다" >&2; exit 2; }
[[ -z $afterok || $afterok =~ ^[0-9]+$ ]] || { echo "--afterok 는 job id" >&2; exit 2; }
[[ -n $afterok || -f $ck/config.json ]] || { echo "체크포인트 없음: $ck (학습 job 뒤에 걸려면 --afterok JOBID)" >&2; exit 1; }
D=/soundai/users/tskim/VAPKT-data; EV=$D/data/evaluation; A=$EV/ast-fleurs-covost2-v1; RES=$D/results; Q=$RES/evalq-$tag; T=$Q/tasks.txt
list() {
  while read -r man out world; do
    for ((r = 0; r < world; r++)); do echo "$man $RES/$out-$tag $world $r"; done
  done <<LIST
$A/manifest-max30s-fleurs.jsonl ast-fleurs 8
$A/manifest-max30s-covost2-sub7.jsonl ast-covost2-sub7 8
$A/manifest-max30s-fleurs-en-peak05.jsonl ast-fleurs-en-peak05 4
$EV/single-turn-vpaa-kspon-v1/manifest.jsonl single-turn-vpaa-kspon 8
$EV/single-turn-yodas-de-dev-v1/manifest.jsonl single-turn-yodas-de 8
$EV/single-turn-librispeech-v1/manifest-test-clean.jsonl single-turn-librispeech-clean 4
$EV/single-turn-librispeech-v1/manifest-test-other.jsonl single-turn-librispeech-other 4
LIST
}
exc=--exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4,slurmmxch200v5-hpc-50,slurmmxch200v5-hpc-51,slurmmxch200v5-hpc-101,slurmmxch200v5-hpc-102
cmd="sbatch --partition=$partition --time=$time_limit ${afterok:+--dependency=afterok:$afterok }$exc --export=ALL,CK=$ck,TASKS=$T slurm/eval-single-turn-queue.sbatch"
if (( dry )); then list; echo "$cmd"; echo "Submitted batch job 0"; exit 0; fi
for m in $(list | awk '{print $1}' | sort -u); do [[ -f $m ]] || { echo "manifest 없음: $m" >&2; exit 1; }; done
mkdir -p "$Q"; [[ -f $T ]] || list > "$T"
echo "제출: $ck · 작업 $(wc -l < "$T") 개 · $partition${afterok:+ · job $afterok 성공 뒤}" >&2
$cmd
