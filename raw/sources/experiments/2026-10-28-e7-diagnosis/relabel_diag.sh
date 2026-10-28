# E7 회의 통째 귀속 사후 진단(mxc 호스트 CPU, 2026-10-28 — 사용자 "1. 사후 진단 … 진행해보자"). 디코드 없이 원출력만 다시 매긴다(표준 정규화 — Whisper).
#   회의 통째 결과 폴더(cst_stage3_eval --window full)의 가설 단어를 슬롯과 상관없이 활동으로 다시 귀속해 cp 를 잰다(cst_stage3_eval --relabel-act):
#     -relabel-e7a    = E7a 헤드의 회의 단위 AOSC 확률(diar_eval run --session --save-probs → <probs>/<cid>@full.npz 의 prob)
#     -relabel-oracle = 같은 npz 의 정답 활동(ref — 회의 전체 도착 순서)
#   비교 기준 = 같은 폴더의 표준 재채점(-wn, rescore_std.sh). LLM 태그가 헤드 슬롯을 따랐다면 얻었을 귀속(e7a)과 활동이 완벽할 때(oracle)를 어림한다.
#   bash relabel_diag.sh <회의 통째 결과 폴더 이름> <probs 폴더> [CTX(기본 30)]
#     예: bash relabel_diag.sh stage3-eval-amifull-c30-aosc-e7a-q17-e6-final /soundai/users/tskim/VAPKT-data/results/diar-session-e7a-probs/aosc/probs
#   원 폴더는 읽기만 한다. 아무것도 지우지 않는다.
set -u
NAME=${1:?결과 폴더 이름}; PR=${2:?probs 폴더}; C=${3:-30}
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results; AB=/soundai/users/tskim/VAPKT-data/data/ami-bench-v1
S=$R/$NAME; [ -f $S/summary.json ] || { echo "원 결과가 끝나지 않았다: $S"; exit 1; }
CK=$($P -I -c "import json, sys; print(json.load(open(sys.argv[1]))['checkpoint'])" $S/config-rank0.json)
for v in e7a oracle; do
  E=$R/$NAME-relabel-$v; O=(); [ $v = oracle ] && O=(--relabel-oracle); mkdir -p $E
  $P experiments/cst_stage3_eval.py run --checkpoint $CK --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 --deltas 4,10 --backchannel 1.0,3 --max-speakers 4 --declare 4 \
    --window full --context-s $C --from-predictions $S --relabel-act $PR "${O[@]}" --en-norm whisper --out $E >> $E/run-rank0.log 2>&1 &
done
wait
for v in e7a oracle; do
  E=$R/$NAME-relabel-$v
  grep -q "^DONE rank=0 " $E/run-rank0.log || { echo "$v 실패 — $E/run-rank0.log"; tail -5 $E/run-rank0.log; continue; }
  $P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
  $P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $E > $E/ami-report.txt 2>&1
done
echo "== 표준 재채점(태그 그대로)"; grep -E "^\s+ami" $S-wn/ami-report.txt 2>/dev/null || echo "(없음 — rescore_std.sh amifull $NAME 먼저)"
for v in e7a oracle; do echo "== 다시 귀속: $v"; grep -E "^\s+ami" $R/$NAME-relabel-$v/ami-report.txt; done
echo "[$(date "+%F %T")] RELABEL_DONE $NAME"
