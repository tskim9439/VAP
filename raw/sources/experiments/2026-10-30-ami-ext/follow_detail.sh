# 활동 추종 세부 진단(2026-10-30, 사용자 "LLM 활동 추종 학습 구현해놓자" — 학습 방식을 고르기 전에 LLM 이 활동을 어긴 단어가 말 차례의 어디에서 나는지 나눈다): 서버 호스트 CPU, 디코드 없음.
#   bash follow_detail.sh ami|amifull <원 결과 폴더 이름(act_runs 가 있는 -follow 등, VAPKT-data/results 아래)>
#   cst_stage3_eval run --from-predictions <원> --follow-detail → <원>-fdetail(행 follow_detail) + follow_detail_report.py 보고. 원 폴더는 읽기만 한다. 아무것도 지우지 않는다.
set -u
K=${1:?ami|amifull}; NAME=${2:?결과 폴더 이름}
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results; AB=/soundai/users/tskim/VAPKT-data/data/ami-bench-v1
X=raw/sources/experiments/2026-10-30-ami-ext
S=$R/$NAME; E=$R/$NAME-fdetail
[ -f $S/summary.json ] || { echo "원 결과가 끝나지 않았다: $S"; exit 1; }
CK=$($P -I -c "import json, sys; print(json.load(open(sys.argv[1]))['checkpoint'])" $S/config-rank0.json)
DL=$($P -I -c "import json, sys; print(','.join(str(x) for x in json.load(open(sys.argv[1]))['deltas']))" $S/config-rank0.json)
case $K in
  ami) WN=(--window 20,90 --window-draws 1 --batch-size 16) ;;
  amifull) C=$($P -I -c "import json, sys; print(json.load(open(sys.argv[1])).get('context_s', 30))" $S/config-rank0.json); WN=(--window full --context-s $C) ;;
  *) echo "ami|amifull: $K"; exit 2 ;;
esac
mkdir -p $E; echo "[$(date '+%F %T')] 활동 추종 세부 $K $S → $E(δ $DL)"
$P experiments/cst_stage3_eval.py run --checkpoint $CK --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 --deltas $DL --backchannel 1.0,3 --max-speakers 4 --declare 4 \
  "${WN[@]}" --from-predictions $S --follow-detail --out $E >> $E/run-rank0.log 2>&1
grep -q "^DONE rank=0 " $E/run-rank0.log || { echo "세부 진단 실패 — $E/run-rank0.log"; tail -5 $E/run-rank0.log; exit 1; }
$P $X/follow_detail_report.py $E > $E/follow-detail-report.txt 2>&1; cat $E/follow-detail-report.txt
echo "[$(date '+%F %T')] FDETAIL_DONE $E"
