# 출력 쪽 재귀속 재채점(2026-10-30, 사용자 "출력 쪽 재귀속을 먼저 실험하고 그 다음에 학습으로 고치는 것이 시도하는 게 나을거 같아"): 서버 호스트 CPU, 디코드 없음.
#   bash relabel_rescore.sh ami|amifull <원 결과 폴더 이름(act_runs 가 있는 -follow 등, VAPKT-data/results 아래)> [BACK,FWD(기본 3,1)]
#   cst_stage3_eval run --from-predictions <원> --relabel-follow --follow-window W --en-norm whisper → <원>-relabel<B>-<F>-wn(단어마다 Whisper 정규화 점수·ami_report)
#   + meeteval cpWER(whisper — 외부 기준과 같은 도구·정규화, ours — 우리 정규화 대조). 원 폴더는 읽기만 한다. 아무것도 지우지 않는다.
set -u
K=${1:?ami|amifull}; NAME=${2:?결과 폴더 이름}; W=${3:-3,1}
[[ $W =~ ^[0-9]+,[0-9]+$ ]] || { echo "BACK,FWD: $W"; exit 2; }
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results; AB=/soundai/users/tskim/VAPKT-data/data/ami-bench-v1
S=$R/$NAME; E=$R/$NAME-relabel${W/,/-}-wn
[ -f $S/summary.json ] || { echo "원 결과가 끝나지 않았다: $S"; exit 1; }
CK=$($P -I -c "import json, sys; print(json.load(open(sys.argv[1]))['checkpoint'])" $S/config-rank0.json)
DL=$($P -I -c "import json, sys; print(','.join(str(x) for x in json.load(open(sys.argv[1]))['deltas']))" $S/config-rank0.json)
case $K in
  ami) WN=(--window 20,90 --window-draws 1 --batch-size 16) ;;
  amifull) C=$($P -I -c "import json, sys; print(json.load(open(sys.argv[1])).get('context_s', 30))" $S/config-rank0.json); WN=(--window full --context-s $C) ;;
  *) echo "ami|amifull: $K"; exit 2 ;;
esac
mkdir -p $E; echo "[$(date '+%F %T')] 재귀속 재채점 $K $S → $E(창 $W, δ $DL)"
$P experiments/cst_stage3_eval.py run --checkpoint $CK --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 --deltas $DL --backchannel 1.0,3 --max-speakers 4 --declare 4 \
  "${WN[@]}" --from-predictions $S --relabel-follow --follow-window $W --en-norm whisper --out $E >> $E/run-rank0.log 2>&1
grep -q "^DONE rank=0 " $E/run-rank0.log || { echo "재채점 실패 — $E/run-rank0.log"; tail -5 $E/run-rank0.log; exit 1; }
$P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
$P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $E > $E/ami-report.txt 2>&1
for n in whisper ours; do R_=(); [ $n = ours ] && R_=(--refs ""); $P experiments/meeteval_cpwer.py --pred $E --norm $n "${R_[@]}" --json $E/meeteval-$n.json > $E/meeteval-$n.txt 2>&1; done
echo "== meeteval cpWER(화자 통째)"; grep -E "meeteval cpWER  " $E/meeteval-whisper.txt $E/meeteval-ours.txt
echo "[$(date '+%F %T')] RELABEL_DONE $E"
