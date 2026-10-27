# AMI 표준 정규화 재채점(mxc 호스트 CPU, 2026-10-27 — 사용자 "표준 정규화로 다시 채점하는 것도 만들어줘"). 디코드 없이 원출력만 다시 매긴다.
#   ① cst_stage3_eval run --from-predictions <원 폴더> --en-norm whisper → <원 폴더>-wn: 창·회의 점수 전부(cp·agn·귀속·EOT — 단어마다 Whisper EnglishTextNormalizer)
#      + summarize + ami_report(공식 dev·eval 별) → <원 폴더>-wn/ami-report.txt
#   ② experiments/meeteval_cpwer.py --norm whisper(화자 통째 문장 정규화 + meeteval cpWER — 외부 AMI 수치와 같은 도구)·--norm ours(우리 정규화 대조)
#      → <원 폴더>-wn/meeteval-{whisper,ours}.{txt,json}
#   bash rescore_std.sh ami|amifull <원 결과 폴더 이름(VAPKT-data/results 아래)> [CTX(amifull 문맥 초, 기본 30 — 원 평가와 같게)]
#     예: bash rescore_std.sh ami stage3-eval-ami-q17-e6-final
#         bash rescore_std.sh amifull stage3-eval-amifull-c30-aosc-e7a-q17-e6-final
#   원 폴더는 읽기만 한다. <원 폴더>-wn 이 있으면 이어 한다(끝난 행은 건너뛴다). 아무것도 지우지 않는다.
set -u
K=${1:?ami|amifull}; NAME=${2:?결과 폴더 이름}; C=${3:-30}
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results; AB=/soundai/users/tskim/VAPKT-data/data/ami-bench-v1
S=$R/$NAME; E=$R/$NAME-wn
[ -f $S/summary.json ] || { echo "원 결과가 아직 끝나지 않았다(summary.json 없음): $S"; exit 1; }
CK=$($P -I -c "import json, sys; print(json.load(open(sys.argv[1]))['checkpoint'])" $S/config-rank0.json)
case $K in
  ami) W=(--window 20,90 --window-draws 1 --batch-size 16) ;;
  amifull) W=(--window full --context-s $C) ;;
  *) echo "ami|amifull: $K"; exit 2 ;;
esac
mkdir -p $E; echo "[$(date "+%F %T")] 재채점 $K $S → $E(체크포인트 $CK — 토크나이저만)"
$P experiments/cst_stage3_eval.py run --checkpoint $CK --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 --deltas 4,10 --backchannel 1.0,3 --max-speakers 4 --declare 4 \
  "${W[@]}" --from-predictions $S --en-norm whisper --out $E >> $E/run-rank0.log 2>&1
grep -q "^DONE rank=0 " $E/run-rank0.log || { echo "재채점 실패 — $E/run-rank0.log"; tail -5 $E/run-rank0.log; exit 1; }
$P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
$P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $E > $E/ami-report.txt 2>&1
for n in whisper ours; do R_=(); [ $n = ours ] && R_=(--refs "")                                   # 외부 AMI 기준은 whisper 보고 끝에 한 번
  $P experiments/meeteval_cpwer.py --pred $S --norm $n "${R_[@]}" --json $E/meeteval-$n.json > $E/meeteval-$n.txt 2>&1 || { echo "meeteval $n 실패"; tail -5 $E/meeteval-$n.txt; }; done
echo "== 원 채점(ours, 단어마다)"; grep -E "^\s+ami" $S/ami-report.txt
echo "== Whisper 정규화(단어마다)"; grep -E "^\s+ami" $E/ami-report.txt
echo "== meeteval cpWER(화자 통째)"; cat $E/meeteval-whisper.txt $E/meeteval-ours.txt
echo "[$(date "+%F %T")] RESCORE_DONE $E"
