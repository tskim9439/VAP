# F1 대 mt4 비교(mxc 호스트, 2026-10-27): 평가 95866 이 끝난 뒤 bash f1_compare.sh — mt4(q17-v3-p1-tel-xlw-mt4, 2,500 step, 평가 ck1500·ck2000·final·avg3)와
#   F1(q17-f1, 5,000 step, ck3000·ck4000·final·avg3)을 q17 체인 README 결과 절과 같은 도구로 잰다. 결과 폴더는 읽기만 하고 TAXI paired json 만 새로 쓴다.
#   1) v3_runs_summary.py — dev(선택)·TAXI(보고)·같은 언어·대화 BLEU·xl·단일 화자 가드(기준 mt4 final)
#   2) ami_compare.py — AMI 공식 dev·eval × IHM-mix·SDM × δ4·10 회의 단위 짝 bootstrap(final 대 final, avg3 대 avg3)
#   3) cst_attr_turns.py --against — TAXI 세션 paired bootstrap 1,000 회(δ10·δ4, final 대 final, avg3 대 avg3)
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; X=raw/sources/experiments/2026-10-21-v3-attribution
R=/soundai/users/tskim/VAPKT-data/results; TAXI=/soundai/users/tskim/VAPKT-data/data/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl
M=q17-v3-p1-tel-xlw-mt4; F=q17-f1; C=$R/f1-compare; mkdir -p $C
echo "# F1 대 mt4 ($(date '+%F %T'))"
echo; echo "## 1. 묶음 요약(기준 mt4 final)"
$P $X/v3_runs_summary.py --base $M-final mt4=$M-final mt4-avg3=$M-avg3-final mt4-ck1500=$M-ck1500 mt4-ck2000=$M-ck2000 \
  f1=$F-final f1-avg3=$F-avg3-final f1-ck3000=$F-ck3000 f1-ck4000=$F-ck4000
echo; echo "## 2. AMI(회의 단위 짝 bootstrap) — 기준 mt4 final"
$P $X/ami_compare.py --base $R/stage3-eval-ami-$M-final $R/stage3-eval-ami-$F-final $R/stage3-eval-ami-$F-avg3-final
echo; echo "## 2b. AMI — 기준 mt4 avg3"
$P $X/ami_compare.py --base $R/stage3-eval-ami-$M-avg3-final $R/stage3-eval-ami-$F-avg3-final
echo; echo "## 3. TAXI 세션 paired bootstrap(F1 − mt4)"
for d in 10 4; do
  for t in final avg3-final; do
    echo "### δ$d $t"
    $P experiments/cst_attr_turns.py --sessions $TAXI --asr $R/stage4-eval-$F-$t/taxi90/asr-d$d.jsonl --against $R/stage4-eval-$M-$t/taxi90/asr-d$d.jsonl \
      --boot 1000 --seed 0 --out $C/taxi-$t-d$d.json
  done
done
echo; echo "F1_COMPARE_DONE"
