# E6 비교(mxc 호스트, 2026-10-27): 평가 95936 이 끝난 뒤 bash e6_compare.sh — E6(q17-e6: F1 레시피 + 슬롯 활동 입력 + 인과 화자 분리 헤드, 헤드 예측 활동으로 디코드)를
#   F1(같은 레시피, 활동 입력 없음)과, AMI 는 E5 정답 활동 상한(amiact)·E6 정답 활동(amiact)과도 잰다. f1_compare.sh 와 같은 도구. 결과 폴더는 읽기만 하고 TAXI paired json 만 새로 쓴다.
#   1) v3_runs_summary.py — dev·TAXI·같은 언어·BLEU·가드(기준 F1 final, mt4 도 함께)
#   2) ami_compare.py — 회의 단위 짝 bootstrap: 기준 F1 final 대 E6 final·avg3, 기준 E5 amiact 대 E6 amiact·E6 final(헤드 예측이 상한에서 얼마나 먼가)
#   3) cst_attr_turns.py --against — TAXI 세션 paired bootstrap(E6 − F1, δ10·δ4, final·avg3)
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; X=raw/sources/experiments/2026-10-21-v3-attribution
R=/soundai/users/tskim/VAPKT-data/results; TAXI=/soundai/users/tskim/VAPKT-data/data/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl
M=q17-v3-p1-tel-xlw-mt4; F=q17-f1; E=q17-e6; C=$R/e6-compare; mkdir -p $C
echo "# E6 대 F1 ($(date '+%F %T'))"
echo; echo "## 1. 묶음 요약(기준 F1 final)"
$P $X/v3_runs_summary.py --base $F-final mt4=$M-final f1=$F-final f1-avg3=$F-avg3-final e6=$E-final e6-avg3=$E-avg3-final e6-ck3000=$E-ck3000 e6-ck4000=$E-ck4000
echo; echo "## 2. AMI — 기준 F1 final(E6 은 헤드 예측 활동)"
$P $X/ami_compare.py --base $R/stage3-eval-ami-$F-final $R/stage3-eval-ami-$E-final $R/stage3-eval-ami-$E-avg3-final
echo; echo "## 2b. AMI — 기준 F1 avg3"
$P $X/ami_compare.py --base $R/stage3-eval-ami-$F-avg3-final $R/stage3-eval-ami-$E-avg3-final
echo; echo "## 2c. AMI — 기준 E5 정답 활동(amiact): E6 정답 활동·E6 헤드 예측"
$P $X/ami_compare.py --base $R/stage3-eval-amiact-q17-e5-final $R/stage3-eval-amiact-$E-final $R/stage3-eval-ami-$E-final
echo; echo "## 3. TAXI 세션 paired bootstrap(E6 − F1)"
for d in 10 4; do
  for t in final avg3-final; do
    echo "### δ$d $t"
    $P experiments/cst_attr_turns.py --sessions $TAXI --asr $R/stage4-eval-$E-$t/taxi90/asr-d$d.jsonl --against $R/stage4-eval-$F-$t/taxi90/asr-d$d.jsonl \
      --boot 1000 --seed 0 --out $C/taxi-$t-d$d.json
  done
done
echo; echo "E6_COMPARE_DONE"
