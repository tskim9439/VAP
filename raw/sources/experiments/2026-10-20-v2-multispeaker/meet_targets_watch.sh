# 회의 번역 타깃(stage4-dlg-units-meet-v1): 수리 job(인자 1, 예 93808) 로그가 'all: 100.0%' 로 끝나면 타깃을 묶는다(mxc 호스트 CPU 16 프로세스).
# 생성·수리 진행은 5 분마다 찍는다. 수리 로그가 100 % 가 아닌 'all:' 로 끝나면(생성 실패 등) 묶지 않고 멈춘다. 아무것도 지우지 않는다.
set -u
JOB=$1; R=/soundai/users/tskim/VAPKT-data/data/stage4-dlg-units-meet-v1; LOG=/soundai/users/tskim/VAPKT/logs/train/SA_SFT_FullDuplex-$JOB.out
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
while true; do
  g=$(find $R/gen-v3-bf16 -name "u-*.jsonl.gz" 2>/dev/null | wc -l); f=$(find $R/fix-gen-v3-bf16 -name "u-*.jsonl.gz" 2>/dev/null | wc -l)
  echo "[$(date '+%F %T')] 생성 $g/164 · 수리 $f"
  if [ -f $LOG ] && grep -q "^errors:" $LOG; then
    if grep -q "^all: 100.0%" $LOG; then break; fi
    echo "!! 수리 job 이 100 % 가 아닌 상태로 끝났다 — 묶지 않는다"; grep "^all:\|^errors:" $LOG | tail -2; exit 1
  fi
  sleep 300
done
grep "^[a-z][a-z]-[a-z][a-z]:\|^all:\|^errors:" $LOG | tail -4
echo "[$(date '+%F %T')] 타깃 묶기"
/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python experiments/ast_build_targets.py --root $R --gen-dir gen-v3-bf16 --fix-dir fix-gen-v3-bf16 --pack --procs 16 2>&1 | grep -v "regex pattern" | tail -15
echo "[$(date '+%F %T')] MEET_TARGETS_DONE rc=${PIPESTATUS[0]}"
