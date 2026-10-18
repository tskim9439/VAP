# 되살리기 재료 빌드(stage3_revive_tokens.py)가 끝나면 로더 경로 점검(revive_probe.py) → dry-run(stage3_dry_revive.sh) → 빈 호스트 GPU 가 있으면
# 스모크 20 step(stage3_smoke_revive.sh) 을 차례로 돌린다. 로그 $G/chain.log, 끝나면 CHAIN_DONE. 아무것도 지우지 않는다.
set -u
G=/soundai/users/tskim/VAPKT-data/results/stage3-revive; PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
while pgrep -u "$(id -u)" -f "[s]tage3_revive_tokens" > /dev/null; do sleep 10; done
echo "빌드 끝 $(date +%T) · 코퍼스 $(grep -c '"corpus"' $G/build2.log)"
$PY $G/revive_probe.py $G/probe.json aihub71631,aihub134-1,aihub134-2,kec,ztc --show > $G/probe.log 2> $G/probe.err; echo "점검 끝 $(date +%T) EXIT $?"
bash $G/stage3_dry_revive.sh > $G/dry.log 2>&1; echo "dry-run 끝 $(date +%T) · $(tail -1 $G/dry.log)"
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1; exit}')
if [ -n "$GPU" ]; then GPUS=$GPU bash $G/stage3_smoke_revive.sh > $G/smoke.log 2>&1; echo "스모크 끝 GPU $GPU $(date +%T) · $(tail -1 $G/smoke.log)"
else echo "빈 GPU 없음 — 스모크 건너뜀 $(date +%T)"; fi
echo CHAIN_DONE
