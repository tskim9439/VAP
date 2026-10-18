# 체크포인트가 생기면 그때 가장 한가한 호스트 GPU(빈 메모리 최대, 같으면 사용률 낮은 쪽)를 골라 ckpt_eval_queue.sh 를 돌린다 — 호스트 GPU 는 다른 사용자와 같이 쓴다.
#   bash pick_gpu_queue.sh <run> <대상>   (대상 = checkpoint-N | final). 아무것도 지우지 않는다.
set -u
RUN=$1; T=$2; OUT=/soundai/Model/VAPASR/stage3-$RUN; G=/soundai/users/tskim/VAPKT-data/results/stage3-regress
echo "[$(date '+%F %T')] $RUN/$T 대기"
if [ "$T" = final ]; then until [ -f $OUT/DONE ]; do sleep 120; done; else until [ -f $OUT/$T/trainer_state.json ]; do sleep 120; done; sleep 60; fi
GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1 | cut -d' ' -f1)
echo "[$(date '+%F %T')] $RUN/$T → GPU $GPU ($(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $GPU))"
bash $G/ckpt_eval_queue.sh $GPU $RUN $T
