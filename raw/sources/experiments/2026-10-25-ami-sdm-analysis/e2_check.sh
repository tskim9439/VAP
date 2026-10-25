# E2 점검(mxc 호스트, 2026-10-25): 대상 화자 모드(--target-spk, 인코더 pre-encode 활동 주입) 1.7B 학습 dry-run(CPU) → 스모크 20 step(가장 빈 호스트 GPU).
#   bash e2_check.sh [smoke 산출 꼬리]   — SMOKE=0 이면 dry-run 만. 산출 /tmp/sa_tskim/e2-{dry,smoke}-train — 지우지 않는다.
# 학습 데이터: 대화 10 셋(회의 AMI·ICSI·NOTSOFAR 포함, split v4) + 합성(3–4 인 v3-multi, 교차 언어·전화) — replay(단일 화자)는 쓰지 않는다(가중에 rep_* 없음).
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; INIT=${INIT:-/soundai/Model/VAPASR/stage3-q17-v3-p1-tel-xlw-mt4/final}
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
W=dlg_ko:2,dlg_en:0.7,dlg_de:0.7,dlg_mt:1.5,sim_ko:0.8,sim_en:0.8,sim_de:0.8,sim_en-de:0.5,sim_en-de-tel:0.5,sim_en-ko:0.5
COMMON="--init $INIT --nemotron-dir $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v4.json --dlg-corpora aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar \
  --max-windows 300 --sim-data $D/sim-dialogue-v3-multi,$D/sim-dialogue-xl-v1,$D/sim-dialogue-xl-tel-v1 --weights $W --set-phone sim_en-de-tel:0 \
  --train-encoder --encoder-right-context 3 --encoder-context-sampling fixed --spec-augment 2,27,2,0.03 --attn-impl varlen --batch-max-tokens 16384 --pack-max-bs 64 \
  --backchannel 1.0,3 --spk-act-weight 0 --revive-dir $P2/revive-asr-v2 --phone-aug 0.3 --max-speakers 4 --target-spk --inject-hidden 512 \
  --lr 4e-5 --max-steps 20 --warmup 5 --save-every 1000 --log-every 2 --num-workers 2 --report-to none"
echo "[$(date "+%F %T")] e2 dry-run"
$P experiments/stage3_train.py --dry-run $COMMON --out-dir /tmp/sa_tskim/e2-dry-train --check-items 32 2>&1 | grep -v "regex pattern"
echo "DRY EXIT ${PIPESTATUS[0]}"
[ "${SMOKE:-1}" = 0 ] && { echo "E2_CHECK_DONE"; exit 0; }
GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F", " "{print \$1, \$3 - \$2, \$4}" | sort -k2,2nr -k3,3n | head -1 | cut -d" " -f1); echo "GPU $GPU"
echo "[$(date "+%F %T")] e2 smoke"
$P experiments/stage3_train.py --gpu $GPU $COMMON --out-dir /tmp/sa_tskim/e2-smoke-train${1:+-$1} --check-items 8 2>&1 | grep -v "regex pattern"
echo "SMOKE EXIT ${PIPESTATUS[0]}"
