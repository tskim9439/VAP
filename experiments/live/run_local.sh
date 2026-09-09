#!/usr/bin/env bash
# 로컬 Mac 에서 실시간 음성인식 서버 실행:  PORT=8790 bash experiments/live/run_local.sh   (이 Mac 의 8765 는 다른 로컬 서비스(uvicorn)가 쓰고 있어 기본 8790)
# 전제: scripts/local-env.sh 가 만드는 conda env vapasr-local, ~/Desktop/VAPKT-models/{hf-C2-final, nemotron-encoder.pt}
set -euo pipefail
cd "$(dirname "$0")/../.."; source scripts/local-env.sh
MODEL=${MODEL:-$VAPASR_LOCAL_MODELS/hf-C2-final}
[ -f "$MODEL/model.safetensors" ] || { echo "!! 체크포인트 없음: $MODEL"; exit 1; }; [ -f "$VAPASR_ENCODER_CACHE" ] || { echo "!! 인코더 캐시 없음: $VAPASR_ENCODER_CACHE"; exit 1; }
DEV=${DEVICE:-mps}; python -c "import torch,sys; sys.exit(0 if (torch.backends.mps.is_available() if '$DEV'=='mps' else True) else 1)" || DEV=cpu
echo "model=$MODEL device=$DEV"; exec python -u experiments/live/server.py --model "$MODEL" --device "$DEV" --default-delay "${DELAY:-4}" --port "${PORT:-8790}" "$@"
