#!/usr/bin/env bash
# tskim_env 컨테이너 안에서 실행: bash scripts/setup-container-env.sh [단계...]
# 단계: conda torch nemo hf verify  (인자 없으면 전부). 각 단계는 재실행해도 안전하다.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
set -a; source .env; [ -f .env.local ] && source .env.local; set +a
source /opt/conda/etc/profile.d/conda.sh
ENV="${CONDA_ENV_NAME:-vapasr}"; PY="${CONDA_PYTHON:-3.11}"
mkdir -p "$HF_HOME" "$NEMO_CACHE_DIR" "$TORCH_HOME" "${DATA_LOG_DIR:-/tmp}"
steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(conda torch nemo hf verify)
log(){ printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$*"; }

for s in "${steps[@]}"; do case "$s" in
conda)
  log "conda env $ENV (python $PY)"
  conda env list | grep -qE "^$ENV\s" || conda create -y -n "$ENV" "python=$PY"
  conda activate "$ENV"; python -V; pip install -q -U pip wheel setuptools ;;
torch)
  log "torch/torchaudio (cu124 — driver 550 / CUDA 12.4)"
  conda activate "$ENV"
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  python -c "import torch,torchaudio;print('torch',torch.__version__,'torchaudio',torchaudio.__version__,'cuda',torch.cuda.is_available(),'gpus',torch.cuda.device_count())" ;;
nemo)
  log "NeMo toolkit [asr] (Nemotron 3.5 ASR streaming 용)"
  conda activate "$ENV"
  pip install Cython packaging
  pip install "nemo_toolkit[asr]"          # 의존성 확보용 (PyPI)
  # 모델 카드 권장: git main. 의존성은 위에서 확보했으므로 코드만 교체.
  pip install --no-deps "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit"
  python -c "import nemo, nemo.collections.asr as a; print('nemo', nemo.__version__)" ;;
hf)
  log "transformers / accelerate / 오디오 / 유틸"
  conda activate "$ENV"
  pip install -U transformers accelerate huggingface_hub soundfile librosa python-dotenv datasets einops matplotlib soxr
  pip install -U qwen-asr   # Qwen3-ASR / Qwen3-ForcedAligner 로더
  python -c "import transformers;print('transformers',transformers.__version__)" ;;
verify)
  log "검증"
  conda activate "$ENV"
  python - <<'PY'
import torch, os
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(f"  gpu{i}: {p.name} {p.total_memory/2**30:.0f}GB")
x = torch.randn(1024,1024,device="cuda"); print("matmul ok", (x@x).sum().item() != 0)
print("HF_HOME", os.environ.get("HF_HOME")); print("NEMO_CACHE_DIR", os.environ.get("NEMO_CACHE_DIR"))
PY
  pip freeze > "$root/env/requirements-lock.txt" 2>/dev/null || true
  echo "lock → env/requirements-lock.txt" ;;
*) echo "알 수 없는 단계: $s" >&2; exit 2 ;;
esac; done
log "완료"
