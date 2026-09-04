# 컨테이너 안에서:  source scripts/activate-env.sh
# .env 를 로드하고 conda env 를 활성화한다. sync-rack4.sh exec/shell 이 자동으로 source 한다.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; source "$_root/.env"; [ -f "$_root/.env.local" ] && source "$_root/.env.local"; set +a
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-vapasr}" 2>/dev/null || echo "(conda env '${CONDA_ENV_NAME:-vapasr}' 없음 — scripts/setup-container-env.sh 실행 필요)"
fi
# 오늘 배정된 GPU (.env.local 의 GPU_DEFAULT). 명시적 CUDA_VISIBLE_DEVICES 가 있으면 그것을 우선.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_DEFAULT:-}}"
mkdir -p "$HF_HOME" "$NEMO_CACHE_DIR" "$TORCH_HOME" 2>/dev/null || true
unset _root
