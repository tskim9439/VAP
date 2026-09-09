# 컨테이너 안에서:  source scripts/activate-env.sh
# .env 를 로드하고 conda env 를 활성화한다. sync-rack4.sh exec/shell 이 자동으로 source 한다.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; source "$_root/.env"; [ -f "$_root/.env.local" ] && source "$_root/.env.local"; set +a
# conda: mxc 프로필이면 .env 의 MXC_CONDA_DIR(/soundai/users/tskim/VAPKT-data/conda) 을, 아니면 /opt/conda 를 쓴다. 컨테이너 기본 python(/usr/local, transformers 5.x) 은 쓰지 않는다
_conda_dir="/opt/conda"; [ "${VAPKT_PROFILE:-}" = mxc ] && [ -n "${MXC_CONDA_DIR:-}" ] && _conda_dir="$MXC_CONDA_DIR"
if [ -f "$_conda_dir/etc/profile.d/conda.sh" ]; then
  source "$_conda_dir/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-vapasr}" 2>/dev/null || echo "(conda env '${CONDA_ENV_NAME:-vapasr}' 없음 — scripts/setup-container-env.sh 실행 필요)"
fi
unset _conda_dir
# 로컬 복사본(scripts/stage-env-local.sh, 기본 /dev/shm/vapasr-env)이 있으면 그 python 을 앞에 둔다 — NFS import 지연(수 분) 회피
_local="${VAPASR_LOCAL_ENV:-/dev/shm/vapasr-env}"
if [ -x "$_local/bin/python" ]; then
  export PATH="$_local/bin:$PATH" CONDA_PREFIX="$_local"
  # NeMo restore_from 은 .nemo(tar) 를 TMPDIR 에 풀어서 읽는다 — 컨테이너 / (overlay) 는 가득 차 있으므로 같은 tmpfs 를 쓴다
  export TMPDIR="${TMPDIR:-$(dirname "$_local")/vapasr-tmp}"; mkdir -p "$TMPDIR" 2>/dev/null || true
fi
unset _local
# 오늘 배정된 GPU (.env.local 의 GPU_DEFAULT). 명시적 CUDA_VISIBLE_DEVICES 가 있으면 그것을 우선.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_DEFAULT:-}}"
mkdir -p "$HF_HOME" "$NEMO_CACHE_DIR" "$TORCH_HOME" 2>/dev/null || true
unset _root
