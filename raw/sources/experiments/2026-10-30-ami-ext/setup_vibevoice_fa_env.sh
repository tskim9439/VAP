# VibeVoice Flash Attention 대조용 별도 env(2026-10-30, 사용자 "별도 env 를 만들어 대조해보자"): 서버 호스트에서 한 번. 학습 env(vapasr, torch 2.9)는 건드리지 않는다.
#   torch 2.9 용 공식 flash-attn 휠이 없어(최신 2.8.3.post2 는 torch ≤ 2.8), vapasr env 를 바탕으로 한 venv(--system-site-packages)에 torch 2.8.0(cu128)과
#   공식 휠 flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311 을 넣어 덮는다(transformers·numpy 등 나머지는 vapasr env 그대로).
#   bash setup_vibevoice_fa_env.sh → VAPKT-data/baselines/vibevoice-fa-v1/{venv, environment.freeze.txt, SETUP_READY}. 이미 있으면 멈춘다(덮어쓰지 않는다).
set -euo pipefail
BASE=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; ROOT=/soundai/users/tskim/VAPKT-data/baselines/vibevoice-fa-v1
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
test ! -e "$ROOT" || { echo "이미 있다: $ROOT — 덮어쓰지 않는다"; exit 1; }
mkdir -p "$ROOT"; echo "[$(date '+%F %T')] venv ← $BASE"
"$BASE" -m venv --system-site-packages "$ROOT/venv"; PY="$ROOT/venv/bin/python"
"$PY" -m pip install --no-compile "torch==2.8.0"
"$PY" -m pip install --no-compile --no-deps "$FA"
"$PY" -m pip freeze > "$ROOT/environment.freeze.txt"
"$PY" - <<'PY'
import torch, flash_attn
from flash_attn import flash_attn_func
print("torch", torch.__version__, "cuda", torch.version.cuda, "cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI, "flash_attn", flash_attn.__version__, "file", torch.__file__)
assert torch.__version__.startswith("2.8.0") and torch._C._GLIBCXX_USE_CXX11_ABI
if torch.cuda.is_available():
    q = torch.randn(1, 64, 8, 128, device="cuda", dtype=torch.bfloat16); o = flash_attn_func(q, q, q, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2), is_causal=True).transpose(1, 2)
    print("flash_attn_func 대 sdpa 최대 차", float((o.float() - ref.float()).abs().max()))
PY
touch "$ROOT/SETUP_READY"; echo "[$(date '+%F %T')] SETUP_READY $ROOT"
