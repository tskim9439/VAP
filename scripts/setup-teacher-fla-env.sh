#!/usr/bin/env bash
# HF generate 용 Qwen3.8-27B env 에 Gated DeltaNet 빠른 경로(flash-linear-attention + causal-conv1d)를 붙인다.
# 기존 semcommit-teacher env 는 Codex 실험이 쓰므로 건드리지 않고, 같은 구성(vapasr 상속 + transformers 5.8.0)의 새 venv 를 만든다.
# 휠: fla 0.5.2·fla-core 0.5.2(순수 파이썬), causal-conv1d 1.7.0(cu12·torch2.9·cxx11abi TRUE·cp311 — vapasr 의 torch 2.9.0+cu128 과 맞춤).
#   mxc: bash scripts/setup-teacher-fla-env.sh
set -euo pipefail
VAPASR_PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
ENV=/soundai/users/tskim/VAPKT-data/conda/envs/teacher-fla
TF_WHEELS=/soundai/users/tskim/VAPKT-data/data/semcommit/wheelhouse/transformers-5.8.0
FLA_WHEELS=/soundai/users/tskim/VAPKT-data/data/semcommit/wheelhouse/fla-0.5.2

test -x "$VAPASR_PY"
(cd "$TF_WHEELS" && sha256sum -c --quiet SHA256SUMS)
(cd "$FLA_WHEELS" && sha256sum -c --quiet SHA256SUMS)
"$VAPASR_PY" -c 'import torch; assert torch.__version__.startswith("2.9.") and torch._C._GLIBCXX_USE_CXX11_ABI, torch.__version__'

if [[ ! -e "$ENV/pyvenv.cfg" ]]; then
  [[ -e "$ENV" ]] && { echo "!! venv 가 아닌 경로가 이미 있다: $ENV" >&2; exit 2; }
  "$VAPASR_PY" -m venv --system-site-packages "$ENV"
fi
"$ENV/bin/python" -m pip install --no-index --find-links "$TF_WHEELS" 'transformers==5.8.0'
"$ENV/bin/python" -m pip install --no-index --no-deps --find-links "$FLA_WHEELS" \
  'flash-linear-attention==0.5.2' 'fla-core==0.5.2' 'causal-conv1d==1.7.0'

"$ENV/bin/python" - <<'PY'
from importlib.metadata import version
import torch, transformers, causal_conv1d
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
print("transformers", transformers.__version__, "fla", version("flash-linear-attention"), "fla-core", version("fla-core"),
      "causal_conv1d", version("causal-conv1d"))
assert m.is_fast_path_available, "Qwen3.5 GDN fast path unavailable"
x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
y = causal_conv1d.causal_conv1d_fn(x, w, None, activation="silu")
print("causal_conv1d ok", tuple(y.shape), "fast path", m.is_fast_path_available)
PY
