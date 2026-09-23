#!/usr/bin/env bash
# 평가 전용 venv·소스·가중치 준비. 학습 env는 읽기만, 기존 평가 env는 덮지 않음.
set -euo pipefail
BASE_PYTHON=${BASE_PYTHON:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}
EVAL_ROOT=${EVAL_ROOT:-/soundai/users/tskim/VAPKT-data/baselines/vibevoice-1.5b-v1}
CODE_REV=1541f590c7099820f10ea012f48d2399282df69f
MODEL_REV=4262d23d8a539a6530cf64fbd0b1751ef9a30853
test ! -e "$EVAL_ROOT" || { echo "이미 존재: $EVAL_ROOT — 자동 덮어쓰기 안 함"; exit 1; }
mkdir -p "$EVAL_ROOT"
"$BASE_PYTHON" -m venv --system-site-packages "$EVAL_ROOT/venv"
PY="$EVAL_ROOT/venv/bin/python"
git clone https://github.com/microsoft/VibeVoice.git "$EVAL_ROOT/source"
git -C "$EVAL_ROOT/source" checkout --detach "$CODE_REV"
# torch/CUDA는 검증된 학습 env에서 읽기 전용 상속. 나머지는 전용 venv에 설치.
"$PY" -m pip install --no-compile 'transformers==4.57.1' 'meeteval==0.4.3' 'num2words==0.5.14' \
  accelerate diffusers librosa soundfile scipy ml-collections absl-py
# 채점기 자체는 상위 env 변경의 영향을 덜 받도록 전용 env에도 고정 설치.
"$PY" -m pip install --no-compile --ignore-installed --no-deps 'meeteval==0.4.3'
"$PY" -m pip install --no-compile --no-deps -e "$EVAL_ROOT/source"
"$PY" - "$EVAL_ROOT/model" "$MODEL_REV" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
dest, revision = sys.argv[1:]
snapshot_download("microsoft/VibeVoice-ASR-Streaming-1.5B", revision=revision, local_dir=dest)
Path(dest, "baseline-provenance.json").write_text(json.dumps({"revision": revision}))
PY
"$PY" -m pip freeze > "$EVAL_ROOT/environment.freeze.txt"
"$PY" - <<'PY'
import importlib.metadata
from pathlib import Path
import sys
import meeteval
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
assert importlib.metadata.version("meeteval") == "0.4.3"
assert importlib.metadata.version("transformers") == "4.57.1"
assert Path(meeteval.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print("imports OK; MeetEval:", meeteval.__file__)
PY
touch "$EVAL_ROOT/SETUP_READY"
echo "설정 완료: $EVAL_ROOT"
