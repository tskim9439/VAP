#!/usr/bin/env bash
# 스트리밍 음성번역 베이스라인 env(오프라인) — q06-s2-ast 와 같은 평가셋·같은 지연 정의로 비교할 때 쓴다(2026-10-11 사용자 결정).
# 가중치·휠·코드는 맥에서 받아(T5 vapkt-baselines) 사용자가 azcopy 로 $BL 에 올렸다(구성: $BL/README.md). mxc 는 인터넷을 쓰지 않는다.
#   mxc 호스트: bash scripts/setup-baseline-envs.sh seamless|iwslt|hibiki|all   → 로컬 디스크 $DST/<이름> 에 설치 + import 검사
#     seamless : SeamlessStreaming(fairseq2 0.2.1, torch 2.2.2) — simuleval 은 --no-deps(bitarray==2.6.0 고정 휠 없음, 코드에서 안 씀)
#     iwslt    : IWSLT 2026 공식 연결형(Qwen3-ASR-1.7B → Qwen3-4B-Instruct-2507, vllm 0.14.0, qwen-asr 0.0.6, simulstream 1.0.0)
#     hibiki   : vapasr env(torch 2.9.0) 위 --system-site-packages venv 에 moshi·sphn·tiktoken(+ ctranslate2) 만 --no-deps 로 얹는다(Hibiki-Zero·SimulStreaming)
# 아무것도 지우지 않는다: venv 가 아닌 기존 경로에는 설치하지 않는다.
set -euo pipefail
BASE_PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python3.11
BL=/soundai/users/tskim/VAPKT-data/vapkt-baselines
W=$BL/wheels
DST=${BASELINE_ENVS:-/tmp/sa_tskim/bl-envs}

venv() {  # venv <이름> [--system-site-packages]
  local d="$DST/$1"; shift
  if [[ ! -e "$d/pyvenv.cfg" ]]; then
    [[ -e "$d" ]] && { echo "!! venv 가 아닌 경로가 이미 있다: $d" >&2; exit 2; }
    mkdir -p "$DST"; "$BASE_PY" -m venv "$@" "$d"
  fi
  echo "$d/bin/python"
}

seamless() {
  local py; py=$(venv seamless)
  "$py" -m pip install -q --no-index --find-links "$W/seamless" -r "$W/seamless.lock.txt"
  "$py" -m pip install -q --no-index --no-deps "$W"/seamless/simuleval-1.1.4-py3-none-any.whl
  "$py" -m pip install -q --no-index --no-deps "$W"/iwslt/sentencepiece-0.2.2-*.whl              # simuleval 의 spm 지연 단위가 요구(seamless 잠금에는 없음)
  local src="$DST/src/seamless_communication"                               # $BL 은 읽기 전용(azcopy 업로드) → 빌드는 로컬 사본에서
  [[ -e "$src" ]] || { mkdir -p "$DST/src"; cp -r "$BL/code/seamless_communication" "$src"; }
  "$py" -m pip install -q --no-index --no-deps --no-build-isolation "$src"
  "$py" - <<'PY'
import torch, fairseq2, simuleval, sentencepiece
from seamless_communication.streaming.agents.seamless_streaming_s2t import SeamlessStreamingS2TAgent
print("seamless ok · torch", torch.__version__, "cuda", torch.version.cuda, "fairseq2", fairseq2.__version__, "gpus", torch.cuda.device_count())
PY
}

iwslt() {
  local py; py=$(venv iwslt)
  "$py" -m pip install -q --no-index --find-links "$W/iwslt" -r "$W/iwslt.lock.txt"
  "$py" - <<'PY'
import torch, vllm, qwen_asr, simulstream, transformers
print("iwslt ok · vllm", vllm.__version__, "torch", torch.__version__, "transformers", transformers.__version__, "gpus", torch.cuda.device_count())
PY
}

hibiki() {
  local py; py=$(venv hibiki --system-site-packages)
  "$py" -m pip install -q --no-index --no-deps "$W"/overlay-hibiki/*.whl
  if compgen -G "$W/overlay-simulstreaming/ctranslate2-*.whl" > /dev/null; then                 # SimulStreaming 연결형(EuroLLM, CTranslate2)
    "$py" -m pip install -q --no-index --no-deps "$W"/overlay-simulstreaming/ctranslate2-*.whl
  fi
  PYTHONPATH="$BL/code/hibiki-zero:$BL/code/SimulStreaming" "$py" - <<'PY'
import numpy, torch, moshi, sphn, tiktoken
from hibiki_zero.inference import decode_outputs, encode_inputs, get_lmgen
print("hibiki ok · moshi", moshi.__version__, "torch", torch.__version__, "numpy", numpy.__version__, "gpus", torch.cuda.device_count())
PY
}

case "${1:-}" in
seamless|iwslt|hibiki) "$1" ;;
all) seamless; iwslt; hibiki ;;
*) echo "usage: $0 seamless|iwslt|hibiki|all" >&2; exit 64 ;;
esac
