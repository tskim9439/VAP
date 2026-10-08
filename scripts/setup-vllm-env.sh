#!/usr/bin/env bash
# vLLM 0.30.0 오프라인 env — Qwen3.8-27B 로 AST 단조 번역을 대량 생성할 때 쓴다.
# torch 2.13+cu130 이라 vapasr(torch 2.9)·teacher env 와 섞지 않고 따로 둔다(base python 만 vapasr 의 3.11 을 빌린다).
# 휠은 맥에서 받아(T5 vapkt-wheels/vllm-0.30.0) 사용자가 WHEELS 로 올린다. mxc 는 인터넷을 쓰지 않는다.
#   mxc 호스트: bash scripts/setup-vllm-env.sh build   → 로컬 디스크 $DST 에 설치 + 검사 (NFS 에 직접 깔면 파일당 왕복으로 느리다)
#               bash scripts/setup-vllm-env.sh pack    → $TAR (노드가 풀어 쓰는 단일 아카이브)
#   노드(sbatch): bash scripts/setup-vllm-env.sh stage → $TAR 를 노드 로컬 $DST 에 푼다(같은 tar 면 건너뜀)
# 아무것도 지우지 않는다: 이미 있는 tar 는 덮어쓰지 않고, venv 가 아닌 기존 경로에는 설치하지 않는다.
set -euo pipefail
BASE_PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python3.11
WHEELS=/soundai/users/tskim/VAPKT-data/data/semcommit/wheelhouse/vllm-0.30.0
TAR=/soundai/users/tskim/VAPKT-data/conda/envs/vllm-0.30.tar
DST=${VLLM_ENV:-/tmp/sa_tskim/vllm-0.30}

smoke() {
  "$DST/bin/python" - <<'PY'
import torch, transformers, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda,
      "transformers", transformers.__version__, "gpus", torch.cuda.device_count())
assert vllm.__version__.startswith("0.30.0"), vllm.__version__
PY
}

case "${1:-}" in
build)
  test -x "$BASE_PY"
  (cd "$WHEELS" && sha256sum -c --quiet SHA256SUMS)
  if [[ ! -e "$DST/pyvenv.cfg" ]]; then
    [[ -e "$DST" ]] && { echo "!! venv 가 아닌 경로가 이미 있다: $DST" >&2; exit 2; }
    mkdir -p "$(dirname "$DST")"
    "$BASE_PY" -m venv "$DST"
  fi
  "$DST/bin/python" -m pip install --no-index --find-links "$WHEELS" -r "$WHEELS/requirements.lock.txt"
  "$DST/bin/python" -m pip check
  smoke
  ;;
pack)
  [[ -x "$DST/bin/python" ]] || { echo "!! 먼저 build: $DST" >&2; exit 1; }
  [[ -e "$TAR" ]] && { echo "!! 이미 있다(덮어쓰지 않음): $TAR" >&2; exit 2; }
  tar -C "$DST" -cf "$TAR.partial" . && mv "$TAR.partial" "$TAR"
  ls -la "$TAR"
  ;;
stage)
  stamp="$TAR@$(stat -c %Y "$TAR")"
  if [[ -x "$DST/bin/python" && "$(cat "$DST/.staged-from" 2>/dev/null)" == "$stamp" ]]; then
    echo "이미 풀림: $DST"; exit 0
  fi
  mkdir -p "$DST"; t0=$(date +%s)
  tar -C "$DST" -xf "$TAR"
  echo "$stamp" > "$DST/.staged-from"
  echo "풀기 완료: $DST ($(( $(date +%s) - t0 )) s)"
  smoke
  ;;
*)
  echo "usage: $0 build|pack|stage" >&2; exit 64 ;;
esac
