#!/usr/bin/env bash
# VAPKT 실시간 앱 설치 (audio-ui-spec §8): macOS 14+ 확인 → arm64 Python 3.11–3.14 고르기 → App/ dataless 거부 → ~/.vapkt-app/venv
# → 고정 버전 설치(--only-binary) → mx.metal 확인 → fake 엔진으로 pytest. 여러 번 실행해도 된다. ~/Desktop/VAPKT-models 는 건드리지 않는다.
#   bash App/setup.sh [--recreate] [--no-test]        (PYTHON=<경로> 로 Python 을, VAPKT_APP_HOME=<폴더> 로 설치 위치를 바꿀 수 있다)
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_HOME="${VAPKT_APP_HOME:-$HOME/.vapkt-app}"
VENV="$APP_HOME/venv"
RECREATE=0
RUN_TESTS=1
for a in "$@"; do
  case "$a" in
    --recreate) RECREATE=1 ;;
    --no-test) RUN_TESTS=0 ;;
    -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
    *) echo "알 수 없는 옵션: $a" >&2; exit 2 ;;
  esac
done

# macOS 14 이상: mlx 0.32.2·mlx-metal 휠은 macOS 14/15/26 arm64 만 있다. 더 낮은 mlx 로 내리지 않는다 (골든 대조 안 됨).
MACOS="$(/usr/bin/sw_vers -productVersion 2>/dev/null || echo 0)"
MACOS_MAJOR="${MACOS%%.*}"
case "$MACOS_MAJOR" in ''|*[!0-9]*) MACOS_MAJOR=0 ;; esac
if [ "$MACOS_MAJOR" -lt 14 ]; then
  echo "macOS 14 이상이 필요합니다 (현재 $MACOS; mlx 0.32.2 휠은 macOS 14/15/26 arm64 만 있다)." >&2
  exit 1
fi

# 사내 CA (TLS 가로채기 환경). 프록시 인증서 하나가 아니라 전체 CA 번들이어야 한다 (PyPI 등 다른 사이트도 이 파일로 검증).
CA="$HOME/.vapkt-ca-bundle.pem"
if [ -f "$CA" ]; then
  export PIP_CERT="$CA" SSL_CERT_FILE="$CA" REQUESTS_CA_BUNDLE="$CA" CURL_CA_BUNDLE="$CA"
  echo "CA 번들 사용: $CA"
fi
mkdir -p "$APP_HOME"
export PYTHONPYCACHEPREFIX="$APP_HOME/pycache"     # __pycache__ 를 iCloud Desktop 저장소 밖에

# 후보 Python 점검: 3.11–3.14, arm64, Rosetta 아님, venv·ensurepip 있음. 통과하면 버전을 stdout 에.
# 3.11 미만: numpy 2.4.6·filelock 4.2.0 휠이 없다. 3.15 이상: aiohttp 3.14.3 등 고정 휠이 없다 (pip --dry-run 로 확인, 2026-10-16).
check_py() {
  "$1" -I - <<'PY'
import platform, subprocess, sys
r = []
if sys.version_info < (3, 11):
    r.append(f"Python {platform.python_version()} < 3.11 (numpy 2.4.6·filelock 4.2.0 은 3.11+)")
elif sys.version_info >= (3, 15):
    r.append(f"Python {platform.python_version()} — 고정 휠(aiohttp 3.14.3 등)은 3.11–3.14 용")
if platform.machine() != "arm64":
    r.append(f"{platform.machine()} 빌드(arm64 아님, Intel/Rosetta)")
try:
    if subprocess.run(["/usr/sbin/sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True).stdout.strip() == "1":
        r.append("Rosetta 로 실행됨")
except Exception:
    pass
for m in ("venv", "ensurepip"):
    try:
        __import__(m)
    except Exception:
        r.append(f"{m} 모듈 없음")
if r:
    print("; ".join(r)); sys.exit(1)
print(platform.python_version())
PY
}

# 후보 순서 (§8 [critic]): $PYTHON → Apple Silicon Homebrew (3.12 먼저: 잠금 파일을 만든 버전) → conda → PATH 의 모든 python3.x
# (첫 이름에서 멈추지 않는다). Homebrew 기본 python3(현재 3.14)도 conda 보다 먼저 본다.
CANDS=()
if [ -n "${PYTHON:-}" ]; then CANDS+=("$PYTHON"); fi
for v in 3.12 3.11 3.13 3.14; do CANDS+=("/opt/homebrew/bin/python$v"); done
CANDS+=("/opt/homebrew/bin/python3")
CANDS+=("$HOME/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3" "$HOME/miniforge3/bin/python3")
for n in python3.12 python3.11 python3.13 python3.14 python3; do
  while IFS= read -r p; do
    if [ -n "$p" ]; then CANDS+=("$p"); fi
  done < <(which -a "$n" 2>/dev/null || true)
done

PY=""
SEEN=" "
for c in "${CANDS[@]}"; do
  [ -x "$c" ] || continue
  real="$(readlink -f "$c" 2>/dev/null || echo "$c")"
  case "$SEEN" in *" $real "*) continue ;; esac
  SEEN="$SEEN$real "
  if out="$(check_py "$c" 2>&1)"; then
    PY="$c"; echo "Python 선택: $c ($out)"; break
  else
    echo "  건너뜀: $c — $out"
  fi
done
if [ -z "$PY" ]; then
  echo "arm64 Python 3.11–3.14 를 찾지 못했습니다." >&2
  echo "  Apple Silicon Homebrew 로 설치:  brew install python@3.12   (/opt/homebrew/bin/python3.12)" >&2
  echo "  또는 PYTHON=<arm64 python 경로> bash App/setup.sh" >&2
  exit 1
fi

# App/ 안의 dataless 파일·폴더는 거부한다 — run.sh 와 같은 검사 (tests/test_packaging.py 가 두 사본이 같은지 본다).
# pip 가 requirements 를, pytest 가 App/*.py 를 읽다가 iCloud 를 기다리며 말없이 멈추지 않게, 아무것도 읽기 전에 본다.
"$PY" -I - "$APP_DIR" <<'PY' || exit 1
import os, sys
SF_DATALESS = 0x40000000
root, bad = sys.argv[1], []
def walk(d):
    if os.stat(d).st_flags & SF_DATALESS:
        bad.append(d + "/"); return
    with os.scandir(d) as it:
        for e in it:
            if e.name in ("__pycache__", ".pytest_cache"): continue
            st = e.stat(follow_symlinks=False)
            isdir = e.is_dir(follow_symlinks=False)
            if st.st_flags & SF_DATALESS: bad.append(e.path + ("/" if isdir else ""))
            elif isdir: walk(e.path)
walk(root)
if bad:
    print("iCloud 에서 내려받지 않은(dataless) 파일이 있어 실행하지 않습니다:", file=sys.stderr)
    for b in bad[:50]: print("  " + b, file=sys.stderr)
    print(f'해결: iCloud 가 되는 네트워크(사내망은 다운로드가 실패한다 — 핫스팟 등)에서 Finder "지금 다운로드" 또는  brctl download "{root}"  후 다시 실행하세요.', file=sys.stderr)
    sys.exit(1)
PY

# venv
if [ "$RECREATE" = 1 ] && { [ -e "$VENV" ] || [ -L "$VENV" ]; }; then
  echo "venv 다시 만들기: $VENV"; rm -rf "$VENV"
fi
if [ -x "$VENV/bin/python" ]; then
  if ! out="$(check_py "$VENV/bin/python" 2>&1)"; then
    echo "기존 venv 를 쓸 수 없습니다($out) — bash App/setup.sh --recreate" >&2; exit 1
  fi
  echo "기존 venv 사용: $VENV (Python $out)"
elif [ -e "$VENV" ] || [ -L "$VENV" ]; then
  # 남은 venv 인데 python 이 실행되지 않는다: 기본 Python 이 지워졌거나(끊긴 링크) 만들다 멈춘 것.
  # 그 위에 python -m venv 를 다시 돌리면 끊긴 링크가 그대로 남으므로 고치지 않고 멈춘다.
  why="bin/python 이 없습니다 (만들다 멈춘 venv?)"
  if [ -L "$VENV/bin/python" ]; then
    why="기본 Python 이 없어졌습니다 (bin/python → $("$PY" -I -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$VENV/bin/python") 없음)"
  fi
  home="$(sed -n 's/^home *= *//p' "$VENV/pyvenv.cfg" 2>/dev/null || true)"
  echo "기존 venv 를 쓸 수 없습니다: $VENV — $why${home:+ · pyvenv.cfg home = $home}" >&2
  echo "  다시 만들기: bash App/setup.sh --recreate" >&2
  exit 1
else
  echo "venv 만들기: $VENV"
  "$PY" -m venv "$VENV"
fi
VPY="$VENV/bin/python"
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
  echo "venv 에 pip 가 없습니다 (만들다 멈춘 venv?): $VENV — bash App/setup.sh --recreate" >&2; exit 1
fi

# 설치 (휠만; mlx 휠은 macOS 버전별)
PIP=("$VPY" -m pip install --disable-pip-version-check --only-binary :all:)
if [ -f "$APP_DIR/requirements.lock" ]; then
  REQ=(-r "$APP_DIR/requirements.lock")
else
  REQ=(-r "$APP_DIR/requirements.txt" -r "$APP_DIR/requirements-dev.txt")
fi
echo "패키지 설치: ${REQ[*]}"
if ! "${PIP[@]}" "${REQ[@]}"; then
  {
    echo "설치 실패 — venv $VENV ($("$VPY" --version 2>&1)), macOS $MACOS."
    echo "  위 pip 오류를 보세요. 'Requires-Python' 이 보이면 Python 버전(arm64 3.11–3.14), mlx 휠을 못 찾으면 macOS 버전(14+) 문제,"
    echo "  '(from versions: none)'·SSL·CERTIFICATE·연결 오류면 네트워크/CA 문제(~/.vapkt-ca-bundle.pem 에 전체 CA 번들)입니다."
    echo "  다른 Python 으로 다시: PYTHON=/opt/homebrew/bin/python3.12 bash App/setup.sh --recreate"
    # pip 24.2 부터 macOS 키체인 인증서(truststore)를 기본으로 쓴다 (릴리스 노트 기준). conda 3.11 의 ensurepip 는 pip 23.1.2.
    if [ ! -f "$CA" ] && "$VPY" -I -c 'import re, sys, pip; m = re.match(r"(\d+)\.(\d+)", pip.__version__); sys.exit(0 if m and tuple(map(int, m.groups())) < (24, 2) else 1)' 2>/dev/null; then
      echo "  참고: 이 venv 의 pip $("$VPY" -I -c 'import pip; print(pip.__version__)') 는 macOS 키체인 인증서를 쓰지 않는다(24.2 미만)."
      echo "        사내 TLS 가로채기 환경이면 ~/.vapkt-ca-bundle.pem 을 두거나 Homebrew Python 으로 --recreate 하세요."
    fi
  } >&2
  exit 1
fi

# 확인
"$VPY" - <<'PY'
import importlib.util, platform
import mlx.core as mx
import numpy, tokenizers, aiohttp, soxr
if not mx.metal.is_available():
    raise SystemExit("Metal GPU 를 쓸 수 없습니다 (mx.metal.is_available() = False)")
bad = [m for m in ("torch", "transformers", "nemo", "mlx_lm") if importlib.util.find_spec(m)]
if bad:
    print("경고: 런타임 venv 에 있으면 안 되는 패키지:", ", ".join(bad))
print(f"확인: Python {platform.python_version()} {platform.machine()} · macOS {platform.mac_ver()[0]} · mlx {mx.__version__} "
      f"(Metal {mx.metal.is_available()}) · numpy {numpy.__version__} · tokenizers {tokenizers.__version__} · "
      f"aiohttp {aiohttp.__version__} · soxr {soxr.__version__}")
PY

if [ "$RUN_TESTS" = 1 ]; then
  echo "테스트 (fake 엔진, 가중치 없음): pytest App/tests — 새 Mac 에서는 테스트 오디오·토크나이저가 필요한 것이 skipped 로 나온다 (정상)"
  if ! (cd "$APP_DIR" && "$VPY" -m pytest tests -q -p no:cacheprovider); then
    echo "설치는 끝났지만 테스트가 실패했습니다 — 위 출력을 확인하세요." >&2
    exit 1
  fi
fi
echo "완료. 실행: bash App/run.sh"
