#!/usr/bin/env bash
# VAPKT 실시간 앱 실행 (audio-ui-spec §8): venv 확인 → App/ 의 iCloud dataless 파일 거부 → 모델 찾기(변환 없음, 원본 체크포인트를 바로 읽는다)
# → caffeinate 아래 서버. 인자는 server.py 로 그대로 넘긴다.
#   bash App/run.sh [--model DIR] [--port N] [--no-browser] [--precision auto|fp32|fp16] [--engine fake] ...
set -euo pipefail

# cd 하지 않는다: 상대 --model / VAPKT_APP_MODEL 은 부른 곳 기준으로 풀린다. App 파일은 모두 "$APP_DIR" 절대 경로로 부른다.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_HOME="${VAPKT_APP_HOME:-$HOME/.vapkt-app}"
PY="$APP_HOME/venv/bin/python"
if [ ! -x "$PY" ]; then
  if [ -L "$PY" ]; then
    echo "venv 의 기본 Python 이 없어졌습니다 ($PY 가 끊긴 링크) — bash App/setup.sh --recreate" >&2
  else
    echo "먼저 bash App/setup.sh 를 실행하세요 (venv 없음: $APP_HOME/venv)" >&2
  fi
  exit 1
fi
export PYTHONPYCACHEPREFIX="$APP_HOME/pycache" PYTHONUNBUFFERED=1

# App/ 안의 dataless 파일·폴더는 거부한다 (읽으면 iCloud 를 기다리며 멈출 수 있다). 폴더는 열기 전에 먼저 본다.
# setup.sh 에 같은 사본이 있다 (tests/test_packaging.py 가 두 사본이 같은지 본다).
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

# 인자에서 --engine / --model 읽기
ENGINE=mlx
MODEL="${VAPKT_APP_MODEL:-}"
prev=""
for a in "$@"; do
  case "$prev" in --engine) ENGINE="$a" ;; --model) MODEL="$a" ;; esac
  case "$a" in --engine=*) ENGINE="${a#--engine=}" ;; --model=*) MODEL="${a#--model=}" ;; esac
  prev="$a"
done

# 모델 찾기 (dataless 안전 — engine/checkpoint.py 만 단독으로 적재, MLX 를 불러오지 않는다).
# -I 는 PYTHONPYCACHEPREFIX 를 무시하므로 -X pycache_prefix 로 다시 준다 (저장소 안에 .pyc 를 쓰거나 읽지 않게).
EXTRA=()
if [ "$ENGINE" != "fake" ]; then
  if FOUND="$("$PY" -I -X pycache_prefix="$APP_HOME/pycache" - "$APP_DIR" "$MODEL" <<'PY'
import importlib.util, os, sys
app, model = sys.argv[1], sys.argv[2] or None
spec = importlib.util.spec_from_file_location("vapkt_checkpoint", os.path.join(app, "engine", "checkpoint.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
try:
    print(m.find_model(model))
except m.ModelNotFound as e:
    print(f"모델을 찾지 못했습니다: {e}", file=sys.stderr); sys.exit(3)
PY
  )"; then
    echo "모델: $FOUND"
    EXTRA=(--model "$FOUND")
  else
    rc=$?
    if [ "$rc" -eq 3 ]; then exit 1; fi
    echo "경고: 모델 사전 점검을 건너뜁니다 (rc $rc) — 서버가 다시 찾습니다" >&2
  fi
fi

# caffeinate: 긴 세션 중 유휴 잠자기 방지. 뒤의 "$@" 가 앞의 --model 보다 우선한다 (서버도 부른 곳을 cwd 로 쓰므로 상대 경로가 같게 풀린다).
exec /usr/bin/caffeinate -i "$PY" "$APP_DIR/server.py" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
