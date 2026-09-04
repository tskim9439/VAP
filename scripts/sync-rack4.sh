#!/usr/bin/env bash
# rack4 학습 서버와 프로젝트 폴더를 동기화하고 컨테이너를 조작한다.
# 설정은 전부 저장소 루트의 .env 에서 읽는다 (.env.local 로 덮어쓰기 가능).
#
#   scripts/sync-rack4.sh status          환경 점검 (접속·컨테이너·경로·GPU·디스크·차이)
#   scripts/sync-rack4.sh push [-n]       로컬 → 원격 (정본은 로컬). -n 은 dry-run
#   scripts/sync-rack4.sh pull [-n]       원격 → 로컬 (--delete 없음, 안전)
#   scripts/sync-rack4.sh shell           컨테이너 안에서 대화형 셸
#   scripts/sync-rack4.sh exec <명령...>  컨테이너 안 프로젝트 폴더에서 명령 실행
#   scripts/sync-rack4.sh bg <이름> <명령...>   컨테이너 안에서 세션과 무관하게(nohup) 실행. 로그: $DATA_LOG_DIR/bg-<이름>-<ts>.log
#   scripts/sync-rack4.sh jobs               bg 작업 상태 (실행 중 프로세스 + 최근 로그 꼬리)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

[[ -f .env ]] || { echo "오류: .env 가 없습니다 ($repo_root/.env)" >&2; exit 1; }
set -a; source .env; [[ -f .env.local ]] && source .env.local; set +a

: "${REMOTE_HOST:?.env 에 REMOTE_HOST 가 없습니다}"
: "${REMOTE_PROJECT_DIR:?.env 에 REMOTE_PROJECT_DIR 가 없습니다}"
: "${DOCKER_CONTAINER:?.env 에 DOCKER_CONTAINER 가 없습니다}"

# 안전장치: 원격 경로가 비었거나 루트면 절대 진행하지 않는다 (--delete 사용)
case "$REMOTE_PROJECT_DIR" in
  ""|"/"|"/home"|"/root"|"/data3"|"/data4")
    echo "오류: REMOTE_PROJECT_DIR 이 위험한 값입니다: '$REMOTE_PROJECT_DIR'" >&2; exit 1 ;;
esac

SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
exclude_file="${SYNC_EXCLUDE_FILE:-.rsyncignore}"
# macOS 기본 rsync(2.6.9)는 --info 를 모른다. 지원 여부를 보고 고른다.
rsync_opts=(-az -O --stats)
if rsync --help 2>&1 | grep -q -- '--info='; then
  rsync_opts=(-az -O --human-readable --info=stats1)
fi
[[ -f "$exclude_file" ]] && rsync_opts+=(--exclude-from="$exclude_file")

# 컨테이너 안에서 명령 실행 (IP 가 바뀌므로 SSH 직결 대신 docker exec 를 쓴다)
in_container() {
  # shellcheck disable=SC2029
  $SSH "$REMOTE_HOST" "docker exec ${1:-} $DOCKER_CONTAINER bash -lc $(printf '%q' "$2")"
}

cmd_status() {
  echo "▶ 저장소      : $repo_root"
  echo "▶ 원격        : $REMOTE_HOST:$REMOTE_PROJECT_DIR"
  echo "▶ 컨테이너    : $DOCKER_CONTAINER"
  echo

  echo "── SSH ──────────────────────────────────────────"
  if $SSH "$REMOTE_HOST" 'echo "  접속 OK  ($(hostname))"' 2>/dev/null; then :; else
    echo "  ✗ $REMOTE_HOST 접속 실패"; exit 1
  fi

  echo "── 컨테이너 ─────────────────────────────────────"
  $SSH "$REMOTE_HOST" "docker inspect $DOCKER_CONTAINER \
    --format '  상태={{.State.Status}}  이미지={{.Config.Image}}  IP={{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'" \
    2>/dev/null || echo "  ✗ 컨테이너 $DOCKER_CONTAINER 없음"

  echo "── 경로 ─────────────────────────────────────────"
  # shellcheck disable=SC2016
  $SSH "$REMOTE_HOST" 'for p in '"$REMOTE_PROJECT_DIR ${CKPT_ROOT:-} ${DATA_ROOT:-}"'; do
      if [ -d "$p" ]; then
        if touch "$p/.wtest" 2>/dev/null; then rm -f "$p/.wtest"; echo "  RW  $p"; else echo "  RO! $p"; fi
      else echo "  없음 $p"; fi
    done' 2>/dev/null

  echo "── 디스크 ───────────────────────────────────────"
  $SSH "$REMOTE_HOST" "df -h ${DATA_ROOT:-/data3} ${CKPT_ROOT:-/data4} 2>/dev/null | awk 'NR==1||/^\\/dev/{printf \"  %s\\n\", \$0}'" 2>/dev/null

  echo "── GPU ──────────────────────────────────────────"
  $SSH "$REMOTE_HOST" "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /'" 2>/dev/null || echo "  (nvidia-smi 없음)"

  echo "── 동기화 차이 (로컬 → 원격, dry-run) ───────────"
  local out=""
  out="$(rsync "${rsync_opts[@]}" --delete --dry-run --itemize-changes \
        -e "$SSH" ./ "$REMOTE_HOST:$REMOTE_PROJECT_DIR/" 2>/dev/null \
        | grep -E '^[<>ch*.]' || true)"
  if [[ -z "$out" ]]; then echo "  동기화됨 — 차이 없음"; else
    echo "$out" | sed 's/^/  /' | head -40
    local n; n="$(printf '%s\n' "$out" | wc -l | tr -d ' ')"
    [[ "$n" -gt 40 ]] && echo "  ... 외 $((n-40))건"
  fi
}

cmd_push() {
  local dry=""
  if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then dry="--dry-run"; echo "(dry-run)"; fi
  $SSH "$REMOTE_HOST" "mkdir -p '$REMOTE_PROJECT_DIR'"
  echo "▶ push: $repo_root → $REMOTE_HOST:$REMOTE_PROJECT_DIR"
  rsync "${rsync_opts[@]}" ${dry} --delete -e "$SSH" ./ "$REMOTE_HOST:$REMOTE_PROJECT_DIR/"
  [[ -z "$dry" ]] && echo "✓ push 완료"
  return 0
}

cmd_pull() {
  local dry=""
  if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then dry="--dry-run"; echo "(dry-run)"; fi
  echo "▶ pull: $REMOTE_HOST:$REMOTE_PROJECT_DIR → $repo_root"
  echo "  (--delete 를 쓰지 않으므로 로컬 전용 파일은 지워지지 않습니다)"
  rsync "${rsync_opts[@]}" ${dry} -e "$SSH" "$REMOTE_HOST:$REMOTE_PROJECT_DIR/" ./
  [[ -z "$dry" ]] && echo "✓ pull 완료"
  return 0
}

cmd_shell() {
  echo "▶ $DOCKER_CONTAINER 진입 (작업 폴더: $REMOTE_PROJECT_DIR, activate-env.sh 자동 로드)"
  ssh -t "$REMOTE_HOST" "docker exec -it -w '$REMOTE_PROJECT_DIR' $DOCKER_CONTAINER bash --rcfile <(echo 'source ~/.bashrc 2>/dev/null; source scripts/activate-env.sh')"
}

cmd_bg() {
  local name="${1:?bg <이름> <명령...>}"; shift; [[ $# -gt 0 ]] || { echo "실행할 명령이 없습니다." >&2; exit 2; }
  local ts; ts="$(date +%Y%m%d-%H%M%S)"
  local log="${DATA_LOG_DIR:-/tmp}/bg-${name}-${ts}.log"
  local inner="source scripts/activate-env.sh >/dev/null 2>&1; echo \"[bg:$name] start \$(date)\"; $*; echo \"[bg:$name] EXIT=\$? \$(date)\""
  # setsid + nohup: ssh 세션이 끊겨도 살아남는다. pid 를 로그 옆에 기록.
  $SSH "$REMOTE_HOST" "docker exec -d -w '$REMOTE_PROJECT_DIR' $DOCKER_CONTAINER bash -lc $(printf '%q' "setsid nohup bash -lc $(printf '%q' "$inner") > '$log' 2>&1 < /dev/null & echo \$! > '${log%.log}.pid'")"
  echo "▶ bg [$name] 시작 → 로그 $log"
  echo "   확인: scripts/sync-rack4.sh jobs   |   tail: scripts/sync-rack4.sh exec tail -f $log"
}

cmd_jobs() {
  in_container "" "for p in \$(ls -t ${DATA_LOG_DIR:-/tmp}/bg-*.pid 2>/dev/null | head -10); do n=\$(basename \$p .pid); pid=\$(cat \$p); if kill -0 \$pid 2>/dev/null; then st=RUNNING; else st=done; fi; printf '%-8s %-50s ' \$st \$n; tail -c 300 \${p%.pid}.log | tr '\\r' '\\n' | grep -avE '^\\s*\$' | tail -1 | cut -c1-90; done"
}

cmd_exec() {
  [[ $# -gt 0 ]] || { echo "실행할 명령이 없습니다." >&2; exit 2; }
  in_container "-w '$REMOTE_PROJECT_DIR'" "[ -f scripts/activate-env.sh ] && source scripts/activate-env.sh >/dev/null 2>&1; $*"
}

case "${1:-status}" in
  status) cmd_status ;;
  push)   shift; cmd_push "${1:-}" ;;
  pull)   shift; cmd_pull "${1:-}" ;;
  shell)  cmd_shell ;;
  exec)   shift; cmd_exec "$@" ;;
  bg)     shift; cmd_bg "$@" ;;
  jobs)   cmd_jobs ;;
  *) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
