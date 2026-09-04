#!/usr/bin/env bash
# mxc (Azure CycleCloud, H200 x8) 와 프로젝트 폴더를 동기화하고 컨테이너를 조작한다.
# 설정은 전부 저장소 루트의 .env 의 MXC_* 항목에서 읽는다 (.env.local 로 덮어쓰기 가능).
#
#   scripts/sync-mxc.sh status          환경 점검 (접속·컨테이너·경로·GPU·디스크·차이)
#   scripts/sync-mxc.sh push [-n]       로컬 → 원격 (정본은 로컬). -n 은 dry-run
#   scripts/sync-mxc.sh pull [-n]       원격 → 로컬. -n 은 dry-run
#
#   push/pull 모두 --delete 를 쓰지 않는다. 원격 파일은 어떤 경우에도 삭제하지
#   않는다(사용자 지시, 2026-09-04). 로컬에서 지운 파일은 원격에 그대로 남으므로
#   필요하면 사람이 직접 확인하고 지운다.
#   scripts/sync-mxc.sh shell           컨테이너 안에서 대화형 셸
#   scripts/sync-mxc.sh exec <명령...>  컨테이너 안 프로젝트 폴더에서 명령 실행
#   scripts/sync-mxc.sh bg <이름> <명령...>   컨테이너 안에서 세션과 무관하게(nohup) 실행
#   scripts/sync-mxc.sh jobs               bg 작업 상태 (실행 중 프로세스 + 최근 로그 꼬리)
#   scripts/sync-mxc.sh bigpush <로컬경로> <원격경로>   대용량 전송 (azcopy 있으면 azcopy, 없으면 rsync)
#
# 대용량(GB~TB) 은 rsync 대신 azcopy 를 쓴다 — .env 의 azcopy 섹션 참고.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

[[ -f .env ]] || { echo "오류: .env 가 없습니다 ($repo_root/.env)" >&2; exit 1; }
set -a; source .env; [[ -f .env.local ]] && source .env.local; set +a

: "${MXC_REMOTE_HOST:?.env 에 MXC_REMOTE_HOST 가 없습니다}"
: "${MXC_REMOTE_PROJECT_DIR:?.env 에 MXC_REMOTE_PROJECT_DIR 가 없습니다}"
: "${MXC_DOCKER_CONTAINER:?.env 에 MXC_DOCKER_CONTAINER 가 없습니다}"

REMOTE_HOST="$MXC_REMOTE_HOST"
REMOTE_PROJECT_DIR="$MXC_REMOTE_PROJECT_DIR"
DOCKER_CONTAINER="$MXC_DOCKER_CONTAINER"

# 안전장치: 원격 경로가 비었거나 공용 루트면 절대 진행하지 않는다
case "$REMOTE_PROJECT_DIR" in
  ""|"/"|"/root"|"/home"|"/soundai"|"/lustre"|"/soundai/users"|"/soundai/users/tskim"|"/lustre/soundai")
    echo "오류: MXC_REMOTE_PROJECT_DIR 이 위험한 값입니다: '$REMOTE_PROJECT_DIR'" >&2; exit 1 ;;
esac

SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
exclude_file="${SYNC_EXCLUDE_FILE:-.rsyncignore}"
# macOS 는 rsync 2.6.9 또는 openrsync 를 쓴다. 둘 다 --info/--human-readable 를 모른다.
# --delete 를 쓰지 않는다: 원격 파일은 어떤 경우에도 지우지 않는다(사용자 지시, 2026-09-04).
rsync_opts=(-az)
if rsync --help 2>&1 | grep -q -- '--info='; then
  rsync_opts+=(-O --human-readable --info=stats1)
else
  rsync_opts+=(--stats)
fi
[[ -f "$exclude_file" ]] && rsync_opts+=(--exclude-from="$exclude_file")

# 컨테이너 안에서 명령 실행 (IP 가 재시작마다 바뀌므로 SSH 직결 대신 docker exec 를 쓴다)
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
  $SSH "$REMOTE_HOST" 'for p in '"$REMOTE_PROJECT_DIR ${MXC_CKPT_ROOT:-} ${MXC_DATA_ROOT:-}"'; do
      if [ -d "$p" ]; then
        if touch "$p/.wtest" 2>/dev/null; then rm -f "$p/.wtest"; echo "  RW  $p"; else echo "  RO! $p"; fi
      else echo "  없음 $p"; fi
    done' 2>/dev/null

  echo "── 디스크 ───────────────────────────────────────"
  $SSH "$REMOTE_HOST" "df -h /soundai /lustre 2>/dev/null | awk 'NR==1||/^[0-9]/{printf \"  %s\\n\", \$0}'" 2>/dev/null

  echo "── GPU ──────────────────────────────────────────"
  $SSH "$REMOTE_HOST" "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /'" 2>/dev/null || echo "  (nvidia-smi 없음)"

  echo "── 동기화 차이 (로컬 → 원격, dry-run) ───────────"
  local out=""
  out="$(rsync "${rsync_opts[@]}" --dry-run --itemize-changes \
        -e "$SSH" ./ "$REMOTE_HOST:$REMOTE_PROJECT_DIR/" 2>/dev/null \
        | grep -E '^[<>ch*.]' | grep -vE '^\.d\.\.t' || true)"
  # openrsync 는 -O(--omit-dir-times) 를 모른다. 디렉터리 mtime 차이만 남는 줄은 잡음이므로 뺀다.
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
  rsync "${rsync_opts[@]}" ${dry} -e "$SSH" ./ "$REMOTE_HOST:$REMOTE_PROJECT_DIR/"
  [[ -z "$dry" ]] && echo "✓ push 완료"
  return 0
}

cmd_pull() {
  local dry=""
  if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then dry="--dry-run"; echo "(dry-run)"; fi
  # pull 은 로컬 전용 파일을 지우지 않는다 — --delete 를 뺀 옵션 집합을 새로 만든다.
  local pull_opts=(); local o
  for o in "${rsync_opts[@]}"; do [[ "$o" == "--delete" ]] || pull_opts+=("$o"); done
  echo "▶ pull: $REMOTE_HOST:$REMOTE_PROJECT_DIR → $repo_root"
  echo "  (--delete 를 쓰지 않으므로 로컬 전용 파일은 지워지지 않습니다)"
  rsync "${pull_opts[@]}" ${dry} -e "$SSH" "$REMOTE_HOST:$REMOTE_PROJECT_DIR/" ./
  [[ -z "$dry" ]] && echo "✓ pull 완료"
  return 0
}

cmd_shell() {
  echo "▶ $DOCKER_CONTAINER 진입 (작업 폴더: $REMOTE_PROJECT_DIR, activate-env.sh 자동 로드)"
  ssh -t "$REMOTE_HOST" "docker exec -it -w '$REMOTE_PROJECT_DIR' $DOCKER_CONTAINER bash --rcfile <(echo 'source ~/.bashrc 2>/dev/null; export VAPKT_PROFILE=mxc; source scripts/activate-env.sh')"
}

cmd_bg() {
  local name="${1:?bg <이름> <명령...>}"; shift; [[ $# -gt 0 ]] || { echo "실행할 명령이 없습니다." >&2; exit 2; }
  local ts; ts="$(date +%Y%m%d-%H%M%S)"
  local logdir="${MXC_DATA_LOG_DIR:-/tmp}"
  local log="${logdir}/bg-${name}-${ts}.log"
  local inner="export VAPKT_PROFILE=mxc; source scripts/activate-env.sh >/dev/null 2>&1; echo \"[bg:$name] start \$(date)\"; $*; echo \"[bg:$name] EXIT=\$? \$(date)\""
  $SSH "$REMOTE_HOST" "docker exec $DOCKER_CONTAINER bash -lc $(printf '%q' "mkdir -p '$logdir'")"
  # setsid + nohup: ssh 세션이 끊겨도 살아남는다. pid 를 로그 옆에 기록.
  $SSH "$REMOTE_HOST" "docker exec -d -w '$REMOTE_PROJECT_DIR' $DOCKER_CONTAINER bash -lc $(printf '%q' "setsid nohup bash -lc $(printf '%q' "$inner") > '$log' 2>&1 < /dev/null & echo \$! > '${log%.log}.pid'")"
  echo "▶ bg [$name] 시작 → 로그 $log"
  echo "   확인: scripts/sync-mxc.sh jobs   |   tail: scripts/sync-mxc.sh exec tail -f $log"
}

cmd_jobs() {
  in_container "" "for p in \$(ls -t ${MXC_DATA_LOG_DIR:-/tmp}/bg-*.pid 2>/dev/null | head -10); do n=\$(basename \$p .pid); pid=\$(cat \$p); if kill -0 \$pid 2>/dev/null; then st=RUNNING; else st=done; fi; printf '%-8s %-50s ' \$st \$n; tail -c 300 \${p%.pid}.log | tr '\\r' '\\n' | grep -avE '^\\s*\$' | tail -1 | cut -c1-90; done"
}

cmd_exec() {
  [[ $# -gt 0 ]] || { echo "실행할 명령이 없습니다." >&2; exit 2; }
  in_container "-w '$REMOTE_PROJECT_DIR'" "[ -f scripts/activate-env.sh ] && export VAPKT_PROFILE=mxc; source scripts/activate-env.sh >/dev/null 2>&1; $*"
}

# 대용량 전송. azcopy(병렬) 가 있으면 azcopy, 없으면 rsync 로 자동 폴백한다.
# azcopy 는 Azure Blob 엔드포인트 + SAS 가 있어야 하므로 MXC_AZCOPY_SAS_URL 이 필요하다
# (.env.local 에 두고 커밋하지 않는다).
cmd_bigpush() {
  local src="${1:?bigpush <로컬경로> <원격경로>}"; local dst="${2:?bigpush <로컬경로> <원격경로>}"
  [[ -e "$src" ]] || { echo "오류: 로컬 경로가 없습니다: $src" >&2; exit 1; }
  case "$dst" in /soundai/*|/lustre/*) ;; *) echo "오류: 원격 경로는 /soundai/ 또는 /lustre/ 아래여야 합니다: $dst" >&2; exit 1 ;; esac

  if command -v azcopy >/dev/null 2>&1 && [[ -n "${MXC_AZCOPY_SAS_URL:-}" ]]; then
    echo "▶ azcopy: $src → $dst  (SAS 인증)"
    echo "  주의: azcopy 대상은 Blob URL 이다. MXC_AZCOPY_SAS_URL 이 컨테이너 루트를 가리켜야 한다."
    azcopy copy "$src" "${MXC_AZCOPY_SAS_URL}" --recursive --overwrite=ifSourceNewer
  else
    if ! command -v azcopy >/dev/null 2>&1; then
      echo "ⓘ 로컬에 azcopy 가 없습니다 (brew install azcopy 로 설치하면 대용량 전송이 훨씬 빠릅니다). rsync 로 진행합니다."
    else
      echo "ⓘ MXC_AZCOPY_SAS_URL 이 .env.local 에 없습니다. rsync 로 진행합니다."
    fi
    $SSH "$REMOTE_HOST" "mkdir -p $(printf '%q' "$(dirname "$dst")")"
    rsync -a --partial --progress -e "$SSH" "$src" "$REMOTE_HOST:$dst"
  fi
  echo "✓ bigpush 완료"
}

case "${1:-status}" in
  status)  cmd_status ;;
  push)    shift; cmd_push "${1:-}" ;;
  pull)    shift; cmd_pull "${1:-}" ;;
  shell)   cmd_shell ;;
  exec)    shift; cmd_exec "$@" ;;
  bg)      shift; cmd_bg "$@" ;;
  jobs)    cmd_jobs ;;
  bigpush) shift; cmd_bigpush "$@" ;;
  *) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
