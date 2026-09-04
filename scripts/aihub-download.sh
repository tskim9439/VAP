#!/usr/bin/env bash
# AI Hub 데이터셋을 rack4 호스트에서 직접 내려받는다 (aihubshell). 승인된 계정의 API 키 필요.
# ※ 2026-09-03: 감정 태깅 자유대화(71631/71632)는 마이페이지에 API 발급이 없고 'PC 다운로드만 가능'.
#   → 이 스크립트 대신 scripts/aihub-upload.sh (맥에서 받아 서버로 전송) 를 쓴다.
#
#   scripts/aihub-download.sh list  <adult|teen|datasetkey>          파일 목록 (filekey 확인)
#   scripts/aihub-download.sh get   <adult|teen|datasetkey> [filekey,...]   다운로드 (filekey 생략 시 전체)
#
# 실행 위치: rack4 호스트 (컨테이너 아님 — 소유권이 tskim 으로 남도록). 로컬에서는
#   ssh rack4 'cd /home/tskim/VAP && scripts/aihub-download.sh list adult'
# API 키: .env.local 의 AIHUB_API_KEY (절대 커밋 금지). 없으면 AI Hub 마이페이지에서 발급.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
set -a; source .env; [ -f .env.local ] && source .env.local; set +a

mode="${1:-}"; ds="${2:-}"; keys="${3:-}"
case "$ds" in adult) ds="$AIHUB_DATASET_ADULT";; teen) ds="$AIHUB_DATASET_TEEN";; esac
[[ -n "$mode" && -n "$ds" ]] || { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

bin="$root/.aihubshell"
if [[ ! -x "$bin" ]]; then
  echo "▶ aihubshell 내려받기"; curl -sSf -o "$bin" https://api.aihub.or.kr/api/aihubshell.do; chmod +x "$bin"
fi

case "$mode" in
  list) "$bin" -mode l -datasetkey "$ds" ;;
  get)
    : "${AIHUB_API_KEY:?.env.local 에 AIHUB_API_KEY 가 없습니다}"
    dest="${AIHUB_DOWNLOAD_DIR:-/data3/tskim/corpora/aihub}/$ds"; mkdir -p "$dest"; cd "$dest"
    echo "▶ 다운로드 → $dest  (필요 여유 공간: 데이터셋 크기의 2~3배)"; df -h "$dest" | tail -1
    if [[ -n "$keys" ]]; then "$bin" -mode d -datasetkey "$ds" -filekey "$keys" -aihubapikey "$AIHUB_API_KEY"
    else "$bin" -mode d -datasetkey "$ds" -aihubapikey "$AIHUB_API_KEY"; fi
    echo "✓ 완료. 다음: experiments/verify_aihub_sample.py 로 실물 검증" ;;
  *) echo "알 수 없는 mode: $mode" >&2; exit 2 ;;
esac
