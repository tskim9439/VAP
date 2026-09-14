#!/bin/bash
# NOTSOFAR-1 (HF microsoft/NOTSOFAR, CC BY 4.0) 최신 3 서브셋을 하나씩 T5 에 받아 mxc /soundai/DB/raw/notsofar/<서브셋> 으로 올린다(검증 후 로컬 삭제).
# corpus-relay.sh 의 upload 서브커맨드를 재사용한다(실행 중인 릴레이 파일을 수정하지 않기 위해 별도 스크립트).
set -uo pipefail
R="$HOME/Desktop/VAPKT"; LOCAL=/Volumes/Samsung_T5/vapkt-corpora; LOG=$LOCAL/relay.log; CA=~/.vapkt-ca-bundle.pem
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
export REQUESTS_CA_BUNDLE=$CA SSL_CERT_FILE=$CA CURL_CA_BUNDLE=$CA; set -a; source "$R/.env.local"; set +a
[ -n "${HF_TOKEN:-}" ] || { log "!! HF_TOKEN 없음"; exit 1; }
log "=== NOTSOFAR-1 릴레이 시작 ==="
for sub in benchmark-datasets/dev_set/240825.1_dev1 benchmark-datasets/train_set/240825.1_train benchmark-datasets/eval_set/240825.1_eval_full_with_GT; do
  name=$(basename "$sub"); d="$LOCAL/notsofar-$name"; mkdir -p "$d"
  if /Users/taesookim/anaconda3/envs/vapasr-local/bin/python "$R/scripts/notsofar_download.py" "$sub" "$d" >> "$LOG" 2>&1; then
    if "$R/scripts/corpus-relay.sh" upload "$d" "/soundai/DB/raw/notsofar/$name" >> "$LOG" 2>&1; then log "notsofar-$name 업로드 검증 완료 → 로컬 삭제"; rm -rf "$d"; echo "notsofar-$name" >> "$LOCAL/relayed.txt"; else log "!! notsofar-$name 업로드 검증 실패 — 로컬 보존"; fi
  else log "!! notsofar-$name 다운로드 실패"; fi
done
log "=== NOTSOFAR-1 릴레이 종료 ==="
