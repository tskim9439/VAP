#!/bin/bash
# 회의 코퍼스 확보 릴레이 — 로컬(맥)에 내려받은 코퍼스를 mxc 로 올리고(rsync, 검증 후 로컬 삭제), 디스크를 비운 뒤 다음 코퍼스를 받는다.
# 규칙: 서버에서 직접 받지 않는다 · 서버 파일은 절대 지우지 않는다 · 로컬 여유 54 GB 라 한 번에 한 코퍼스.
#   scripts/corpus-relay.sh run         # 전체 순서 실행(백그라운드 권장):  DiPCo·AMI 완료 대기→업로드 → ICSI 받기→업로드 → CHiME-6 (chunk 릴레이)
#   scripts/corpus-relay.sh upload <로컬 dir> <서버 dir>   # 한 코퍼스만 업로드+검증(삭제 없음)
set -uo pipefail
LOCAL=~/Downloads/vapkt-corpora; REMOTE=/soundai/DB/raw; SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -p 3206 mxc"
CA=~/.vapkt-ca-bundle.pem; LOG=$LOCAL/relay.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_retry(){ local i; for i in 1 2 3 4 5 6; do $SSH "$@" && return 0; log "ssh 실패, 60 s 후 재시도 ($i)"; sleep 60; done; return 1; }

upload(){ # upload <local dir> <remote dir>  — rsync 이어받기, 바이트 합계 검증
  local src="$1" dst="$2" i
  ssh_retry "mkdir -p '$dst'" || return 1
  for i in 1 2 3 4 5 6 7 8; do
    rsync -a --partial --exclude '*.part' --exclude 'run.log' -e "$SSH" "$src/" "mxc:$dst/" && break
    log "rsync 중단, 120 s 후 재개 ($i)"; sleep 120
  done
  local lb rb; lb=$(find "$src" -type f ! -name '*.part' ! -name run.log -exec stat -f %z {} + | awk '{s+=$1} END{print s+0}')
  rb=$(ssh_retry "find '$dst' -type f -exec stat -c %s {} + | awk '{s+=\$1} END{print s+0}'")
  log "검증 $src → $dst : local=$lb remote=$rb"; [ "$lb" = "$rb" ] && [ "$lb" -gt 0 ]
}
finish(){ # finish <name> <local dir> <remote dir>: 업로드+검증 성공 시에만 로컬 삭제
  local name="$1" src="$2" dst="$3"
  if upload "$src" "$dst"; then log "$name 업로드 검증 완료 → 로컬 삭제"; rm -rf "$src"; echo "$name" >> "$LOCAL/relayed.txt"; else log "!! $name 업로드 검증 실패 — 로컬 보존"; return 1; fi
}
wait_file(){ until [ -f "$1" ]; do sleep 120; done; }

chunk_relay(){ # chunk_relay <url> <remote file> <chunk GB>: 로컬 디스크를 넘는 파일을 범위 다운로드 → 업로드 → 삭제로 릴레이, 서버에서 cat 으로 합침
  local url="$1" rfile="$2" gb="${3:-10}" tmp="$LOCAL/_chunk" total off=0 n=0 part size
  mkdir -p "$tmp"; total=$(curl -sIL --cacert $CA "$url" | grep -i '^content-length' | tail -n 1 | tr -dc '0-9'); [ -n "$total" ] || { log "!! 크기 확인 실패 $url"; return 1; }
  ssh_retry "mkdir -p '$(dirname "$rfile")/_parts_$(basename "$rfile")'"; local rparts="$(dirname "$rfile")/_parts_$(basename "$rfile")"
  size=$((gb*1024*1024*1024)); log "chunk_relay $url total=$total chunk=$size"
  while [ "$off" -lt "$total" ]; do
    part=$(printf "part-%04d" $n); local end=$((off+size-1)); [ "$end" -ge "$total" ] && end=$((total-1))
    if ssh_retry "[ -f '$rparts/$part' ] && [ \$(stat -c %s '$rparts/$part') -eq $((end-off+1)) ]"; then log "$part 서버에 이미 있음"; off=$((end+1)); n=$((n+1)); continue; fi
    local i; for i in 1 2 3 4 5; do curl -sS -L --cacert $CA -r "$off-$end" -o "$tmp/$part" "$url" && [ "$(stat -f %z "$tmp/$part")" -eq $((end-off+1)) ] && break; log "$part 다운로드 재시도 ($i)"; sleep 60; done
    [ "$(stat -f %z "$tmp/$part" 2>/dev/null || echo 0)" -eq $((end-off+1)) ] || { log "!! $part 실패"; return 1; }
    for i in 1 2 3 4 5; do rsync -a --partial -e "$SSH" "$tmp/$part" "mxc:$rparts/" && break; sleep 60; done
    if ssh_retry "[ \$(stat -c %s '$rparts/$part') -eq $((end-off+1)) ]"; then rm -f "$tmp/$part"; log "$part 완료 ($((end+1))/$total)"; else log "!! $part 서버 검증 실패"; return 1; fi
    off=$((end+1)); n=$((n+1))
  done
  ssh_retry "cd '$rparts' && cat \$(ls part-* | sort) > '$rfile' && [ \$(stat -c %s '$rfile') -eq $total ] && echo MERGED" | grep -q MERGED && log "합침 완료 $rfile" || { log "!! 합침 실패 $rfile"; return 1; }
}

case "${1:-}" in
  upload) upload "$2" "$3" ;;
  run)
    log "=== 릴레이 시작 ==="
    # 1) DiPCo: curl 종료(curl.log 에 exit=) 대기 → 업로드
    until grep -q '^exit=' "$LOCAL/dipco/curl.log" 2>/dev/null; do sleep 300; done
    if grep -q '^exit=0' "$LOCAL/dipco/curl.log"; then finish dipco "$LOCAL/dipco" "$REMOTE/dipco"; else log "!! DiPCo curl 실패: $(tail -n 1 $LOCAL/dipco/curl.log)"; fi
    # 2) AMI: ALL_DONE 대기 → 업로드
    until grep -q '^ALL_DONE' "$LOCAL/ami/progress.log" 2>/dev/null; do sleep 300; done
    finish ami "$LOCAL/ami" "$REMOTE/ami"
    # 3) ICSI: 받기 → 업로드
    (cd "$LOCAL/icsi" && ./download.sh > run.log 2>&1); finish icsi "$LOCAL/icsi" "$REMOTE/icsi"
    # 3b) NOTSOFAR-1 (HF microsoft/NOTSOFAR, CC BY 4.0, 근접 마이크 포함): 최신 버전 3 서브셋을 하나씩 받기→업로드→삭제 (각 15–49 GB)
    export REQUESTS_CA_BUNDLE=$CA SSL_CERT_FILE=$CA CURL_CA_BUNDLE=$CA; set -a; source "$HOME/Desktop/VAPKT/.env.local"; set +a
    for sub in benchmark-datasets/dev_set/240825.1_dev1 benchmark-datasets/train_set/240825.1_train benchmark-datasets/eval_set/240825.1_eval_full_with_GT; do
      name=$(basename "$sub"); mkdir -p "$LOCAL/notsofar-$name"
      /Users/taesookim/anaconda3/envs/vapasr-local/bin/python "$HOME/Desktop/VAPKT/scripts/notsofar_download.py" "$sub" "$LOCAL/notsofar-$name" >> "$LOG" 2>&1
      finish "notsofar-$name" "$LOCAL/notsofar-$name" "$REMOTE/notsofar/$name"
    done
    # 4) CHiME-6 (OpenSLR 150, CC BY-SA 4.0): 작은 것은 통째로, train(97 GB)은 chunk 릴레이
    mkdir -p "$LOCAL/chime6"; B=https://www.openslr.org/resources/150
    for f in LICENSE.txt CHiME6_transcriptions.tar.gz CHiME6_floorplans.tar.gz CHiME6_dev.tar.gz CHiME6_eval.tar.gz; do curl -sS -L --cacert $CA -C - -o "$LOCAL/chime6/$f" "$B/$f" || log "!! $f 실패"; done
    finish chime6-small "$LOCAL/chime6" "$REMOTE/chime6"
    chunk_relay "$B/CHiME6_train.tar.gz" "$REMOTE/chime6/CHiME6_train.tar.gz" 10
    log "=== 릴레이 종료 (NOTSOFAR-1 은 HF 토큰 필요, 별도) ===" ;;
  *) sed -n '2,6p' "$0"; exit 2 ;;
esac
