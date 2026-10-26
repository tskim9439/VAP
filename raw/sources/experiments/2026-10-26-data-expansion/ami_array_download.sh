#!/usr/bin/env bash
# AMI Array1 원거리 8 채널 내려받기(맥 → T5, 2026-10-26, 사용자 "모두 진행하는 데, 모두 진행하고 나서 E5 E6 학습 진행하자"):
#   공식 Full-corpus-ASR train(SA) 136 회의 + IB4005 중 미러에 Array1 이 있는 135 회의(ami-array1-train-meetings.txt — IS1003b·IS1007d 는 Array1 없음).
#   dev(SB)·eval(SC) 회의는 받지 않는다. 업로드는 사용자가 한다(데이터 업로드는 사용자만).
#   bash ami_array_download.sh <회의 목록> <받을 폴더> [동시 수(기본 4)]
#   채널 순서로 받는다(먼저 모든 회의의 Array1-01 = 평가 SDM 과 같은 마이크, 그다음 02–08). 크기가 Content-Length 와 같은 파일은 건너뛰고,
#   끊긴 파일은 이어 받는다. 진행 로그 <받을 폴더>/progress.log(OK·FAIL 줄, 끝에 ALL_DONE). 아무것도 지우지 않는다.
set -u
LIST=$1; OUT=$2; P=${3:-4}; U=https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus
mkdir -p "$OUT"; LOG=$OUT/progress.log
one() {
  local m=$1 c=$2 f want have try
  f=$OUT/$m/$m.Array1-0$c.wav; mkdir -p "$OUT/$m"
  for try in 1 2 3; do
    want=$(curl -sSI --max-time 90 "$U/$m/audio/$m.Array1-0$c.wav" 2>/dev/null | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
    [ -n "$want" ] && break; sleep 20
  done
  [ -n "$want" ] || { echo "FAIL $m $c 크기를 못 읽음" >> "$LOG"; return; }
  for try in 1 2 3 4 5 6; do
    have=$(stat -f %z "$f" 2>/dev/null || echo 0)
    [ "$have" = "$want" ] && { [ "$(head -c 4 "$f")" = RIFF ] && echo "OK $m $c $want $(date '+%F %T')" >> "$LOG" || echo "FAIL $m $c 머리가 RIFF 가 아님(오류 페이지가 섞였다 — 지우고 다시 받는다)" >> "$LOG"; return; }
    curl -sS --fail -C - --max-time 7200 --retry 3 --retry-delay 30 -o "$f" "$U/$m/audio/$m.Array1-0$c.wav" 2>>"$OUT/curl.err" || sleep 30   # --fail: HTTP 오류 본문(HTML)을 파일에 쓰지 않는다(10-26 ES2012d-02 앞 331 B 사고)
  done
  echo "FAIL $m $c $(stat -f %z "$f" 2>/dev/null)/$want" >> "$LOG"
}
export -f one; export OUT LOG U
echo "START $(date '+%F %T') 회의 $(wc -l < "$LIST") × 8 채널 · 동시 $P" >> "$LOG"
for c in 1 2 3 4 5 6 7 8; do while read -r m; do echo "$m $c"; done < "$LIST"; done | xargs -P "$P" -n 2 bash -c 'one "$0" "$1"'
echo "ALL_DONE $(date '+%F %T') · OK $(grep -c '^OK' "$LOG") · FAIL $(grep -c '^FAIL' "$LOG")" >> "$LOG"
