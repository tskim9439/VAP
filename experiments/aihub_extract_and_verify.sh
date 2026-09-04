#!/usr/bin/env bash
# VS_02.실외.zip 이 도착하면 해제하고 실물 검증을 돌린다. 컨테이너 안에서 실행.
#   bash experiments/aihub_extract_and_verify.sh [zip 이름 접두사=VS_02] [--n 100]
set -uo pipefail
root="/data3/tskim/corpora/aihub/71631/134-1.감정이_태깅된_자유대화_(성인)/01-1.정식개방데이터"
pfx="${1:-VS_02}"; n="${3:-100}"
log(){ printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$*"; }

log "aihubshell 종료 대기 (호스트 프로세스 — 컨테이너에서는 zip 완성 여부로 판단)"
zip=""
for i in $(seq 1 360); do   # 최대 1시간
  zip=$(find "$root" -name "${pfx}*.zip" ! -name "*.part*" 2>/dev/null | head -1)
  if [[ -n "$zip" ]] && ! find "$root" -name "${pfx}*.zip.part*" | grep -q .; then
    s1=$(stat -c %s "$zip"); sleep 20; s2=$(stat -c %s "$zip")
    [[ "$s1" == "$s2" ]] && break     # 20 s 동안 크기 변화 없음 = 병합 완료
  fi
  sleep 10
done
[[ -n "$zip" ]] || { echo "zip 을 찾지 못했습니다"; exit 1; }
log "zip: $zip ($(du -h "$zip" | cut -f1))"
unzip -tq "$zip" >/dev/null 2>&1 && echo "zip 무결성 OK" || { echo "zip 무결성 실패 (아직 다운로드 중?)"; exit 1; }

d="${zip%.zip}"; mkdir -p "$d"
log "unzip → $d"
unzip -q -o "$zip" -d "$d" 2>/dev/null; echo "wav: $(find "$d" -name '*.wav' | wc -l)"
echo "--- 샘플 헤더 (soxi/ffprobe 대용: python soundfile) ---"
python - "$d" <<'PY'
import sys, glob, soundfile as sf, collections
ws = glob.glob(sys.argv[1] + "/**/*.wav", recursive=True)
c = collections.Counter()
for w in ws[:200]:
    i = sf.info(w); c[(i.samplerate, i.channels, i.subtype)] += 1
print("헤더 분포 (samplerate, channels, subtype) — 앞 200개:", dict(c))
print("첫 파일:", ws[0] if ws else None)
PY

log "실물 검증 (experiments/verify_aihub_sample.py, n=$n)"
python experiments/verify_aihub_sample.py "$root" --n "$n"
log "완료"
