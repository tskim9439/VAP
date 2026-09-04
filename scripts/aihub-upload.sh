#!/usr/bin/env bash
# AI Hub 에서 PC(맥) 브라우저로 내려받은 파일을 rack4 의 코퍼스 폴더로 올린다.
# (AI Hub 는 이 계정/데이터셋에 API 키·aihubshell 을 제공하지 않아 PC 다운로드만 가능 — 2026-09-03 확인)
#
#   scripts/aihub-upload.sh <adult|teen|datasetkey> <로컬 파일 또는 폴더>... [--move] [--bwlimit KB/s]
#
#   --move    전송·검증 후 로컬 원본 삭제 (맥 디스크 여유가 적을 때 "받고→올리고→지우기" 순환용)
#   재개 가능: 중단되면 같은 명령을 다시 실행 (rsync --partial)
#   업로드 후 서버에서:  ssh rack4 'cd /home/tskim/VAP && scripts/aihub-upload.sh extract adult'
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
set -a; source .env; [ -f .env.local ] && source .env.local; set +a

ds="${1:-}"; shift || true
case "$ds" in adult) ds="$AIHUB_DATASET_ADULT";; teen) ds="$AIHUB_DATASET_TEEN";; extract) ;; esac
[[ -n "$ds" ]] || { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
dest="${AIHUB_DOWNLOAD_DIR:-/data3/tskim/corpora/aihub}"

# ── 서버 측: 압축 해제 ───────────────────────────────────────────────────────
if [[ "$ds" == "extract" ]]; then
  key="${1:-}"; case "$key" in adult) key="$AIHUB_DATASET_ADULT";; teen) key="$AIHUB_DATASET_TEEN";; esac
  [[ -n "$key" ]] || { echo "extract <adult|teen|datasetkey>" >&2; exit 2; }
  cd "$dest/$key"
  echo "▶ $PWD"; ls -la | head -30
  # AI Hub 분할 압축: xxx.zip.part0, .part1 ... → cat 으로 합친 뒤 unzip
  for base in $(ls | grep -E '\.zip\.part[0-9]+$' | sed 's/\.part[0-9]*$//' | sort -u); do
    if [[ ! -f "$base" ]]; then echo "▶ 분할 병합: $base"; cat $(ls "$base".part* | sort -V) > "$base"; fi
  done
  for z in *.zip; do [[ -f "$z" ]] || continue; d="${z%.zip}"; mkdir -p "$d"
    echo "▶ unzip $z → $d/"; unzip -q -o "$z" -d "$d" && echo "  ok"; done
  echo "▶ 결과"; find . -maxdepth 3 -type d | head -20; echo "wav: $(find . -name '*.wav' | wc -l)  json: $(find . -name '*.json' | wc -l)"
  exit 0
fi

# ── 맥 측: 업로드 ───────────────────────────────────────────────────────────
move=0; bw=""; files=()
for a in "$@"; do case "$a" in --move) move=1;; --bwlimit=*) bw="${a#--bwlimit=}";; --bwlimit) :;; *) files+=("$a");; esac; done
[[ ${#files[@]} -gt 0 ]] || { echo "올릴 파일/폴더를 지정하세요." >&2; exit 2; }
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=30"
$SSH "$REMOTE_HOST" "mkdir -p '$dest/$ds'"
opts=(-a --partial --progress); [[ -n "$bw" ]] && opts+=(--bwlimit="$bw")
for f in "${files[@]}"; do
  [[ -e "$f" ]] || { echo "없음: $f" >&2; continue; }
  echo "▶ $f → $REMOTE_HOST:$dest/$ds/   ($(du -sh "$f" | cut -f1))"
  rsync "${opts[@]}" -e "$SSH" "$f" "$REMOTE_HOST:$dest/$ds/"
  if [[ $move -eq 1 ]]; then
    # 크기 재확인 후 삭제
    lsz=$(du -sk "$f" | cut -f1); rsz=$($SSH "$REMOTE_HOST" "du -sk '$dest/$ds/$(basename "$f")' | cut -f1")
    if [[ "$rsz" -ge $((lsz * 98 / 100)) ]]; then rm -rf "$f"; echo "  ✓ 전송 확인, 로컬 삭제"; else echo "  ! 크기 불일치 (local $lsz KB vs remote $rsz KB) — 삭제 안 함" >&2; fi
  fi
done
echo "완료. 서버에서 압축 해제:  ssh $REMOTE_HOST 'cd $REMOTE_PROJECT_DIR && scripts/aihub-upload.sh extract $ds'"
