#!/usr/bin/env bash
# conda env 를 NFS(/soundai) 에서 로컬 tmpfs/디스크로 복사해 import 지연을 없앤다(2026-09-09 측정: NFS 에서 import torch 66 s, transformers 197 s, vapasr.hf 180 s).
#   컨테이너:  bash scripts/stage-env-local.sh            → /dev/shm/vapasr-env (256 GB tmpfs, 컨테이너가 살아있는 동안 유지)
#   이후:      source scripts/activate-env.sh              (VAPASR_LOCAL_ENV 가 있으면 그 python 을 PATH 앞에 둔다)
#   확인:      which python  → /dev/shm/vapasr-env/bin/python
#   sbatch:    VAPASR_LOCAL_ENV=/dev/shm/vapasr-env bash scripts/stage-env-local.sh 를 노드마다(srun) 실행한 뒤 학습 시작(노드 로컬 디스크가 있으면 그쪽이 낫다)
# 이미 복사돼 있고(마커 파일의 원본 mtime 이 같으면) 건너뛴다. 원본은 읽기만 한다.
set -euo pipefail
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; source "$_root/.env"; [ -f "$_root/.env.local" ] && source "$_root/.env.local"; set +a
SRC="${MXC_CONDA_DIR:?}/envs/${CONDA_ENV_NAME:-vapasr}"; DST="${VAPASR_LOCAL_ENV:-/dev/shm/vapasr-env}"
[ -d "$SRC" ] || { echo "!! 원본 env 없음: $SRC"; exit 1; }
stamp=$(stat -c %Y "$SRC/conda-meta/history" 2>/dev/null || echo 0)
if [ -x "$DST/bin/python" ] && [ "$(cat "$DST/.staged-from" 2>/dev/null)" = "$SRC@$stamp" ]; then echo "이미 복사됨: $DST (원본 $SRC)"; exit 0; fi
echo "[$(date '+%T')] $SRC → $DST 복사 시작 (df: $(df -h "$(dirname "$DST")" | awk 'NR==2{print $4" 남음"}'))"
mkdir -p "$DST"; t0=$(date +%s)
# NFS 는 파일당 왕복 지연이 커서 단일 tar 스트림은 0.1 MB/s 수준(2026-09-09 실측) → 디렉토리를 먼저 만들고 파일을 128 개 병렬로 복사
( cd "$SRC" && find . -type d -not -path './pkgs*' -print0 ) | ( cd "$DST" && xargs -0 -r mkdir -p )
( cd "$SRC" && find . \( -type f -o -type l \) -not -path './pkgs*' -print0 ) | ( cd "$SRC" && xargs -0 -r -P "${STAGE_PAR:-128}" -n 64 cp -a --parents -t "$DST" )
# 콘솔 스크립트(torchrun·pytest 등)의 shebang 이 원본 경로를 가리키면 NFS python 이 다시 뜬다 → 복사본 경로로 고친다
grep -lZ "^#!$SRC/bin/python" "$DST"/bin/* 2>/dev/null | xargs -0 -r sed -i "1s|^#!$SRC/bin/python|#!$DST/bin/python|"
echo "$SRC@$stamp" > "$DST/.staged-from"
echo "[$(date '+%T')] 완료: $(du -sh "$DST" | cut -f1), $(( $(date +%s) - t0 )) s → source scripts/activate-env.sh 로 사용"
