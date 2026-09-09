#!/usr/bin/env bash
# conda env 를 NFS(/soundai) 에서 로컬 tmpfs/디스크로 복사해 import 지연을 없앤다(2026-09-09 측정: NFS 에서 import torch 66 s, transformers 197 s, vapasr.hf 180 s).
#   컨테이너:  bash scripts/stage-env-local.sh            → /dev/shm/vapasr-env (256 GB tmpfs, 컨테이너가 살아있는 동안 유지)
#   이후:      source scripts/activate-env.sh              (VAPASR_LOCAL_ENV 가 있으면 그 python 을 PATH 앞에 둔다)
#   확인:      which python  → /dev/shm/vapasr-env/bin/python
#   pack:      bash scripts/stage-env-local.sh pack   → $MXC_CONDA_DIR/envs/vapasr.tar (한 번만; env 를 바꾼 뒤 다시)
#   sbatch:    슬럼 스크립트가 노드마다 srun 으로 이 스크립트를 돌린다(VAPASR_LOCAL_ENV=/tmp/sa_tskim/vapasr-env, 컴퓨트 노드 /tmp 는 로컬 md0 28 TB). 두 번째부터는 마커로 건너뛴다
# 이미 복사돼 있고(마커 파일의 원본 mtime 이 같으면) 건너뛴다. 원본은 읽기만 한다.
set -euo pipefail
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_pre_local="${VAPASR_LOCAL_ENV:-}"       # 호출자(sbatch: /tmp/sa_tskim/vapasr-env)가 정한 값이 .env 의 컨테이너용 값(/scratch)에 덮이지 않게(67293 에서 노드가 /scratch 를 만들려다 실패)
set -a; source "$_root/.env"; [ -f "$_root/.env.local" ] && source "$_root/.env.local"; set +a
[ -n "$_pre_local" ] && VAPASR_LOCAL_ENV="$_pre_local"
SRC="${MXC_CONDA_DIR:?}/envs/${CONDA_ENV_NAME:-vapasr}"; DST="${VAPASR_LOCAL_ENV:-/dev/shm/vapasr-env}"
[ -d "$SRC" ] || { echo "!! 원본 env 없음: $SRC"; exit 1; }
stamp=$(stat -c %Y "$SRC/conda-meta/history" 2>/dev/null || echo 0)
TAR="${VAPASR_ENV_TAR:-$SRC.tar}"        # scripts/stage-env-local.sh pack 으로 만든 단일 아카이브(NFS 순차 읽기 ~300 MB/s → 8.7 GB 에 1–2 분). 없으면 파일 단위 병렬 복사(NFS 왕복 지연 때문에 15–20 분)
if [ "${1:-}" = pack ]; then                # pack: 로컬 복사본을 tar 로 묶어 NFS 에 둔다(다른 노드·컨테이너가 이걸 푼다)
  [ -x "$DST/bin/python" ] || { echo "!! 먼저 복사본이 있어야 pack 할 수 있다: $DST"; exit 1; }
  echo "[$(date '+%T')] pack $DST → $TAR"; tar -C "$DST" -cf "$TAR.tmp" --exclude='./.staged-from' . && mv "$TAR.tmp" "$TAR" && echo "$SRC@$stamp" > "$TAR.stamp"; ls -la "$TAR"; exit 0
fi
if [ -x "$DST/bin/python" ] && [ "$(cat "$DST/.staged-from" 2>/dev/null)" = "$SRC@$stamp" ]; then echo "이미 복사됨: $DST (원본 $SRC)"; exit 0; fi
echo "[$(date '+%T')] $SRC → $DST 복사 시작 (df: $(df -h "$(dirname "$DST")" | awk 'NR==2{print $4" 남음"}'))"
mkdir -p "$DST"; t0=$(date +%s)
if [ -f "$TAR" ] && [ "$(cat "$TAR.stamp" 2>/dev/null)" = "$SRC@$stamp" ]; then
  echo "tar 아카이브 사용: $TAR ($(du -h "$TAR" | cut -f1))"; tar -C "$DST" -xf "$TAR"
else
  # NFS 는 파일당 왕복 지연이 커서 단일 tar 스트림은 0.1 MB/s 수준(2026-09-09 실측) → 디렉토리를 먼저 만들고 파일을 128 개 병렬로 복사
  ( cd "$SRC" && find . -type d -not -path './pkgs*' -print0 ) | ( cd "$DST" && xargs -0 -r mkdir -p )
  ( cd "$SRC" && find . \( -type f -o -type l \) -not -path './pkgs*' -print0 ) | ( cd "$SRC" && xargs -0 -r -P "${STAGE_PAR:-128}" -n 64 cp -a --parents -t "$DST" ) || true   # 드문 'File exists'(병렬 경합)는 무시
  n_src=$(cd "$SRC" && find . \( -type f -o -type l \) -not -path './pkgs*' | wc -l); n_dst=$(cd "$DST" && find . \( -type f -o -type l \) | wc -l)
  [ "$n_src" -eq "$n_dst" ] || { echo "!! 파일 수 불일치: 원본 $n_src, 복사본 $n_dst — 다시 실행하면 이어서 복사"; exit 1; }
fi
# 콘솔 스크립트(torchrun·pytest 등)의 shebang 이 원본 경로를 가리키면 NFS python 이 다시 뜬다 → 복사본 경로로 고친다
grep -lZ "^#!$SRC/bin/python" "$DST"/bin/* 2>/dev/null | xargs -0 -r sed -i "1s|^#!$SRC/bin/python|#!$DST/bin/python|"
echo "$SRC@$stamp" > "$DST/.staged-from"
echo "[$(date '+%T')] 완료: $(du -sh "$DST" | cut -f1), $(( $(date +%s) - t0 )) s → source scripts/activate-env.sh 로 사용"
