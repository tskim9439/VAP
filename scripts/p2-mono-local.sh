#!/usr/bin/env bash
# Phase 2 mono 혼합 캐시를 SLURM 없이 mxc 로그인 노드에서 미리 생성 (CPU 작업; 사용자 지시 2026-09-16).
#   nohup bash scripts/p2-mono-local.sh > logs/train/p2mono-local.log 2>&1 &
#   옵션: PROCS(기본 16), CORPORA(기본 7 개 전부), OUT(기본 VAPKT-data/data/phase2)
set -uo pipefail
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/phase2}
echo "[$(date '+%F %T')] mono cache → $OUT/mono procs=${PROCS:-16} host=$(hostname)"
python experiments/p2_build_mono.py --data "$OUT" --mono "$OUT/mono" --procs "${PROCS:-16}" ${CORPORA:+--corpora "$CORPORA"}
echo "[$(date '+%F %T')] 종료"; du -sh "$OUT/mono"/* 2>/dev/null
