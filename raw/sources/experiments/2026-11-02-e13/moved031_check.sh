#!/usr/bin/env bash
# 2026-11-02 AI Hub 031 폴더 이름 변경 점검(서버 호스트 CPU, 읽기만 — 공용 DB 는 건드리지 않는다):
# 경로 찾기 단위 시험(Linux 는 NFC·NFD 가 실제로 갈린다) + manifest 의 031 압축마다 항목 하나를 학습 로더 경로로 읽기. 산출 tmp/e13-031/check.log
#   bash moved031_check.sh [manifest glob ...]   (기본 asr-words-v1 의 aihub-bc parts)
set -u
cd /soundai/users/tskim/VAPKT
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e13-031; P=$D/conda/envs/vapasr/bin/python; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
echo "[$(date '+%F %T')] 시작" > $T/check.log
$P -m pytest -q -p no:cacheprovider tests/test_moved_paths.py >> $T/check.log 2>&1
$P raw/sources/experiments/2026-11-02-e13/moved031_check.py "${@:-$D/data/semcommit-work/asr-words-v1/parts/asr-aihub-bc-*/words.jsonl}" >> $T/check.log 2>&1
echo "[$(date '+%F %T')] CHECK_EXIT $?" >> $T/check.log
