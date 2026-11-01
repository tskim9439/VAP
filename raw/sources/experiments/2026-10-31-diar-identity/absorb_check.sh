#!/usr/bin/env bash
# N3D 흡수 확인(2026-11-01): ① E11 final → 흡수판 변환·재로드 대조(experiments/n3d_absorb.py, CPU, 산출 tmp/e11-check/e11-absorb/final)
# ② tests/test_n3d.py(실제 .nemo, CPU). 아무것도 지우지 않는다.
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e11-check; P=$D/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES= VAPASR_STAGE_DIR=
mkdir -p $T/e11-absorb
$P experiments/n3d_absorb.py --ckpt /soundai/Model/VAPASR/stage3-q17-e11/final --out $T/e11-absorb/final --nemotron-dir $MXC_NEMOTRON_DIR > $T/e11-absorb.log 2>&1
echo "ABSORB_DONE exit $? $(date +%T)" >> $T/absorb-done.txt
VAPASR_N3D_PATH=$D/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo $P -m pytest -q -p no:cacheprovider tests/test_n3d.py > $T/absorb-tests.log 2>&1
echo "ABSORB_TESTS_DONE exit $? $(date +%T)" >> $T/absorb-done.txt
