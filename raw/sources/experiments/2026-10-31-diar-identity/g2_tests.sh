#!/usr/bin/env bash
# G2 시험(2026-10-31): 서버 CPU 에서 tests/test_n3d_graft.py(실제 .nemo). 산출 tmp/e11-check/g2-tests.log.
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e11-check
CUDA_VISIBLE_DEVICES= VAPASR_N3D_PATH=$D/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo PYTHONPATH=/soundai/users/tskim/VAPKT \
  $D/conda/envs/vapasr/bin/python -m pytest -q tests/test_n3d_graft.py > $T/g2-tests.log 2>&1
echo "G2_TESTS_DONE exit $? $(date +%T)" >> $T/g2-smoke-done.txt
