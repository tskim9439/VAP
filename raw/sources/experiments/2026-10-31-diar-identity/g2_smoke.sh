#!/usr/bin/env bash
# G2 스모크(2026-10-31): bash g2_smoke.sh <GPU 쉼표> <이름> [추가 인자] — U3 final 에서 학생 N3D 위층 LoRA, AMI·ICSI 60–90 s 창, 배치 2, 4 step(GPU 둘이면 torchrun 2 rank 3 step)
# + 저장·재로드 대조. 산출 tmp/e11-check/<이름>(새 폴더). 아무것도 지우지 않는다.
set -u
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e11-check; P=$D/conda/envs/vapasr/bin/python
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage CUDA_VISIBLE_DEVICES=$1
O=$T/$2; shift 2; IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
ARGS=(--init /soundai/Model/VAPASR/stage3-q17-e9/final --n3d $D/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo --nemotron-dir $MXC_NEMOTRON_DIR
  --out-dir $O --corpora ami,icsi --window 60,90 --batch 2 --num-workers 2 --noise-bank $D/data/noise-bank-v1
  --init-graft /soundai/Model/VAPASR/graft-q17-graft-u3-l12/final/graft.pt --stream-chunk 16 --hid-weight 0.1 --ref-weight 0.5
  --unfreeze-layers 12 --enc-lr 2e-5 --lr 5e-5 --guard-weight 10 --n3d-train lora --n3d-lr 1e-4 --lora-rank 32 --lora-alpha 32 --log-every 1 "$@")
if [ ${#G[@]} -gt 1 ]; then $P -m torch.distributed.run --nproc_per_node ${#G[@]} --master_port 29631 experiments/n3d_graft_train.py "${ARGS[@]}" --smoke 3 > $O.log 2>&1
else $P experiments/n3d_graft_train.py "${ARGS[@]}" --smoke 4 > $O.log 2>&1; fi
echo "G2_SMOKE_DONE $(basename $O) exit $? $(date +%T)" >> $T/g2-smoke-done.txt
