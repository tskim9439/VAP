#!/usr/bin/env bash
# 2026-11-01 스모크: ① E11 지연 완화(N3D 청크 124 + 인코더 우측 문맥 13) — AMI SDM 창 8 개, 회의 1 개(--verify)
# ② Parakeet + 저장 N3D 확률(--diar-probs, 청크 4) — AMI SDM 회의 2 개(직접 돌린 ext-ami-mtp-n3d 의 같은 회의와 견준다). bash relax_smoke.sh <GPU>. 아무것도 지우지 않는다.
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e11-check; P=$D/conda/envs/vapasr/bin/python; CK=/soundai/Model/VAPASR/stage3-q17-e11/final
N3=$D/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo; MTP=$D/vapkt-baselines/multitalker-parakeet-v1/multitalker-parakeet-streaming-0.6b-v1.nemo
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage CUDA_VISIBLE_DEVICES=$1
$P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $D/data/ami-bench-v1 --corpora ami-sdm --limit-windows 8 --deltas 10 --batch-size 8 \
  --backchannel 1.0,3 --max-speakers 4 --window 20,90 --window-draws 1 --declare 4 --n3d $N3 --n3d-chunk 124 --encoder-rc 13 --out $T/relax-smoke-win > $T/relax-smoke-win.log 2>&1
$P experiments/cst_stage3_eval.py summarize --out $T/relax-smoke-win >> $T/relax-smoke-win.log 2>&1
$P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $D/data/ami-bench-v1 --corpora ami-sdm --limit-windows 1 --deltas 10 \
  --backchannel 1.0,3 --max-speakers 4 --window full --context-s 30 --declare 4 --verify --n3d $N3 --n3d-chunk 124 --encoder-rc 13 --out $T/relax-smoke-full > $T/relax-smoke-full.log 2>&1
$P experiments/cst_stage3_eval.py summarize --out $T/relax-smoke-full >> $T/relax-smoke-full.log 2>&1
$P experiments/baseline_ami_ext.py run --system parakeet --asr $MTP --diar $N3 --diar-kind n3d --diar-probs $T/diar-probs/n3d/n3d/probs --checkpoint $CK --data $D/data/ami-bench-v1 \
  --corpora ami-sdm --limit-windows 2 --deltas 10 --batch-size 2 --wav-dir /tmp/sa_tskim/ami-ext-wav-replay-smoke --out $T/mtp-replay-smoke > $T/mtp-replay-smoke.log 2>&1
echo "RELAX_SMOKE_DONE $(date +%T)" >> $T/relax-smoke-done.txt
