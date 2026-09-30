#!/bin/bash
# speechlm 파트 words 만 병렬로 만들기(라벨링 없음, CPU) — task 마다 slurm/semcommit-words-only-worker.sh(파트를 rank 로 나눈다, 파트 잠금 공유).
# 제출(사용자): sbatch [--partition=P] [--ntasks=N] slurm/semcommit-words-only.sbatch
#   입력은 워커 기본값(라벨링 v0.3.5 와 같은 RESULTS·QC_SPLIT·APPROVAL_SUMMARY·OUT, 순서 mixed-20260928b.tsv, MAIN_ONLY=1)이며 --export 로 바꿀 수 있다.
#   끝나면 experiments/semcommit_asr_words_list.py 로 SEM 중립 ASR 목록을 만든다(라벨이 나중에 생기면 스냅숏이 그 파트를 라벨과 함께 가져간다).
#SBATCH --job-name=SA_SFT_FullDuplex
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
mkdir -p /soundai/users/tskim/VAPKT/logs/train
echo "[$(date '+%F %T')] words-only job $SLURM_JOB_ID ntasks=$SLURM_NTASKS nodes=$SLURM_JOB_NODELIST"
srun --export=ALL --ntasks="$SLURM_NTASKS" --cpus-per-task="${SLURM_CPUS_PER_TASK:-2}" /bin/bash /soundai/users/tskim/VAPKT/slurm/semcommit-words-only-worker.sh
echo "[$(date '+%F %T')] words-only done rc=$?"
