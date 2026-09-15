#!/bin/bash
# Phase 2 Q0 — 코퍼스 → dialogues.jsonl (CPU). 정본 output-phase2-lane-plan §8, task-phase2-data-prep 순서 2–3.
#   제출:  sbatch --partition=apex --export=ALL,CORPUS=ami slurm/p2_build_dialogues.sbatch
#          CORPUS ∈ aihub71631 | aihub134-1 | aihub134-2 | otoSpeech | ami | notsofar | icsi   (전부: CORPUS=all)
#   출력:  $OUT(기본 /soundai/users/tskim/VAPKT-data/data/phase2)/<corpus>.dialogues.jsonl + .stats.json
#   ICSI 는 sph2pipe 변환본(VAPKT-data/data/audio/icsi, 75회의 FLAC)을 root 로 쓴다. 134-2 라벨은 NIA24 Json/TL_02.실외 해제본.
#   입력은 읽기만 한다(/soundai/DB, databricks 사본, VAPKT-data/data/labels).
#SBATCH --job-name=SA_SFT_FullDuplexStage5-p2build
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/phase2}; mkdir -p "$OUT"
NIA=/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/asr_db/korean_16kHz/NIA24
LAB=/soundai/users/tskim/VAPKT-data/data/labels/aihub71631
find_dir() { python - "$1" "$2" <<'EOF'
import os,sys,unicodedata as u; p,n=sys.argv[1:3]
for d in os.listdir(p):
    if u.normalize("NFC",d)==u.normalize("NFC",n): print(os.path.join(p,d)); break
EOF
}
run() {
  local c=$1; echo "[$(date '+%F %T')] build $c"
  case "$c" in
    aihub71631) python experiments/p2_build_dialogues.py --corpus aihub71631 --audio-root "/soundai/DB/raw/aihub/71631_audio/Training" --label-root "$(find_dir "$LAB" TL_01.실내)" --out "$OUT" ;;
    aihub134-1) python experiments/p2_build_dialogues.py --corpus aihub134-1 --crop-dir "$(find_dir "$NIA/134-1_Emotion_Conv_Adult/Training" TS_02.실외)" --label-root "$(find_dir "$LAB" TL_02.실외)" --out "$OUT" ;;
    aihub134-2) python experiments/p2_build_dialogues.py --corpus aihub134-2 --crop-dir "$(find_dir "$NIA/134-2_Emotion_Conv_Youth/Training" TS_02.실외)" --label-root "${YOUTH_LABELS:-$(find_dir "$NIA/134-2_Emotion_Conv_Youth/Json" TL_02.실외)}" --out "$OUT" ;;
    otoSpeech)  python experiments/p2_build_dialogues.py --corpus otoSpeech --root /soundai/DB/raw/otoSpeech16k --out "$OUT" ;;
    ami)        python experiments/p2_build_dialogues.py --corpus ami --root /soundai/DB/raw/ami --annotations /soundai/DB/raw/ami/annotations/ami_public_manual_1.6.2.zip --out "$OUT" ;;
    notsofar)   python experiments/p2_build_dialogues.py --corpus notsofar --root /soundai/DB/raw/notsofar --out "$OUT" ;;
    icsi)       python experiments/p2_build_dialogues.py --corpus icsi --root /soundai/users/tskim/VAPKT-data/data/audio/icsi --annotations /soundai/DB/raw/icsi/annotations/ICSI_core_NXT.zip --transcripts /soundai/DB/raw/icsi/annotations/ICSI_original_transcripts.zip --out "$OUT" ;;
    *) echo "unknown corpus $c"; return 1 ;;
  esac
}
if [ "${CORPUS:?CORPUS=ami 등}" = all ]; then for c in otoSpeech ami notsofar icsi aihub134-1 aihub134-2 aihub71631; do run "$c"; done; else run "$CORPUS"; fi
echo "[$(date '+%F %T')] p2 build done"; echo EXIT=0
