# D4·D7·D8 새 코퍼스 대화 빌드(mxc 호스트, 2026-10-26): bash newcorp_build.sh <swbd|chime6|nia23mt> [p2_build_dialogues 인자 덧붙임]
#   준비 폴더 VAPKT-data/data/phase2x 에 <corpus>.dialogues.jsonl·.stats.json 을 만든다(phase2 의 기존 파일 — 예 CHiME-6 dev 평가용 chime6.dialogues.jsonl — 을 건드리지 않게).
#   그 뒤 정렬(p2_align, GPU — 사용자 SLURM)·정제(p2_refine) → phase2 로 refined 를 복사하고 mono 캐시를 만든다. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; X=/soundai/users/tskim/VAPKT-data/data/phase2x; mkdir -p $X
DB=/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f; C6=/soundai/users/tskim/VAPKT-data/data/audio/chime6
echo "[$(date "+%F %T")] newcorp_build $*"
case $1 in
  swbd)   $P experiments/p2_build_dialogues.py --corpus swbd --root $DB/EN/TRAIN/OPEN --transcripts $DB/EN/TRAIN/OPEN/switchboard/switchboard_train.csv --out $X --split train "${@:2}" ;;
  chime6) $P experiments/p2_build_dialogues.py --corpus chime6 --root $C6/CHiME6_train/CHiME6/audio/train --transcripts $C6/transcriptions/transcriptions/train --out $X --split train "${@:2}" ;;
  nia23mt) $P experiments/p2_build_dialogues.py --corpus nia23mt --root $DB/asr_db/korean_16kHz/NIA23/002_Meeting/Training --out $X --split train "${@:2}" ;;
  *) echo "알 수 없는 코퍼스 $1"; exit 2 ;;
esac
echo "[$(date "+%F %T")] NEWCORP_BUILD_EXIT $?"
