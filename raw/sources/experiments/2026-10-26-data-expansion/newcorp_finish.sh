# D4·D7·D8 새 코퍼스 마무리(mxc 호스트, 2026-10-26): 정렬(p2_align — 사용자 SLURM)이 끝난 뒤
#   bash newcorp_finish.sh <corpus> <준비 폴더> [--stems] [p2_refine 인자…]
#   1) 정제: <준비 폴더>/<corpus>.refined.dialogues.jsonl(p2_refine — 화자 채널 VAD 로 경계 보정, ASR 대조 파일이 있으면 그 플래그)
#   2) phase2 로 복사(이미 있으면 멈춤 — 덮어쓰지 않는다)  3) mono 캐시(+ --stems 면 화자별 신호 캐시). 아무것도 지우지 않는다.
set -u
C=$1; X=$2; shift 2; ST=0; [ "${1:-}" = "--stems" ] && { ST=1; shift; }
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; D=/soundai/users/tskim/VAPKT-data/data/phase2
echo "[$(date "+%F %T")] newcorp_finish $C ← $X"
R=$X/$C.refined.dialogues.jsonl
if [ ! -f $R ]; then
  FL=""; [ -f "$X/asrcheck/$C.utts.jsonl" ] && FL="--asr-flags $X/asrcheck/$C.utts.jsonl"
  $P experiments/p2_refine.py --dialogues $X/$C.dialogues.jsonl --align $X/align-asr-tn-v1/$C --workers 16 $FL "$@" || { echo "!! 정제 실패"; exit 1; }
fi
[ -f $D/$C.refined.dialogues.jsonl ] && { echo "!! 이미 있다(덮어쓰지 않는다): $D/$C.refined.dialogues.jsonl"; exit 1; }
cp $R $D/$C.refined.dialogues.jsonl.tmp && mv $D/$C.refined.dialogues.jsonl.tmp $D/$C.refined.dialogues.jsonl && echo "복사 → $D/$C.refined.dialogues.jsonl"
$P experiments/p2_build_mono.py --data $D --corpora $C --procs 16
(( ST )) && $P experiments/p2_build_mono.py --data $D --corpora $C --procs 16 --stems
echo "[$(date "+%F %T")] NEWCORP_FINISH_EXIT $C"
