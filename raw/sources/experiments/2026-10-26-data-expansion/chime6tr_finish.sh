# CHiME-6 train 마무리(mxc 호스트, 2026-10-26, 사용자 "train 세션만 학습에 넣기" — wiki decision-chime6-train-sessions-20261026): bash chime6tr_finish.sh
#   1) phase2x/chime6.refined(정렬 95744·정제 끝) → 자기 착용 마이크에서 들리지 않는 발화 표시(own_ch_inaudible — 죽은 채널 위 정렬이라 학습이 그 구간을 마스크)
#      + S05 제외(P14 채널이 세션 내내 죽어 mono 혼합 게인 +65 dB 가 잡음을 키운다) → phase2x/chime6tr.refined (p2_flag_channel_level)
#   2) newcorp_finish.sh chime6tr --stems: phase2/chime6tr.refined 복사(있으면 멈춤), mono + 화자별 신호 캐시
#   3) 원거리 버전: far_build.sh chime6 --key chime6tr(Kinect U01–U06 CH1) → phase2/chime6tr-far.refined  4) chime6tr-far mono 캐시.
#   키를 chime6 와 다르게 둔다 — phase2/chime6.dialogues.jsonl 은 dev 평가 셋이고 예전 평가 스크립트가 <corpus>.refined 를 먼저 찾는다. 아무것도 지우지 않는다.
set -u
S=$(cd "$(dirname "$0")" && pwd)
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; X=/soundai/users/tskim/VAPKT-data/data/phase2x; D=/soundai/users/tskim/VAPKT-data/data/phase2
C6=/soundai/users/tskim/VAPKT-data/data/audio/chime6/CHiME6_train/CHiME6/audio/train
echo "[$(date "+%F %T")] chime6tr_finish"
if [ ! -f $X/chime6tr.refined.dialogues.jsonl ]; then
  $P experiments/p2_flag_channel_level.py --src $X/chime6.refined.dialogues.jsonl --out $X/chime6tr.refined.dialogues.jsonl --drop-conv chime6:S05 || { echo "!! 표시 실패"; exit 1; }
fi
bash $S/newcorp_finish.sh chime6tr $X --stems || { echo "!! newcorp_finish 실패"; exit 1; }
bash $S/far_build.sh chime6 --key chime6tr --chime6-root $C6
[ -f $D/chime6tr-far.refined.dialogues.jsonl ] || { echo "!! 원거리 버전 없음"; exit 1; }
$P experiments/p2_build_mono.py --data $D --corpora chime6tr-far --procs 16
echo "[$(date "+%F %T")] CHIME6TR_FINISH_EXIT"
