# AMI 원거리 마무리(mxc 호스트, 2026-10-26): 사용자 azcopy 업로드 뒤 bash ami_far_finish.sh
#   업로드 위치 /soundai/users/tskim/VAPKT-data/data/audio/ami-array1-train/<회의>/<회의>.Array1-0k.wav(사용자 SAS 는 users/tskim 아래만 쓰기 — DB/raw 는 403).
#   Blob API 로 만든 폴더·파일은 상위 폴더 시각을 바꾸지 않아 이 호스트 NFS 가 예전 조회(없음)를 캐시할 수 있다 — 업로드가 끝나기 전에는 이 경로를 조회하지 않는다.
#   1) 대조: 회의 폴더 135 · wav 1,080 · 합계 74,174,109,160 B(T5 원본) · 머리 RIFF   2) far_build.sh ami --ami-root(Array1 8 채널 → phase2/ami-far)
#   3) ami-far mono 캐시   4) F1 dry-run(--allow-missing 없이 — 모든 셋). 아무것도 지우지 않는다.
set -u
S=$(cd "$(dirname "$0")" && pwd)
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; D=/soundai/users/tskim/VAPKT-data/data/phase2; A=/soundai/users/tskim/VAPKT-data/data/audio/ami-array1-train
echo "[$(date "+%F %T")] ami_far_finish ← $A"
n=$(find $A -mindepth 1 -maxdepth 1 -type d | wc -l); w=$(find $A -name "*.wav" ! -name "._*" | wc -l)
b=$(find $A -name "*.wav" ! -name "._*" -exec stat -c %s {} + | awk '{s+=$1} END {print s}')
r=$(find $A -name "*.wav" ! -name "._*" -exec sh -c 'for f; do [ "$(head -c 4 "$f")" = RIFF ] || echo "$f"; done' sh {} + | wc -l)
echo "회의 $n · wav $w · 바이트 $b · RIFF 아님 $r"
[ "$n" = 135 ] && [ "$w" = 1080 ] && [ "$b" = 74174109160 ] && [ "$r" = 0 ] || { echo "!! 대조 실패(기대: 135 · 1080 · 74174109160 · 0)"; exit 1; }
bash $S/far_build.sh ami --ami-root $A
[ -f $D/ami-far.refined.dialogues.jsonl ] || { echo "!! 원거리 버전 없음"; exit 1; }
$P experiments/p2_build_mono.py --data $D --corpora ami-far --procs 16
bash $S/f1_check.sh
echo "[$(date "+%F %T")] AMI_FAR_FINISH_EXIT"
