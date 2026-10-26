# D-1 원거리 버전 대화 빌드(mxc 호스트, 2026-10-26): bash far_build.sh <notsofar|icsi|ami|chime6> [experiments/p2_build_far.py 인자 — --dry-run·--limit·--ami-root 등]
#   → <phase2>/<corpus>-far.refined.dialogues.jsonl + .build.json(장치 지연·버린 변형). 그 뒤 mono 캐시는 p2_build_mono.py --corpora <corpus>-far. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; D=/soundai/users/tskim/VAPKT-data/data/phase2
echo "[$(date "+%F %T")] far_build $*"
$P experiments/p2_build_far.py --corpus "$1" --data $D --split $D/splits/v4.json "${@:2}"
echo "[$(date "+%F %T")] FAR_BUILD_EXIT $?"
