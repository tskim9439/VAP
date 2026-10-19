# CallHome 독일어 후처리(mxc 호스트): 정렬(Qwen3-ForcedAligner, 감독 발화만) → 정제(--no-vad: 화자 채널이 없다) → mono 캐시 → 창 → split v3(v2 + 통화 10 %).
# 산출 phase2/{align-asr-tn-de-v1/callhome-de, callhome-de.refined.dialogues.jsonl, mono/callhome-de, windows-chde/*, splits/v3.json}. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export MXC_QWEN_ASR_DIR=/soundai/Model/Qwen3-ASR-0.6B MXC_ALIGNER_DIR=/soundai/Model/Qwen3-ForcedAligner-0.6B
D=/soundai/users/tskim/VAPKT-data/data/phase2; A=$D/align-asr-tn-de-v1; C=callhome-de
GPUS=${GPUS:-"2 6"}; N=$(echo $GPUS | wc -w); i=0
echo "[$(date '+%F %T')] align"
for g in $GPUS; do CUDA_VISIBLE_DEVICES=$g $P experiments/p2_align.py --dialogues $D/$C.dialogues.jsonl --out-root $A --shard $i/$N --gpu 0 --mem-frac 0.2 > $D/../callhome-deu/work/align-$i.log 2>&1 & i=$((i + 1)); sleep 5; done; wait
tail -2 $D/../callhome-deu/work/align-*.log
echo "[$(date '+%F %T')] refine"
$P experiments/p2_refine.py --dialogues $D/$C.dialogues.jsonl --align $A/$C --no-vad 2>&1 | tail -4
echo "[$(date '+%F %T')] mono"
$P experiments/p2_build_mono.py --data $D --corpora $C --procs 16 2>&1 | tail -2
echo "[$(date '+%F %T')] windows"
mkdir -p $D/windows-chde
$P experiments/p2_build_windows.py --dir $D --corpora $C --out $D/windows-chde 2>&1 | grep -v "regex pattern" | tail -4
echo "[$(date '+%F %T')] split v3"
[ -f $D/splits/v3.json ] || $P experiments/p2_make_split.py --data $D --base $D/splits/v2.json --corpora $C --frac 0.10 --version v3 --out $D/splits/v3.json 2>&1 | tail -3
echo "CHDE_POST_DONE $(date '+%F %T')"
