# 독일어 대화(KEC·ZTC) Phase 2 뒷단: ASR 대조 → 정렬 → 경계 보정 → mono 혼합 → 창. 호스트 GPU 0·4·6 × 워커 3. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
D=/soundai/users/tskim/VAPKT-data/data/phase2; L=$D/logs-de; mkdir -p $L $D/asrcheck
GPUS="0 4 6"; W=3; N=$(( $(echo $GPUS | wc -w) * W ))
run_shards() {   # $1 = 이름, 나머지 = 명령(샤드 인자 SHARD 자리)
  local name=$1; shift; local i=0
  for g in $GPUS; do for w in $(seq 1 $W); do
    CUDA_VISIBLE_DEVICES=$g "$@" --shard $i/$N --gpu 0 --mem-frac 0.26 > $L/$name-$i.log 2>&1 &
    i=$((i + 1)); sleep 5
  done; done; wait
}
for C in kec ztc; do
  echo "[$(date '+%F %T')] $C asr check" >> $L/status
  MXC_QWEN_ASR_DIR=/soundai/Model/Qwen3-ASR-1.7B run_shards asr-$C $P experiments/p2_asr_check.py --dialogues $D/$C.dialogues.jsonl --out $D/asrcheck/$C.json --limit-utts 100000000
  $P - "$D/asrcheck" "$C" <<'PY' >> $L/status
import sys, os, json, glob
d, c = sys.argv[1:3]; rows = []
for f in sorted(glob.glob(os.path.join(d, f"{c}.shard*.utts.jsonl"))): rows += [json.loads(l) for l in open(f, encoding="utf-8")]
with open(os.path.join(d, f"{c}.utts.jsonl"), "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
errs = sorted(r["err"] for r in rows); n = len(errs)
print(json.dumps(dict(corpus=c, n=n, err_median=errs[n // 2] if n else None, share_ge_0_2=sum(e >= 0.2 for e in errs) / max(1, n), share_ge_0_3=sum(e >= 0.3 for e in errs) / max(1, n))))
PY
  echo "[$(date '+%F %T')] $C align" >> $L/status
  MXC_QWEN_ASR_DIR=/soundai/Model/Qwen3-ASR-0.6B MXC_ALIGNER_DIR=/soundai/Model/Qwen3-ForcedAligner-0.6B run_shards align-$C $P experiments/p2_align.py --dialogues $D/$C.dialogues.jsonl
  echo "[$(date '+%F %T')] $C align done" >> $L/status
done
echo "[$(date '+%F %T')] GPU STAGES DONE" >> $L/status
