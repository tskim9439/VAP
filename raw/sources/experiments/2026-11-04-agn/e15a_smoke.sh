#!/usr/bin/env bash
# 2026-11-04 E15a 학습 전 점검(서버 호스트, 제출 없음): slurm/submit-q17-e15a.sh --dry-run 이 만든 stage3_train 인자 그대로 GPU 하나에서 2 step 학습 스모크 →
#   final 저장·재로드 parity(인코더 포함 — 아래 12 층은 G2b graft.pt 에서 왔다) → 저장본 인코더 대조(아래 층 = G2b enc, 위층 = 학습됨, config.encoder_train_layers).
#   산출 VAPKT-data/tmp/e15a-check(새 폴더). bash e15a_smoke.sh <GPU>. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e15a-check; P=$D/conda/envs/vapasr/bin/python; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VAPASR_LOCAL_CACHE=/tmp/sa_tskim VAPASR_TAR_INDEX_ROOT=$D/data/speechlm-asr-tar-index-v1 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
A=$(bash slurm/submit-q17-e15a.sh --dry-run --partition hpc 2>&1 | grep -o "ARGS_EXTRA=.*" | head -1 | sed -e 's/^ARGS_EXTRA=//' -e 's# slurm/stage3-train-apex.sbatch$##')
[ -n "$A" ] || { echo "인자를 못 읽었다"; exit 1; }
[ -e $T/smoke ] && { echo "이미 있다: $T/smoke(새 폴더에만 쓴다)"; exit 1; }
echo "$A" > $T/args.txt
echo "[$(date '+%F %T')] 스모크 시작(GPU $1)" > $T/smoke.log
CUDA_VISIBLE_DEVICES=$1 $P experiments/stage3_train.py --nemotron-dir $MXC_NEMOTRON_DIR --out-dir $T/smoke ${A//%2C/,} --max-steps 2 --save-every 1000 --log-every 1 >> $T/smoke.log 2>&1
echo "[$(date '+%F %T')] SMOKE_EXIT $?" >> $T/smoke.log
CUDA_VISIBLE_DEVICES= $P - >> $T/smoke.log 2>&1 <<'EOF'
import glob, json, torch
from safetensors.torch import load_file
T = "/soundai/users/tskim/VAPKT-data/tmp/e15a-check/smoke/final"
cfg = json.load(open(f"{T}/config.json")); g = torch.load("/soundai/Model/VAPASR/graft-q17-graft-g2b/final/graft.pt", map_location="cpu", weights_only=False)["enc"]
sd = {k[len("encoder.enc."):]: v for f in sorted(glob.glob(f"{T}/*.safetensors")) for k, v in load_file(f).items() if k.startswith("encoder.enc.")}
lay = lambda k: int(k.split(".")[1]) if k.startswith("layers.") else -1
same = sorted({lay(k) for k, v in sd.items() if k in g and torch.equal(v.float(), g[k].float())}); moved = sorted({lay(k) for k, v in sd.items() if k in g and not torch.equal(v.float(), g[k].float())})
print(f"ENC-CHECK 저장 인코더 텐서 {len(sd)}(graft {len(g)}) · graft 와 같은 층 {same} · 달라진 층 {moved} · encoder_train_layers {cfg.get('encoder_train_layers')} "
      f"· encoder_trainable {cfg.get('encoder_trainable')} · encoder_saved {cfg.get('encoder_saved')}")
ok = len(sd) == len(g) and all(l >= 12 for l in moved) and cfg.get("encoder_train_layers") == [12, 24]   # 서브샘플링(−1)·아래 12 층은 graft 그대로
print("ENC-CHECK OK" if ok else "ENC-CHECK FAIL")
EOF
echo "[$(date '+%F %T')] ALL_DONE" >> $T/smoke.log
