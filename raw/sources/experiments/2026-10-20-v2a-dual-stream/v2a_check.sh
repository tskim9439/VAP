# V2-a 두 스트림 입력 학습 전 점검(mxc 호스트): 단위 시험(CPU) → GPU 대기 → v2a_check.py(V1 = st4-m1 ck1500) → 학습 dry-run(CPU) → 스모크 20 step → 탭 가중치·평가 경로 스모크.
# 학습 설정은 st4-m1 과 같고 --enc-tap-layer 0 만 더했다(stage4_m2_check.sh 와 같은 축소 데이터). 산출 /tmp/sa_tskim/v2a-* — 지우지 않는다.
# GPU 는 빈 메모리 MIN_FREE_MB(기본 70000) 이상이 될 때까지 기다린다(호스트를 다른 사용자와 같이 쓴다).
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; M=/soundai/Model/VAPASR; X=raw/sources/experiments/2026-10-20-v2a-dual-stream
SNAP=$D/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637; ASR=$D/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
CV4=$D/cv4-en-v1/lists/cv4-en-words-train.list; DE=$D/yodas-de-v1/lists/de-words-train-tn2.list
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_TAR_INDEX_ROOT=$D/speechlm-asr-tar-index-v1 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; MIN_FREE_MB=${MIN_FREE_MB:-70000}
echo "[$(date "+%F %T")] 단위 시험"
CUDA_VISIBLE_DEVICES= $P -m pytest -q tests/test_enc_tap.py tests/test_offline_mode.py tests/test_packing.py tests/test_semcommit_tokens_loss.py tests/test_stage3_train.py tests/test_twopass.py tests/test_lane_state.py tests/test_phone_aug.py 2>&1 | tail -3
TRAIN="--init $M/stage3-st3-p11-revive/checkpoint-3000 --nemotron-dir $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json \
  --max-windows 300 --replay-words @$SNAP/main-words.list --replay-words @$ASR --replay-words @$CV4 --replay-de-words @$DE --replay-files-frac 0.01 --replay-max-items 300 \
  --ast-targets $D/ast-mono-v1/targets-fix-gen-v3-bf16,$D/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16 --ast-frac 0.8 --offline-frac 0.3 --gain-augment 0.5,-70,-20 --speed-perturb 0.9,1.0,1.1 --noise-bank $D/noise-bank-v1 \
  --train-encoder --encoder-right-context 3 --encoder-context-sampling fixed --spec-augment 2,27,2,0.03 --attn-impl varlen --batch-max-tokens 16384 --pack-max-bs 64 \
  --backchannel 1.0,3 --spk-act-weight 0.5 --revive-dir $P2/revive-asr-v2 \
  --dlg-trans $D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16 --dlg-trans-input $D/stage4-dlg-units-revive-v1/targets-fix-gen-v3-bf16 --dlg-ast-frac 0.5 \
  --sim-data $D/sim-dialogue-v2,$D/sim-dialogue-xl-v1 --sim-trans-frac 0.8 --trans-lead-space --enc-tap-layer 0 \
  --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,sim_ko:0.5,sim_en:0.8,sim_de:0.8,sim_en-de:1.0,sim_en-ko:1.0,rep_ko:1.2,rep_en:1.2,rep_de:1.2 \
  --lr 4e-5 --max-steps 20 --warmup 5 --save-every 1000 --log-every 2 --num-workers 2 --report-to none"
echo "[$(date "+%F %T")] 학습 dry-run"
$P experiments/stage3_train.py --dry-run $TRAIN --out-dir /tmp/sa_tskim/v2a-dry-train --check-items 32 2>&1 | grep -v "regex pattern" | tail -25
echo "DRY EXIT ${PIPESTATUS[0]}"
while true; do
  g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
  [ "$(echo $g | cut -d' ' -f2)" -ge $MIN_FREE_MB ] && break; echo "[$(date "+%F %T")] GPU 대기(최대 빈 메모리 $(echo $g | cut -d' ' -f2) MiB)"; sleep 300
done
GPU=$(echo $g | cut -d' ' -f1); echo "[$(date "+%F %T")] GPU $GPU ($g)"
echo "[$(date "+%F %T")] v2a_check(V1 = st4-m1 ck1500, 탭 L00)"
CUDA_VISIBLE_DEVICES=$GPU $P $X/v2a_check.py --ckpt $M/stage3-st4-m1/checkpoint-1500 --data $P2 --split $P2/splits/v2.json --out /tmp/sa_tskim/v2a-check.json --layer 0 --windows 4 2>&1 | grep -v -i "warn\|regex\|warprnnt" | tail -45
echo "[$(date "+%F %T")] 스모크 20 step"
$P experiments/stage3_train.py --gpu $GPU $TRAIN --out-dir /tmp/sa_tskim/v2a-smoke-train --check-items 8 2>&1 | grep -v "regex pattern" | grep -E "model ←|인코더 탭|'loss'|check-save|PARITY|train 종료|Error|error" | cut -c1-400 | tail -30
echo "SMOKE EXIT ${PIPESTATUS[0]}"
F=/tmp/sa_tskim/v2a-smoke-train/final
$P - $F <<'EOF'
import glob, json, sys
from safetensors import safe_open
f = sys.argv[1]; c = json.load(open(f + "/config.json")); w = {}
for p in glob.glob(f + "/*.safetensors"):
    with safe_open(p, "pt") as s:
        for k in s.keys():
            if k.startswith("adapter_tap."): w[k] = s.get_tensor(k).float()
print("config enc_tap_layer", c.get("enc_tap_layer"), "hidden", c.get("enc_tap_hidden"), "· stage3.enc_tap", c.get("stage3", {}).get("enc_tap"))
print("adapter_tap 키", len(w), "· net.3.weight |w|", round(float(w["adapter_tap.net.3.weight"].abs().mean()), 6), "· net.3.bias |b|", round(float(w["adapter_tap.net.3.bias"].abs().mean()), 6))
EOF
echo "[$(date "+%F %T")] 평가 경로 스모크(스모크 final)"
E=/tmp/sa_tskim/v2a-eval-smoke
CUDA_VISIBLE_DEVICES=$GPU $P experiments/cst_stage3_eval.py run --checkpoint $F --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --out $E/s3 --corpora aihub134-2,otoSpeech --deltas 10 \
  --limit-windows 4 --batch-size 4 --verify --backchannel 1.0,3 2>&1 | grep -E "PARITY|DONE|Error" | tail -5
CUDA_VISIBLE_DEVICES=$GPU $P experiments/cst_stage4_eval.py run --checkpoint $F --encoder $MXC_NEMOTRON_DIR --data $D/sim-dialogue-xl-eval-v1 --trans $D/sim-dialogue-xl-eval-v1/targets-sim_en-de \
  --corpora sim_en-de --deltas 10 --limit-windows 4 --batch-size 4 --trans-lead-space --out $E/xl 2>&1 | grep -E "DONE|Error" | tail -3
echo "[$(date "+%F %T")] V2A_CHECK_DONE"
