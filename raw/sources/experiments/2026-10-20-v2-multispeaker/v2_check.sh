# V2 첫 학습 전 점검(mxc 호스트): 단위 시험(CPU) → 학습 dry-run(CPU) → GPU 대기 → 스모크 20 step → 슬롯·탭 가중치 확인 → 평가 경로 스모크(4 슬롯 직렬 = 배치).
# 설정 = st4-m2a(전화 대역 증강) + V2(--enc-tap-layer 0 · --max-speakers 4 · --window 20,90 --window-draws 6 · 회의 dlg_mt · 3–4 인 합성 v3) — 축소 데이터(stage4_m2_check.sh 와 같은 방식).
# CODE(기본 /soundai/users/tskim/VAPKT) 의 코드를 쓴다. MEET_TR=1 이면 회의 번역 타깃(stage4-dlg-units-meet-v1)도 넣는다. 산출 /tmp/sa_tskim/v2-* — 지우지 않는다.
set -u
CODE=${CODE:-/soundai/users/tskim/VAPKT}; cd $CODE; set -a; source /soundai/users/tskim/VAPKT/.env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; M=/soundai/Model/VAPASR
SNAP=$D/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637; ASR=$D/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
CV4=$D/cv4-en-v1/lists/cv4-en-words-train.list; DE=$D/yodas-de-v1/lists/de-words-train-tn2.list
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_TAR_INDEX_ROOT=$D/speechlm-asr-tar-index-v1 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; MIN_FREE_MB=${MIN_FREE_MB:-70000}
DT=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; [ "${MEET_TR:-0}" = 1 ] && DT=$DT,$D/stage4-dlg-units-meet-v1/targets-fix-gen-v3-bf16
echo "[$(date "+%F %T")] 단위 시험"
CUDA_VISIBLE_DEVICES= $P -m pytest -q -p no:cacheprovider tests/test_v2_slots.py tests/test_enc_tap.py tests/test_cst_dialogue.py tests/test_cst_eval.py tests/test_packing.py tests/test_stage3_train.py tests/test_sim_dialogue.py tests/test_dialogue_dataset.py 2>&1 | tail -2
TRAIN="--init $M/stage3-st3-p11-revive/checkpoint-3000 --nemotron-dir $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json \
  --dlg-corpora aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar --max-speakers 4 --declare-all-p 0.5 --window 20,90 --window-draws 6 \
  --max-windows 300 --replay-words @$SNAP/main-words.list --replay-words @$ASR --replay-words @$CV4 --replay-de-words @$DE --replay-files-frac 0.01 --replay-max-items 300 \
  --ast-targets $D/ast-mono-v1/targets-fix-gen-v3-bf16,$D/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16 --ast-frac 0.8 --offline-frac 0.3 --gain-augment 0.5,-70,-20 --speed-perturb 0.9,1.0,1.1 --noise-bank $D/noise-bank-v1 \
  --train-encoder --encoder-right-context 3 --encoder-context-sampling fixed --spec-augment 2,27,2,0.03 --attn-impl varlen --batch-max-tokens 16384 --pack-max-bs 64 \
  --backchannel 1.0,3 --spk-act-weight 0.5 --revive-dir $P2/revive-asr-v2 --dlg-trans $DT --dlg-trans-input $D/stage4-dlg-units-revive-v1/targets-fix-gen-v3-bf16 --dlg-ast-frac 0.5 \
  --sim-data $D/sim-dialogue-v3-multi,$D/sim-dialogue-xl-v1 --sim-trans-frac 0.8 --trans-lead-space --phone-aug 0.3 --enc-tap-layer 0 \
  --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,dlg_mt:1.0,sim_ko:0.8,sim_en:0.8,sim_de:0.8,sim_en-de:1.0,sim_en-ko:1.0,rep_ko:1.2,rep_en:1.2,rep_de:1.2 \
  --lr 4e-5 --max-steps 20 --warmup 5 --save-every 1000 --log-every 2 --num-workers 2 --report-to none"
echo "[$(date "+%F %T")] 학습 dry-run"
$P experiments/stage3_train.py --dry-run $TRAIN --out-dir /tmp/sa_tskim/v2-dry-train --check-items 32 2>&1 | grep -v "regex pattern" | grep -E "check |dlg_|sim_|rep_|섞기|dry-run 끝|Error|error|Traceback" | cut -c1-300 | tail -40
echo "DRY EXIT ${PIPESTATUS[0]}"
while true; do
  g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
  [ "$(echo $g | cut -d' ' -f2)" -ge $MIN_FREE_MB ] && break; echo "[$(date "+%F %T")] GPU 대기(최대 빈 메모리 $(echo $g | cut -d' ' -f2) MiB)"; sleep 300
done
GPU=$(echo $g | cut -d' ' -f1); echo "[$(date "+%F %T")] 스모크 20 step — GPU $GPU ($g)"
$P experiments/stage3_train.py --gpu $GPU $TRAIN --out-dir /tmp/sa_tskim/v2-smoke-train --check-items 8 2>&1 | grep -v "regex pattern" | grep -E "model ←|섞기|'loss'|check-save|PARITY|train 종료|Error|error|Traceback" | cut -c1-400 | tail -30
echo "SMOKE EXIT ${PIPESTATUS[0]}"
F=/tmp/sa_tskim/v2-smoke-train/final
$P - $F <<'EOF'
import glob, json, sys
from safetensors import safe_open
f = sys.argv[1]; c = json.load(open(f + "/config.json")); st = c.get("stage3", {})
print("slot_tokens", c.get("slot_tokens"), "· spk_slots", c.get("spk_slots"), "· enc_tap_layer", c.get("enc_tap_layer"), "· 차단에 SPK_3/4", [i in c["blocked_ids"] for i in (151717, 151718)],
      "· 지문", {k: st.get(k) for k in ("max_speakers", "declare_all_p", "window_draws", "enc_tap", "window")})
w = {}
for p in glob.glob(f + "/*.safetensors"):
    with safe_open(p, "pt") as s:
        for k in s.keys():
            if k.startswith(("adapter_tap.net.3", "spk_head.2")): w[k] = tuple(s.get_slice(k).get_shape())
print("헤드·탭", w)
EOF
echo "[$(date "+%F %T")] 평가 경로 스모크(스모크 final, 4 슬롯)"
E=/tmp/sa_tskim/v2-eval-smoke
CUDA_VISIBLE_DEVICES=$GPU $P experiments/cst_stage3_eval.py run --checkpoint $F --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --out $E/meet --corpora ami,icsi --deltas 10 \
  --limit-windows 4 --batch-size 4 --verify --backchannel 1.0,3 --max-speakers 4 --window 20,90 --window-draws 1 --declare 4 2>&1 | grep -E "PARITY|DONE|Error|모델 슬롯" | tail -5
CUDA_VISIBLE_DEVICES=$GPU $P experiments/cst_stage3_eval.py run --checkpoint $F --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --out $E/s3 --corpora aihub134-2,otoSpeech --deltas 10 \
  --limit-windows 4 --batch-size 4 --verify --backchannel 1.0,3 2>&1 | grep -E "PARITY|DONE|Error" | tail -4
echo "[$(date "+%F %T")] V2_CHECK_DONE"
