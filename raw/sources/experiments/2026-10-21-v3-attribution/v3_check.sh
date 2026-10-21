# V3-p1 학습 전 점검(mxc 호스트): bash v3_check.sh r2|r1|r4|r2x|r4x|r1x|r1x4|r2m — 학습 dry-run(CPU) → 스모크 20 step(가장 빈 호스트 GPU). SMOKE=0 이면 dry-run 만.
# 둘 다 st4-m2a 설정(stage4_m2_check.sh 에서 rep_chde 를 뺀 것) + V3 플래그. r2 = C1 + C5(교대 3·머무름 1.5·교차 언어 합성 <EOT> 문맥 제거 1.0),
# r1 = r2 + C2(sim-dialogue-xl-tel-v1 1.0, sim_en-de 0.5, tel 셋 창 단위 전화 대역 끔). 산출 /tmp/sa_tskim/v3-<모드>-{dry,smoke}-train — 지우지 않는다.
# r4 = r2 + --xl-slot-order arrival(교차 언어 슬롯 = 먼저 말한 순서, prefix 는 두 언어만). 스모크 체크포인트로 config 키와 평가 경로(합성 xl 창 8·전화 dev 3 세션,
#   /tmp/sa_tskim/v3-r4-smoke-{xl,tel})까지 돌린다 — 출력 품질이 아니라 경로 점검.
# r2x·r4x = r2·r4 + --decision-weight-sets xl(교대·머무름 가중을 교차 언어 합성 셋에만 — 같은 언어 가드가 깨졌을 때의 사전 등록 대비책).
# r1x·r1x4 = r1(전화 합성 셋 C2) + --decision-weight-sets xl [+ --xl-slot-order arrival], r2m = r2 + 같은 언어 셋 교대·머무름 1.5(--same-lang-*-weight).
set -u
MODE=${1:?r2|r1|r4|r2x|r4x|r1x|r1x4|r2m}
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2
SNAP=$D/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637; ASR=$D/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
CV4=$D/cv4-en-v1/lists/cv4-en-words-train.list; DE=$D/yodas-de-v1/lists/de-words-train-tn2.list
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_TAR_INDEX_ROOT=$D/speechlm-asr-tar-index-v1 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
SIM=$D/sim-dialogue-v2,$D/sim-dialogue-xl-v1; W=dlg_ko:2,dlg_en:0.7,dlg_de:0.7,sim_ko:0.5,sim_en:0.8,sim_de:0.8,sim_en-de:1.0,sim_en-ko:1.0,rep_ko:1.2,rep_en:1.2,rep_de:1.2; EXTRA=""
if [ "$MODE" = r1 ] || [ "$MODE" = r1x ] || [ "$MODE" = r1x4 ]; then
  [ -f $D/sim-dialogue-xl-tel-v1/sim_en-de-tel.refined.dialogues.jsonl ] || { echo "tel 셋이 아직 없다(build_tel.sh 먼저)"; exit 2; }
  SIM=$SIM,$D/sim-dialogue-xl-tel-v1; W=dlg_ko:2,dlg_en:0.7,dlg_de:0.7,sim_ko:0.5,sim_en:0.8,sim_de:0.8,sim_en-de:0.5,sim_en-de-tel:1.0,sim_en-ko:1.0,rep_ko:1.2,rep_en:1.2,rep_de:1.2
  EXTRA="--set-phone sim_en-de-tel:0"
fi
[ "$MODE" = r4 ] && EXTRA="--xl-slot-order arrival"
[ "$MODE" = r2x ] && EXTRA="--decision-weight-sets xl"
[ "$MODE" = r4x ] && EXTRA="--decision-weight-sets xl --xl-slot-order arrival"
[ "$MODE" = r1x ] && EXTRA="$EXTRA --decision-weight-sets xl"
[ "$MODE" = r1x4 ] && EXTRA="$EXTRA --decision-weight-sets xl --xl-slot-order arrival"
[ "$MODE" = r2m ] && EXTRA="--same-lang-change-weight 1.5 --same-lang-hold-weight 1.5"
V3="--change-tag-weight 3 --hold-tag-weight 1.5 --eot-drop-synth 1.0 --eot-drop-sets xl $EXTRA"
COMMON="--init /soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000 --nemotron-dir $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json \
  --max-windows 300 --replay-words @$SNAP/main-words.list --replay-words @$ASR --replay-words @$CV4 --replay-de-words @$DE --replay-files-frac 0.01 --replay-max-items 300 \
  --ast-targets $D/ast-mono-v1/targets-fix-gen-v3-bf16,$D/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16 --ast-frac 0.8 --offline-frac 0.3 --gain-augment 0.5,-70,-20 --speed-perturb 0.9,1.0,1.1 --noise-bank $D/noise-bank-v1 \
  --train-encoder --encoder-right-context 3 --encoder-context-sampling fixed --spec-augment 2,27,2,0.03 --attn-impl varlen --batch-max-tokens 16384 --pack-max-bs 64 \
  --backchannel 1.0,3 --spk-act-weight 0.5 --revive-dir $P2/revive-asr-v2 \
  --dlg-trans $D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16 --dlg-trans-input $D/stage4-dlg-units-revive-v1/targets-fix-gen-v3-bf16 --dlg-ast-frac 0.5 \
  --sim-data $SIM --sim-trans-frac 0.8 --trans-lead-space --phone-aug 0.3 --weights $W $V3 \
  --lr 4e-5 --max-steps 20 --warmup 5 --save-every 1000 --log-every 2 --num-workers 2 --report-to none"
echo "[$(date "+%F %T")] $MODE dry-run"
$P experiments/stage3_train.py --dry-run $COMMON --out-dir /tmp/sa_tskim/v3-$MODE-dry-train --check-items 32 2>&1 | grep -v "regex pattern"
echo "DRY EXIT ${PIPESTATUS[0]}"
[ "${SMOKE:-1}" = 0 ] && { echo "V3_CHECK_DONE $MODE"; exit 0; }
GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F", " "{print \$1, \$3 - \$2, \$4}" | sort -k2,2nr -k3,3n | head -1 | cut -d" " -f1); echo "GPU $GPU"
echo "[$(date "+%F %T")] $MODE smoke"
$P experiments/stage3_train.py --gpu $GPU $COMMON --out-dir /tmp/sa_tskim/v3-$MODE-smoke-train --check-items 8 2>&1 | grep -v "regex pattern"
echo "SMOKE EXIT ${PIPESTATUS[0]}"
if [ "$MODE" = r4 ]; then
  CK=/tmp/sa_tskim/v3-r4-smoke-train/final; XE=$D/sim-dialogue-xl-eval-v1; DEV=$D/v3-dev/xl-tel-dev; SS=/tmp/sa_tskim/v3-r4-smoke-sessions.jsonl
  $P -c "import json, sys; c = json.load(open(sys.argv[1])); print('config xl_slot_order', c.get('xl_slot_order'), '· 지문', c['stage3'].get('xl_slot_order'))" $CK/config.json
  echo "[$(date "+%F %T")] r4 합성 xl 평가 경로"
  CUDA_VISIBLE_DEVICES=$GPU $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $XE --trans $XE/targets-sim_en-de --corpora sim_en-de --deltas 10 \
    --limit-windows 8 --trans-lead-space --batch-size 8 --out /tmp/sa_tskim/v3-r4-smoke-xl 2>&1 | grep -v "regex pattern" | tail -4; echo "XL EXIT ${PIPESTATUS[0]}"
  $P experiments/cst_stage4_eval.py summarize --out /tmp/sa_tskim/v3-r4-smoke-xl 2>&1 | tail -6
  $P -c "import json, os, sys; d = os.path.dirname(sys.argv[1]); [print(json.dumps(dict(r, mix=os.path.join(d, r['mix'])), ensure_ascii=False)) for r in map(json.loads, open(sys.argv[1]).readlines()[:3])]" $DEV/sessions-check.jsonl > $SS
  echo "[$(date "+%F %T")] r4 전화 dev 경로(cst_stage4_taxi)"
  CUDA_VISIBLE_DEVICES=$GPU $P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $SS --out /tmp/sa_tskim/v3-r4-smoke-tel --deltas 10 --max-window 90 \
    --batch-size 4 2>&1 | grep -v "regex pattern" | tail -3; echo "TEL EXIT ${PIPESTATUS[0]}"
  $P experiments/cst_stage4_taxi.py score --sessions $SS --out /tmp/sa_tskim/v3-r4-smoke-tel --turn-metrics 2>&1 | grep -o "'split': {[^}]*}" | head -1
  cat /tmp/sa_tskim/v3-r4-smoke-tel/slot-map.json; echo; grep -ho '"xl_slot_order": "[a-z]*"' /tmp/sa_tskim/v3-r4-smoke-xl/config-rank0.json /tmp/sa_tskim/v3-r4-smoke-tel/config.json
fi
echo "V3_CHECK_DONE $MODE"
