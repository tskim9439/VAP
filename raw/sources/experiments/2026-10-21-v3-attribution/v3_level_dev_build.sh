# V3 레벨 진단 dev(mxc 호스트 CPU): 전화 교차 언어 dev(sim-dialogue-xl-tel-eval-v1)와 같은 인자·시드로 대화를 다시 섞되 한 언어 화자만 레벨을 낮춘다(--level-offset-db).
#   en12 = 영어 화자 −12 dB, de12 = 독일어 화자 −12 dB(대조). 대화(발화·시각)는 원래 dev 와 같아야 한다(점검) → 레벨만 다른 짝 비교 셋.
#   TAXI 는 쓰지 않는다(평가 전용 — 레벨 차 가설은 TAXI 집계에서 나왔지만 값은 TAXI 에 맞추지 않고 대칭 −12 dB 두 방향으로 잰다).
#   산출 VAPKT-data/data/sim-dialogue-xl-tel-eval-lv-{en12,de12}-v1, v3-dev/xl-tel-dev-{en12,de12} — 있으면 멈춘다. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VAPASR_TAR_INDEX_ROOT=$D/speechlm-asr-tar-index-v1
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; TOKZ=/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000; K=sim_en-de-tel
TEL="--p-hold 0.05 --bc-scale 0 --fto-clip 0.05,0.8 --intra-pause 0.1,0.24 --jitter-db 8 --phone-per-speaker 1.0 --phone-same-frac 0.1 --name-suffix tel"   # build_tel.sh 와 같다
DE="--lang2 German --phase2-corpora2 kec,ztc,nsc --pool-weights2 kec:0.35,ztc:0.35,nsc:0.3"
for v in en12:English:-12 de12:German:-12; do
  n=${v%%:*}; off=${v#*:}; O=$D/sim-dialogue-xl-tel-eval-lv-$n-v1; S=$D/v3-dev/xl-tel-dev-$n
  for x in $O $S; do [ -e $x ] && { echo "이미 있다 — 멈춘다: $x"; exit 1; }; done
  echo "[$(date '+%F %T')] $n($off) 빌드"
  $PY experiments/sim_build_dialogues.py --lang English $DE --out $O --sessions 160 --dur 30,75 --workers 24 --phase2 $P2 --split $P2/splits/v2.json --noise-bank $D/noise-bank-v1 --noise-p 0.3 --heldout \
    --phase2-corpora otoSpeech --pool-weights otoSpeech:1 --trans $S1 $TEL --level-offset-db $off 2>&1 | grep -v "regex pattern" | tail -2
  $PY - $D/sim-dialogue-xl-tel-eval-v1/$K.refined.dialogues.jsonl $O/$K.refined.dialogues.jsonl <<'EOF'
import json, sys
A = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]; B = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
strip = lambda x: {k: v for k, v in x.items() if k != "meta"}
same = len(A) == len(B) and all(strip(a) == strip(b) for a, b in zip(A, B))
print(f"  같은 대화(meta 밖 전부): {same} · 세션 {len(A)}/{len(B)} · level_offset_db {B[0]['meta'].get('level_offset_db')}")
if not same: sys.exit(1)
EOF
  [ $? = 0 ] || { echo "원래 dev 와 대화가 다르다 — 멈춘다"; exit 1; }
  $PY experiments/sim_to_sessions.py --data $O --corpus $K --tokenizer $TOKZ --out $S --limit 160 2>&1 | grep -v "regex pattern" | tail -1
done
echo "V3_LEVEL_DEV_DONE $(date '+%F %T')"
