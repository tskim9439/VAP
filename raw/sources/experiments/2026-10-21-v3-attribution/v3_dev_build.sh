# V3 C4 dev 준비(mxc 호스트 CPU): 전화 교차 언어 dev(sim-dialogue-xl-tel-eval-v1, 160 세션)와 짝 비교용 기존 교차 언어 평가(sim-dialogue-xl-eval-v1 앞 160 세션 — 같은 시드라 화자 짝이 같다)를
# TAXI 형식(sim_to_sessions)으로 바꾸고, tel dev 를 tune(앞 80)·check(뒤 80) 으로 나눈다. 산출 VAPKT-data/data/v3-dev/{xl-tel-dev,xl-eval-v1}/ — 있으면 멈춘다. 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; D=/soundai/users/tskim/VAPKT-data/data; O=$D/v3-dev
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; TOKZ=/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000
[ -e $O ] && { echo "이미 있다: $O"; exit 1; }
mkdir -p $O
$PY experiments/sim_to_sessions.py --data $D/sim-dialogue-xl-tel-eval-v1 --corpus sim_en-de-tel --tokenizer $TOKZ --out $O/xl-tel-dev --limit 160 2>&1 | grep -v "regex pattern" | tail -2
$PY experiments/sim_to_sessions.py --data $D/sim-dialogue-xl-eval-v1 --corpus sim_en-de --tokenizer $TOKZ --out $O/xl-eval-v1 --limit 160 2>&1 | grep -v "regex pattern" | tail -2
head -n 80 $O/xl-tel-dev/sessions.jsonl > $O/xl-tel-dev/sessions-tune.jsonl; tail -n +81 $O/xl-tel-dev/sessions.jsonl > $O/xl-tel-dev/sessions-check.jsonl
wc -l $O/*/sessions*.jsonl
echo "V3_DEV_BUILD_DONE"
