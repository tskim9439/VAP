# 대화 발화 단독 디코드 대조(mxc 호스트): bash isolated_chain.sh <GPU 둘 쉼표> — Stage 3 회귀 창의 전사 발화를 하나씩 잘라 단일 화자 형식으로 디코드한다.
# 1. prepare(CPU): VAPKT-data/data/evaluation/dialogue-isolated-v1 (mix = 대화 mono 혼합 구간, chan = 그 화자 채널만)
# 2. eval_single_turn_asr run: st4-p1 final · Stage 2 최종 q06-s4-ast final 을 GPU 둘에 나눠 동시에(δ4·10, 직렬/배치 대조)
# 3. score: 발화 단독 오류 vs 같은 창의 대화 평가 agn·cp(st4-p1 final, p11 ck3000). 아무것도 지우지 않는다.
set -u
G1=${1%%,*}; G2=${1##*,}
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; O=$D/evaluation/dialogue-isolated-v1; R=/soundai/users/tskim/VAPKT-data/results
ST4=/soundai/Model/VAPASR/stage3-st4-p1/final; S2=/soundai/Model/VAPASR/semcommit-q06-s4-ast/final
mkdir -p $O
echo "[$(date '+%F %T')] prepare"
[ -f $O/manifest.jsonl ] || $P experiments/cst_isolated_asr.py prepare --checkpoint $ST4 --data $P2 --split $P2/splits/v2.json --out $O > $O/prepare.log 2>&1
tail -30 $O/prepare.log
echo "[$(date '+%F %T')] decode GPU $G1(st4-p1 final) · $G2(q06-s4-ast)"
CUDA_VISIBLE_DEVICES=$G1 $P experiments/eval_single_turn_asr.py run --manifest $O/manifest.jsonl --checkpoint $ST4 --encoder $MXC_NEMOTRON_DIR --out $O/st4-p1-final \
  --deltas 4 10 --batch-size 64 --verify > $O/st4-p1-final.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G2 $P experiments/eval_single_turn_asr.py run --manifest $O/manifest.jsonl --checkpoint $S2 --encoder $MXC_NEMOTRON_DIR --out $O/q06-s4-ast \
  --deltas 4 10 --batch-size 64 --verify > $O/q06-s4-ast.log 2>&1 &
wait
echo "[$(date '+%F %T')] score"
$P experiments/cst_isolated_asr.py score --out $O --runs st4-p1-final,q06-s4-ast --dialogue st4=$R/stage3-eval-st4-p1-final,p11=$R/stage3-eval-st3-p11-revive-ck3000 > $O/score.log 2>&1
cat $O/score.log
echo "CHAIN DONE"
