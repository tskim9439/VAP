# VibeVoice-ASR 외부 기준선(mxc 호스트): bash vibevoice_chain.sh <GPU 둘 쉼표> [smoke]
# 전제: 사용자가 T5 의 vapkt-corpora/vibevoice 를 $V 로 올렸다(모델 두 개, Qwen2.5-7B tokenizer, 소스 zip, pydeps 휠).
# 1. 소스 zip·휠을 $V/src·$V/pydeps 에 푼다(이미 있으면 건너뜀). vapasr env(transformers 4.57) + PYTHONPATH 로 돌린다 — env 는 건드리지 않는다.
# 2. stream(VibeVoice-ASR-Streaming-7B)·offline(VibeVoice-ASR)을 GPU 둘에서 동시에: Stage 3 회귀와 같은 창(코퍼스마다 200), smoke 면 코퍼스마다 2 창.
# 3. score: 같은 창의 st4-p1 final·p11 ck3000 과 나란히. 아무것도 지우지 않는다.
set -u
G1=${1%%,*}; G2=${1##*,}; SMOKE=${2:-}
V=/soundai/users/tskim/VAPKT-data/models/vibevoice; R=/soundai/users/tskim/VAPKT-data/results; P2=/soundai/users/tskim/VAPKT-data/data/phase2
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
[ -d $V/src ] || (mkdir -p $V/src && cd $V/src && $P -m zipfile -e $V/VibeVoice-src-16fb2cb.zip .)
SRC=$(ls -d $V/src/VibeVoice-* | head -1)
if [ ! -d $V/pydeps/diffusers ]; then for w in $V/pydeps/*.whl; do $P -m zipfile -e $w $V/pydeps/; done; fi
export PYTHONPATH=/soundai/users/tskim/VAPKT:$V/pydeps PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
ST4=/soundai/Model/VAPASR/stage3-st4-p1/final
LIM=200; TAG=""; [ "$SMOKE" = smoke ] && LIM=2 && TAG=-smoke
for m in stream offline; do
  if [ $m = stream ]; then G=$G1; M=$V/VibeVoice-ASR-Streaming-7B; X=""; else G=$G2; M=$V/VibeVoice-ASR; X="--lm-tokenizer $V/Qwen2.5-7B-tokenizer"; fi
  O=$R/baseline-vibevoice-$m$TAG; mkdir -p $O
  CUDA_VISIBLE_DEVICES=$G $P experiments/baseline_vibevoice.py run --mode $m --model $M --src $SRC $X --checkpoint $ST4 --data $P2 --split $P2/splits/v2.json \
    --out $O --limit-windows $LIM > $O/run.log 2>&1 &
done
wait
for m in stream offline; do
  O=$R/baseline-vibevoice-$m$TAG
  $P experiments/baseline_vibevoice.py score --out $O --compare st4final=$R/stage3-eval-st4-p1-final,p11=$R/stage3-eval-st3-p11-revive-ck3000 > $O/score.log 2>&1
  echo "== $m"; tail -3 $O/run.log; cat $O/score.log
done
echo "CHAIN DONE"
