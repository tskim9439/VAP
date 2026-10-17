# 안내 평가 + 오류 유형 분해: bash guide_chain.sh <GPU> <이름> <guide> [hold]
set -u
GPU=$1; NAME=$2; G=$3; HOLD=${4:-}
D=/soundai/users/tskim/VAPKT-data; CK=/soundai/Model/VAPASR/stage3-st3-p9-spkhead/checkpoint-3000; C=aihub134-2,aihub134-1,aihub71631,otoSpeech,kec,ztc,nsc
bash $D/results/stage3-regress/guide_eval.sh $GPU $CK $NAME $C 4,10 $G $HOLD
A=$D/results/stage3-attr-errors-$NAME; mkdir -p $A; cd /soundai/users/tskim/VAPKT
$D/conda/envs/vapasr/bin/python experiments/cst_attr_errors.py --pred $D/results/stage3-eval-$NAME --data $D/data/phase2 --split $D/data/phase2/splits/v2.json --out $A --deltas 4,10 --workers 16 > $A/run.log 2>&1
echo "[$(date '+%F %T')] ATTR-DONE"
