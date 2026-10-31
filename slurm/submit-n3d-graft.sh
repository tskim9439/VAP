#!/usr/bin/env bash
# G1 이식 어댑터 학습 제출(2026-10-29, 사용자 "G1 도 구현해놓자" — wiki output-n3d-branch-20261029 4.2 절):
# slurm/stage3-train-apex.sbatch 를 SCRIPT=experiments/n3d_graft_train.py 로 낸다(선점 저장·requeue·감시 그대로). 제출은 사용자(스케줄러 셸).
#   학생 = E9 final 인코더(고정)의 홀수 층 + 마지막 층 탭 → 어댑터(능형 회귀에서 시작, 인과 블록 3, 약 16M) → Nemotron 3 Diarization 8 층 위·머리(고정).
#   교사 = Nemotron 3 Diarization 창 통째. 손실 = 교사 10 ms 확률 BCE + 0.1·h_8 MSE(--extra 로 바꾼다).
#   데이터 = 대화 셋(F1 과 같은 목록 — 원거리 회의 포함, split v4 held-out 제외) + 합성 셋, 창 20–40 s draw 2, 잔향·잡음·이득(diar_train 과 같은 증강).
#   교사가 라벨을 주므로 회의·원거리 셋에 가중을 더 준다(아래 W).
#   산출 /soundai/Model/VAPASR/graft-<run>(checkpoint-N · final/graft.pt · parity.json · DONE). 평가 묶음 q17-graft(창·회의 DER, 스트리밍 청크 4)를 이어서 낸다.
# 쓰기: bash slurm/submit-n3d-graft.sh [--run q17-graft-l8] [--layer 8] [--init <체크포인트>] [--n3d <.nemo>] [--nodes 1] [--partition apex] [--max-steps 6000]
#        [--batch 8] [--after-job JOBID] [--extra "n3d_graft_train.py 인자"] [--no-eval] [--asr-eval] [--dry-run]. 아무것도 지우지 않는다.
# U3(2026-10-30, 사용자 "U3 아래 12 층 푸는 걸로 구현해놓자"): --extra 에 --unfreeze-layers N --enc-lr --guard-weight 를 준다(n3d_graft_train 설명).
#   --asr-eval: 학습 뒤 평가 묶음 q17-u3-asr(E11 에 새 인코더를 끼워 AMI 창 δ10 — ASR 지킴 확인)도 이어서 낸다(U3_RUN=--run).
# G2(2026-10-31, 사용자 "진행해봐" — wiki output-diarizer-replacement-20261031 6 절): U3 final 에서 --init-graft 로 시작하고 --extra 에
#   --n3d-train lora|full --n3d-lr --lora-rank --lora-alpha [--n3d-head] 를 더한다(학생 N3D 위층을 함께 학습 — n3d_graft_train 설명). q17-graft 평가가 n3d_state 를 읽는다.
set -euo pipefail
cd "$(dirname "$0")/.."
run=q17-graft-l8; layer=8; init=/soundai/Model/VAPASR/stage3-q17-e9/final; n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
nodes=1; part=apex; steps=6000; batch=8; after=""; dry=0; extra=""; ev=1; asr=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --run) run=$2; shift 2 ;;  --layer) layer=$2; shift 2 ;;  --init) init=$2; shift 2 ;;  --n3d) n3d=$2; shift 2 ;;  --nodes) nodes=$2; shift 2 ;;
    --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --batch) batch=$2; shift 2 ;;  --after-job) after=$2; shift 2 ;;  --extra) extra=$2; shift 2 ;;
    --no-eval) ev=0; shift ;;  --dry-run) dry=1; shift ;;  --asr-eval) asr=1; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ -z $after || $after =~ ^[0-9]+$ ]] || { echo "--after-job 은 job id" >&2; exit 2; }
for f in $init/config.json $n3d; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; OUT=/soundai/Model/VAPASR/graft-$run
CORP=aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar
for c in swbd chime6tr nia23mt ami-far icsi-far notsofar-far chime6tr-far; do [[ -f $P2/$c.refined.dialogues.jsonl ]] && CORP+=",$c"; done
SIMS=""; for s in sim-dialogue-v2 sim-dialogue-v3-multi sim-dialogue-xl-v1 sim-dialogue-xl-tel-v1 sim-dialogue-v4-meet; do [[ -d $D/$s ]] && SIMS+="${SIMS:+,}$D/$s"; done
W="ami:3,icsi:2,notsofar:0.5,ami-far:3,icsi-far:2,notsofar-far:0.5,chime6tr:1,chime6tr-far:1,sim-dialogue-v3-multi:3,sim-dialogue-v4-meet:3,sim-dialogue-v2:1,sim-dialogue-xl-v1:0.5,sim-dialogue-xl-tel-v1:0.5"
ARGS="--init $init --n3d $n3d --graft-layer $layer --corpora $CORP --sim-data $SIMS --weights $W --noise-bank $D/noise-bank-v1"
ARGS="$ARGS --max-steps $steps --batch $batch --save-every 1000 --num-workers 4${extra:+ $extra}"
EXC="--exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4,slurmmxch200v5-hpc-50,slurmmxch200v5-hpc-51,slurmmxch200v5-hpc-101,slurmmxch200v5-hpc-102"
export MAX_RESTART=${MAX_RESTART:-$([[ $part == apex ]] && echo 5 || echo 20)}
cmd=(sbatch --job-name=SA_GRAFT --partition="$part" --nodes="$nodes" --time=24:00:00 $EXC ${after:+--dependency=afterok:$after}
     "--export=ALL,OUT=$OUT,SCRIPT=experiments/n3d_graft_train.py,MAX_RESTART=$MAX_RESTART,ARGS_EXTRA=${ARGS//,/%2C}" slurm/stage3-train-apex.sbatch)
echo "[$(date '+%F %T')] G1 이식 어댑터 graft-$run ← 인코더 $init · 이식 층 $layer · $nodes 노드 · $part · $steps step · 코퍼스 $CORP · 합성 $SIMS${after:+ · job $after 성공 뒤}" >&2
if (( dry )); then printf '%q ' "${cmd[@]}"; echo; echo "1 학습 job 0"; (( ev )) && GRAFT_RUN=$run bash slurm/submit-evals.sh q17-graft --after-job 0 --partition "$part" --dry-run | tail -1
  (( asr )) && U3_RUN=$run bash slurm/submit-evals.sh q17-u3-asr --after-job 0 --partition "$part" --dry-run | tail -2; exit 0; fi
[[ -e $OUT/DONE ]] && { echo "이미 끝났다: $OUT/DONE" >&2; exit 1; }
out=$("${cmd[@]}"); echo "$out" >&2; dep=$(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)
[[ -n $dep ]] || { echo "!! job 번호를 못 읽었다" >&2; exit 1; }
echo "1 학습 job $dep"
(( ev )) || exit 0
out=$(GRAFT_RUN=$run bash slurm/submit-evals.sh q17-graft --after-job "$dep" --partition "$part" 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 평가 제출 실패" >&2; exit 1; }
echo "2 평가 job $(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)"
(( asr )) || exit 0
out=$(U3_RUN=$run bash slurm/submit-evals.sh q17-u3-asr --after-job "$dep" --partition "$part" 2>&1) || { echo "$out" | tail -5 >&2; echo "!! ASR 지킴 평가 제출 실패" >&2; exit 1; }
echo "3 ASR 지킴 평가 job $(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)"
