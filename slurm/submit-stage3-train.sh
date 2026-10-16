#!/usr/bin/env bash
# Stage 3 학습 제출(사용자) — `bash slurm/submit-stage3-train.sh --run <이름> [옵션]`. 실패 재시도 한도는 서브셸로: `(MAX_RESTART=20 bash slurm/submit-stage3-train.sh …)`.
# 산출물 /soundai/Model/VAPASR/stage3-<이름>. 같은 --run 으로 다시 내면 마지막 checkpoint 에서 이어서 한다(설정이 다르면 학습 스크립트가 재개를 거부한다).
# 데이터:
#   대화(Stage 3)  Phase 2 정제 대화 7 코퍼스(71631·134-1·134-2·otoSpeech·KEC·ZTC·NSC), split v2 held-out 제외 — dlg_ko·dlg_en·dlg_de
#   replay(Stage 2) q06-s4-ast 와 같은 입력(EN·KO words 스냅숏·ASR 목록·CV4, 독일어 YODAS, 번역 타깃 2 묶음) — rep_ko·rep_en·rep_de
# 레시피(s4 와 같음): 인코더 학습([56,3] 고정)·SpecAugment·속도/잡음/음량 증강(replay)·varlen·GPU 당 16,384 토큰 예산·δ 4–10·lr 6e-5.
# 파일럿 옵션(2026-10-15 사용자 결정: 비율은 파일럿에서 비교해 정한다):
#   --weights W        셋 가중치(기본 dlg_ko:1,dlg_en:1,dlg_de:1,rep_ko:0.5,rep_en:0.5,rep_de:0.5 → 대화 2/3·replay 1/3 step)
#   --eot-mode M       soft(기본) | hard      --backchannel B   off(기본) | 1.0,3     --decision-weight W(기본 2)
#   --start-mode M     grid(기본) | mixed     --replay-files-frac F(기본 0.2 — replay 초기화 시간 절약)
#   --recover-p P      정체 회복 학습(대화 창마다 문맥 슬롯 한 구간을 뒤집을 확률, 기본 0 = 끔; 세부는 --extra "--recover-boundary-frac …")
#   --max-steps N(기본 2000)  --epochs E  --lr LR  --warmup N(기본 300)  --save-every N(기본 500)  --batch-tokens N(기본 16384)  --max-bs N(기본 64)
#   --nodes N(기본 1)  --partition P(기본 apex)  --time HH:MM:SS(기본 24:00:00)  --after JOBID  --init DIR(기본 q06-s4-ast/final)
#   --extra "..."      stage3_train.py 에 그대로 붙일 인자   --dry-run   sbatch 대신 명령만 출력
set -euo pipefail
cd "$(dirname "$0")/.."
D=/soundai/users/tskim/VAPKT-data/data/phase2
SNAP=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
CV4=/soundai/users/tskim/VAPKT-data/data/cv4-en-v1/lists/cv4-en-words-train.list
DE=/soundai/users/tskim/VAPKT-data/data/yodas-de-v1/lists/de-words-train-tn2.list
AST=/soundai/users/tskim/VAPKT-data/data/ast-mono-v1/targets-fix-gen-v3-bf16
AST_CV4=/soundai/users/tskim/VAPKT-data/data/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16
NB=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
run=""; init=/soundai/Model/VAPASR/semcommit-q06-s4-ast/final; nodes=1; partition=apex; time_limit=24:00:00; after=""; dry=0; extra=""
weights="dlg_ko:1,dlg_en:1,dlg_de:1,rep_ko:0.5,rep_en:0.5,rep_de:0.5"; eot_mode=soft; backchannel=off; decision_weight=2; start_mode=grid; frac=0.2; recover_p=0
max_steps=2000; epochs=1; lr=6e-5; warmup=300; save_every=500; batch_tokens=16384; max_bs=64
while [[ $# -gt 0 ]]; do
  case $1 in
    --run) run=$2; shift 2 ;;            --init) init=$2; shift 2 ;;          --nodes) nodes=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;; --time) time_limit=$2; shift 2 ;;  --after) after=$2; shift 2 ;;
    --weights) weights=$2; shift 2 ;;    --eot-mode) eot_mode=$2; shift 2 ;;  --backchannel) backchannel=$2; shift 2 ;;
    --decision-weight) decision_weight=$2; shift 2 ;; --start-mode) start_mode=$2; shift 2 ;; --replay-files-frac) frac=$2; shift 2 ;;
    --recover-p) recover_p=$2; shift 2 ;;
    --max-steps) max_steps=$2; shift 2 ;; --epochs) epochs=$2; shift 2 ;;    --lr) lr=$2; shift 2 ;;
    --warmup) warmup=$2; shift 2 ;;      --save-every) save_every=$2; shift 2 ;; --batch-tokens) batch_tokens=$2; shift 2 ;;
    --max-bs) max_bs=$2; shift 2 ;;      --extra) extra=$2; shift 2 ;;        --dry-run) dry=1; shift ;;
    *) echo "알 수 없는 인자: $1" >&2; exit 2 ;;
  esac
done
[[ $run =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--run 이름 필요(영숫자·._-)" >&2; exit 2; }
[[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo "--nodes 는 양의 정수" >&2; exit 2; }
[[ -z $after || $after =~ ^[0-9]+$ ]] || { echo "--after 는 job id" >&2; exit 2; }
[[ $eot_mode =~ ^(soft|hard)$ ]] || { echo "--eot-mode soft|hard" >&2; exit 2; }
[[ $backchannel =~ ^(off|[0-9.]+,[0-9]+)$ ]] || { echo "--backchannel off | 최대초,최대토큰" >&2; exit 2; }
[[ $start_mode =~ ^(grid|mixed)$ ]] || { echo "--start-mode grid|mixed" >&2; exit 2; }
[[ $recover_p =~ ^(0|0?\.[0-9]+|1|1\.0+)$ ]] || { echo "--recover-p 는 0–1" >&2; exit 2; }
[[ -f $init/config.json ]] || { echo "init 없음: $init" >&2; exit 1; }
for f in "$D/splits/v2.json" "$SNAP/main-words.list" "$ASR" "$CV4" "$DE" "$AST/pack/en-de/meta.json" "$AST_CV4/pack/en-de/meta.json" "$NB/noise.jsonl"; do
  [[ -s $f ]] || { echo "입력 없음·비어 있음: $f" >&2; exit 1; }
done
for c in aihub71631 aihub134-1 aihub134-2 otoSpeech kec ztc nsc; do [[ -s $D/$c.refined.dialogues.jsonl ]] || { echo "정제 대화 없음: $D/$c.refined.dialogues.jsonl" >&2; exit 1; }; done
out=/soundai/Model/VAPASR/stage3-$run
args="--init $init --data $D --split $D/splits/v2.json --weights $weights --eot-mode $eot_mode --backchannel $backchannel --decision-weight $decision_weight"
args+=" --start-mode $start_mode --recover-p $recover_p --replay-words @$SNAP/main-words.list --replay-words @$ASR --replay-words @$CV4 --replay-de-words @$DE --replay-files-frac $frac"
args+=" --ast-targets $AST,$AST_CV4 --ast-frac 0.8 --offline-frac 0.3 --gain-augment 0.5,-70,-20 --gain-augment-langs English"
args+=" --speed-perturb 0.9,1.0,1.1 --noise-bank $NB --train-encoder --lr-encoder 1e-5 --encoder-right-context 3 --encoder-context-sampling fixed --spec-augment 2,27,2,0.03"
args+=" --attn-impl varlen --batch-max-tokens $batch_tokens --pack-max-bs $max_bs --delays 4,5,6,7,8,9,10 --lr $lr --warmup $warmup --epochs $epochs --max-steps $max_steps"
args+=" --save-every $save_every --num-workers 4 $extra"
echo "Stage 3 run $run → $out · init $init · $nodes 노드 × 8 GPU · $partition · 가중치 $weights · EOT $eot_mode · 맞장구 $backchannel · 시작 $start_mode · 정체 회복 $recover_p · max-steps $max_steps · MAX_RESTART ${MAX_RESTART:-3}" >&2
[[ -d $out ]] && echo "이어서: $out 에 checkpoint $(ls -d "$out"/checkpoint-* 2>/dev/null | wc -l) 개" >&2
dep=""; [[ -n $after ]] && dep="--dependency=afterany:$after"
(( dry )) && sb="echo sbatch" || sb=sbatch
$sb --job-name=SA_SFT_FullDuplex --partition="$partition" $dep --nodes="$nodes" --time="$time_limit" \
  --export="ALL,OUT=$out,ARGS_EXTRA=${args//,/%2C}" slurm/stage3-train-apex.sbatch   # --export 는 쉼표로 변수를 가른다 → %2C 로 넘기고 sbatch 가 되돌린다
