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
#   --spk-act-weight W 화자 슬롯 활동 보조 손실 가중(기본 0 = 끔; 청크마다 슬롯 A/B 가 말하는 중인지 BCE)
#   --sim-data DIR[,DIR] 합성 2 인 대화 폴더(sim_build_dialogues 산출, 쉼표로 여럿) — --weights 에 sim_ko·sim_en·sim_de, 교차 언어 sim_en-de·sim_en-ko 를 넣어 쓴다
#   Stage 4(같은 스크립트, --init 은 Stage 3 최종): --dlg-trans DIR[,DIR](대화 번역 타깃) --dlg-trans-input DIR(되살린 발화 번역, 입력만)
#                      --dlg-ast-frac F(대화 창 번역 모드 비율, 기본 0.5) --sim-trans-frac F(교차 언어 합성 창 번역 모드 비율, 기본 0.8)
#                      --trans-lead-space(번역 행마다 앞 공백 — 행을 이은 번역 흐름에서 단어가 붙지 않게, Stage 4 본 학습부터)
#   --phone-aug P      전화 대역 증강 확률(대화 창·단일 화자 모두, 기본 0 = 끔)   --replay-set 이름:언어:@목록  따로 가중하는 단일 화자 셋(반복 가능, --weights 에 이름)
#   V2: --max-speakers N(2–4, 기본 2) --declare-all-p P(기본 0.5) --window lo,hi(기본 20,40) --window-draws R(기본 0 = 격자 창)
#       --dlg-corpora 목록(기본 7 코퍼스; 회의 ami·icsi·notsofar 는 --weights 의 dlg_mt 로, --max-speakers 3 이상과 함께)
#   --split NAME       held-out split(기본 v2; V2 는 v4 — 회의 공식 dev·eval 을 학습에서 뺀다)
#   --enc-tap-layer L  V2-a 두 스트림 입력(인코더 L 층 → adapter_tap → LLM 입력에 더함, 기본 −1 = 끔). init 에 탭이 있으면 같은 층
#   --revive-dir DIR   격리 발화 되살리기 재료(stage3_revive_tokens 산출 <corpus>.revive.jsonl) — asr_disagree 발화의 화자 태그·<EOT> 학습
#   --exclude N,…      더 뺄 노드(hpc-102 처럼 짧게 써도 된다) — sbatch 기본 제외 hpc-52·hpc-4 에 더한다
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
weights="dlg_ko:1,dlg_en:1,dlg_de:1,rep_ko:0.5,rep_en:0.5,rep_de:0.5"; eot_mode=soft; backchannel=off; decision_weight=2; start_mode=grid; frac=0.2; recover_p=0; spk_act=0; sim_data=""; revive_dir=""; exclude=""; dlg_trans=""; dlg_trans_input=""; dlg_ast_frac=0.5; sim_trans_frac=0.8; trans_lead_space=0; phone_aug=0; replay_sets=""; enc_tap=-1; split=v2; max_spk=2; declare_p=0.5; window=20,40; draws=0; dlg_corpora=aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc
max_steps=2000; epochs=1; lr=6e-5; warmup=300; save_every=500; batch_tokens=16384; max_bs=64
while [[ $# -gt 0 ]]; do
  case $1 in
    --run) run=$2; shift 2 ;;            --init) init=$2; shift 2 ;;          --nodes) nodes=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;; --time) time_limit=$2; shift 2 ;;  --after) after=$2; shift 2 ;;
    --weights) weights=$2; shift 2 ;;    --eot-mode) eot_mode=$2; shift 2 ;;  --backchannel) backchannel=$2; shift 2 ;;
    --decision-weight) decision_weight=$2; shift 2 ;; --start-mode) start_mode=$2; shift 2 ;; --replay-files-frac) frac=$2; shift 2 ;;
    --recover-p) recover_p=$2; shift 2 ;;  --spk-act-weight) spk_act=$2; shift 2 ;;  --sim-data) sim_data=$2; shift 2 ;;
    --revive-dir) revive_dir=$2; shift 2 ;;  --exclude) exclude=$2; shift 2 ;;
    --trans-lead-space) trans_lead_space=1; shift ;;
    --phone-aug) phone_aug=$2; shift 2 ;;  --enc-tap-layer) enc_tap=$2; shift 2 ;;
    --split) split=$2; shift 2 ;;  --max-speakers) max_spk=$2; shift 2 ;;  --declare-all-p) declare_p=$2; shift 2 ;;  --window) window=$2; shift 2 ;;  --window-draws) draws=$2; shift 2 ;;  --dlg-corpora) dlg_corpora=$2; shift 2 ;;  --replay-set) replay_sets+=" --replay-set $2"; shift 2 ;;
    --dlg-trans) dlg_trans=$2; shift 2 ;;  --dlg-trans-input) dlg_trans_input=$2; shift 2 ;;  --dlg-ast-frac) dlg_ast_frac=$2; shift 2 ;;  --sim-trans-frac) sim_trans_frac=$2; shift 2 ;;
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
[[ $spk_act =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]] || { echo "--spk-act-weight 는 0 이상의 수" >&2; exit 2; }
[[ $enc_tap =~ ^(-1|[0-9]+)$ ]] || { echo "--enc-tap-layer 는 -1 또는 층 번호" >&2; exit 2; }
[[ $max_spk =~ ^[2-4]$ ]] || { echo "--max-speakers 는 2–4" >&2; exit 2; }
[[ $window =~ ^[0-9.]+,[0-9.]+$ && $draws =~ ^[0-9]+$ ]] || { echo "--window lo,hi · --window-draws 정수" >&2; exit 2; }
[[ -f $init/config.json ]] || { echo "init 없음: $init" >&2; exit 1; }
for sd in ${sim_data//,/ }; do [[ -d $sd/mono ]] || { echo "--sim-data 에 mono/ 가 없다: $sd" >&2; exit 1; }; done
for td in ${dlg_trans//,/ } ${dlg_trans_input//,/ }; do [[ -d $td/pack ]] || { echo "번역 타깃에 pack/ 이 없다: $td" >&2; exit 1; }; done
[[ -z $dlg_trans_input || -n $dlg_trans ]] || { echo "--dlg-trans-input 은 --dlg-trans 와 함께" >&2; exit 2; }
[[ -z $revive_dir ]] || ls "$revive_dir"/*.revive.jsonl >/dev/null 2>&1 || { echo "--revive-dir 에 *.revive.jsonl 이 없다: $revive_dir" >&2; exit 1; }
[[ $split =~ ^v[0-9]+$ ]] || { echo "--split 은 v2·v4 처럼" >&2; exit 2; }
for f in "$D/splits/$split.json" "$SNAP/main-words.list" "$ASR" "$CV4" "$DE" "$AST/pack/en-de/meta.json" "$AST_CV4/pack/en-de/meta.json" "$NB/noise.jsonl"; do
  [[ -s $f ]] || { echo "입력 없음·비어 있음: $f" >&2; exit 1; }
done
for c in ${dlg_corpora//,/ }; do [[ -s $D/$c.refined.dialogues.jsonl ]] || { echo "정제 대화 없음: $D/$c.refined.dialogues.jsonl" >&2; exit 1; }; done
out=/soundai/Model/VAPASR/stage3-$run
args="--init $init --data $D --split $D/splits/$split.json --weights $weights --eot-mode $eot_mode --backchannel $backchannel --decision-weight $decision_weight"
args+=" --start-mode $start_mode --recover-p $recover_p --spk-act-weight $spk_act --replay-words @$SNAP/main-words.list --replay-words @$ASR --replay-words @$CV4 --replay-de-words @$DE --replay-files-frac $frac"
args+=" --ast-targets $AST,$AST_CV4 --ast-frac 0.8 --offline-frac 0.3 --gain-augment 0.5,-70,-20 --gain-augment-langs English"
args+=" --speed-perturb 0.9,1.0,1.1 --noise-bank $NB --train-encoder --lr-encoder 1e-5 --encoder-right-context 3 --encoder-context-sampling fixed --spec-augment 2,27,2,0.03"
args+=" --attn-impl varlen --batch-max-tokens $batch_tokens --pack-max-bs $max_bs --delays 4,5,6,7,8,9,10 --lr $lr --warmup $warmup --epochs $epochs --max-steps $max_steps"
args+=" --save-every $save_every --num-workers 4 ${sim_data:+--sim-data $sim_data} ${revive_dir:+--revive-dir $revive_dir}"
[[ -n $dlg_trans ]] && args+=" --dlg-trans $dlg_trans --dlg-ast-frac $dlg_ast_frac ${dlg_trans_input:+--dlg-trans-input $dlg_trans_input}"
[[ $weights =~ sim_[a-z][a-z]-[a-z][a-z] ]] && args+=" --sim-trans-frac $sim_trans_frac"
[[ $trans_lead_space == 1 ]] && args+=" --trans-lead-space"
[[ $phone_aug != 0 ]] && args+=" --phone-aug $phone_aug"
[[ $enc_tap != -1 ]] && args+=" --enc-tap-layer $enc_tap"
args+=" --dlg-corpora $dlg_corpora --window $window --max-speakers $max_spk"; [[ $max_spk != 2 ]] && args+=" --declare-all-p $declare_p"; [[ $draws != 0 ]] && args+=" --window-draws $draws"
args+="$replay_sets"
for rs in $replay_sets; do [[ $rs == --replay-set ]] && continue; n=${rs%%:*}; [[ $weights == *"$n:"* ]] || { echo "--replay-set $n 이 --weights 에 없다" >&2; exit 2; }; f=${rs#*:*:}; f=${f#@}; [[ -f $f ]] || { echo "--replay-set 목록 없음: $f" >&2; exit 1; }; done
args+=" $extra"
echo "Stage 3 run $run → $out · init $init · $nodes 노드 × 8 GPU · $partition · 가중치 $weights · EOT $eot_mode · 맞장구 $backchannel · 시작 $start_mode · 정체 회복 $recover_p · 화자 슬롯 헤드 $spk_act · 합성 ${sim_data:-없음} · 되살리기 ${revive_dir:-없음} · 대화 번역 ${dlg_trans:-없음}${dlg_trans:+(번역 모드 $dlg_ast_frac)} · 번역 행 앞 공백 $trans_lead_space · 전화 대역 $phone_aug · 인코더 탭 $enc_tap · split $split · 화자 상한 $max_spk · 창 $window(draw $draws) · 대화 코퍼스 $dlg_corpora · 따로 가중 셋${replay_sets:- 없음} · 제외 추가 ${exclude:-없음} · max-steps $max_steps · MAX_RESTART ${MAX_RESTART:-3}" >&2
[[ -d $out ]] && echo "이어서: $out 에 checkpoint $(ls -d "$out"/checkpoint-* 2>/dev/null | wc -l) 개" >&2
dep=""; [[ -n $after ]] && dep="--dependency=afterany:$after"
exc=""; if [[ -n $exclude ]]; then exc="--exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4"; for n in ${exclude//,/ }; do [[ $n == slurmmxch200v5-* ]] || n=slurmmxch200v5-$n; exc+=",$n"; done; fi   # 명령줄 --exclude 는 sbatch 안 기본값을 덮으므로 기본 제외도 함께 쓴다
(( dry )) && sb="echo sbatch" || sb=sbatch
$sb --job-name=SA_SFT_FullDuplex --partition="$partition" $dep $exc --nodes="$nodes" --time="$time_limit" \
  --export="ALL,OUT=$out,ARGS_EXTRA=${args//,/%2C}" slurm/stage3-train-apex.sbatch   # --export 는 쉼표로 변수를 가른다 → %2C 로 넘기고 sbatch 가 되돌린다
