#!/usr/bin/env bash
# semcommit 학습 제출 — 스케줄러 셸에서 `bash slurm/submit-semcommit-train-apex.sh --run <이름> [옵션]`.
# 라벨링 중간 스냅숏(끝난 파트만, semcommit_collect_done.py)으로 다중 노드 DDP 학습. 입력은 이 스크립트의 지역 변수이고 sbatch --export 로만
# job 에 전달된다(제출 셸 환경에 남지 않는다). 같은 --run 으로 다시 제출하면 checkpoint-N 에서 이어서 한다.
#   --run NAME         산출물 /soundai/Model/VAPASR/semcommit-<NAME> (필수)
#   --snap DIR         스냅숏(기본: 라벨 루트의 가장 최근 snapshots/*)
#   --init DIR         초기 HF 체크포인트(기본 /soundai/Model/VAPASR/hf-E2/final — semcommit recipe v0.3 과 같다)
#   --nodes N          노드 수(노드당 8 GPU, 기본 1)   --time HH:MM:SS(기본 24:00:00)   --partition P(기본 apex)   --job-name NAME
#   --after JOBID      그 job 이 끝난 뒤 시작(afterany)
#   --afterok JOBID    그 job 이 성공한 뒤에만 시작(afterok) — 그 job 이 --init 을 만들 예정이면 init 존재 검사를 건너뛴다
#   --dry-run          sbatch 대신 명령만 출력("Submitted batch job 0" 형식으로 끝낸다)
#   --batch-tokens N   GPU 당 동적 배치 예산(샘플 수 × 최장 길이 ≤ N, 기본 32768)   --max-bs N(기본 256)
#   --epochs E(기본 2)  --lr LR(기본 6e-5)  --warmup N(기본 50)  --save-every N(기본 200)  --max-steps N(기본 0 = epochs)
#   참고(2026-09-28 스냅숏 1,274 파트, 1 노드): 32k 예산이면 GPU 당 ≈110 스트림/step → 8 GPU ≈900 스트림/step, epoch ≈420 step.
#   KO 오디오(databricks tar) 읽기가 병목(단일 GPU 스모크 4.6 s/step, GPU 계산 ≈1.1 s) — 노드를 늘려도 저장소 처리량 이상은 빨라지지 않는다.
#   --delays LIST      학습 δ(지연 청크) 목록, 기본 2,3,4,6(hf-E2 와 같음). 예: 2,3,4,6,8 — <DELAY_1..8> 은 어휘에 있고, E2 가 배우지 않은 δ 는
#                      이 학습에서 처음 배운다. 샘플마다 목록에서 고르게 뽑고, 꼬리 패딩은 가장 큰 δ 에 맞춘다.
#   --extra "..."      semcommit_train.py 에 그대로 붙일 인자
#   ASR 정확도 옵션(2026-09-29, 기본은 모두 끔 — 켜면 지문이 달라져 새 --run 이 필요하다):
#   --train-encoder    인코더도 학습(--lr-encoder 1e-5, E2 레시피). 활성화 메모리가 커서 --batch-tokens 를 함께 줄인다(스모크 실측 참고)
#   --spec-augment X   off | light(주파수 2×27·시간 5×0.05) | nemo(2×27·10×0.05) | Fm,Fw,Tm,Tw — 입력 log-mel 마스크(학습 중에만)
#   --speed-perturb L  배율 목록, 예 0.9,1.0,1.1 — 오디오 리샘플 + 이벤트 시각 1/배율
#   --asr-list FILE    SEM 중립 ASR words 목록(semcommit_asr_words_list.py) → --asr-words @FILE,  --asr-max-ratio R(셋마다 ≤ R × 라벨 항목)
#   --varlen           --attn-impl varlen(torch FA2 varlen; 처리량 +22–36 %, 4430c94)
#   단계 학습(2026-10-01):
#   --qwen-dir DIR     --init 대신 Qwen3-ASR(0.6B·1.7B)에서 새로 시작(adapter random, 출력 차원 = thinker hidden)
#   --asr-only         Stage 1: labels 무시, <SEM_END> 학습 안 함   --offline-frac F  final(오프라인) 모드 항목 비율(스트리밍과 함께)
#   --freeze-thinker   Stage 0 adapter 워밍업(--offline-frac 1 과 함께)
#   --noise-bank DIR   잡음·잔향 증강(experiments/build_noise_bank.py 뱅크; 확률·SNR 은 --extra "--noise-p 0.4 --noise-snr 5,30 --rir-p 0.2")
set -euo pipefail
cd "$(dirname "$0")/.."
LABELS=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035
run=""; snap=""; init=/soundai/Model/VAPASR/hf-E2/final; nodes=1; time_limit=24:00:00; partition=apex; job_name=SA_SFT_FullDuplex
batch_tokens=32768; max_bs=256; epochs=2; lr=6e-5; warmup=50; save_every=200; max_steps=0; extra=""; delays=2,3,4,6; after=""
train_encoder=0; spec_augment=off; speed_perturb=""; asr_list=""; asr_max_ratio=""; varlen=0; noise_bank=""; qwen_dir=""; asr_only=0; offline_frac=0; freeze_thinker=0; afterok=""; dry=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) run=$2; shift 2 ;;            --snap) snap=$2; shift 2 ;;          --init) init=$2; shift 2 ;;
    --nodes) nodes=$2; shift 2 ;;        --time) time_limit=$2; shift 2 ;;    --partition) partition=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;  --batch-tokens) batch_tokens=$2; shift 2 ;; --max-bs) max_bs=$2; shift 2 ;;
    --epochs) epochs=$2; shift 2 ;;      --lr) lr=$2; shift 2 ;;              --warmup) warmup=$2; shift 2 ;;
    --save-every) save_every=$2; shift 2 ;; --max-steps) max_steps=$2; shift 2 ;; --extra) extra=$2; shift 2 ;;
    --delays) delays=$2; shift 2 ;;      --after) after=$2; shift 2 ;;
    --train-encoder) train_encoder=1; shift ;;  --spec-augment) spec_augment=$2; shift 2 ;;  --speed-perturb) speed_perturb=$2; shift 2 ;;
    --asr-list) asr_list=$2; shift 2 ;;  --asr-max-ratio) asr_max_ratio=$2; shift 2 ;;  --varlen) varlen=1; shift ;;
    --noise-bank) noise_bank=$2; shift 2 ;;
    --afterok) afterok=$2; shift 2 ;;  --dry-run) dry=1; shift ;;
    --qwen-dir) qwen_dir=$2; shift 2 ;;  --asr-only) asr_only=1; shift ;;  --offline-frac) offline_frac=$2; shift 2 ;;  --freeze-thinker) freeze_thinker=1; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--run 이름 필요(영숫자·._-)" >&2; exit 2; }
[[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo "--nodes 는 양의 정수" >&2; exit 2; }
[[ -z $after || $after =~ ^[0-9]+$ ]] || { echo "--after 는 job id" >&2; exit 2; }
[[ -z $afterok || $afterok =~ ^[0-9]+$ ]] || { echo "--afterok 는 job id" >&2; exit 2; }
[[ -z $after || -z $afterok ]] || { echo "--after 와 --afterok 는 함께 쓰지 않는다" >&2; exit 2; }
[[ $delays =~ ^[1-8](,[1-8])*$ ]] || { echo "--delays 는 1–8 의 쉼표 목록(<DELAY_1..8>)" >&2; exit 2; }
[[ $spec_augment =~ ^(off|light|nemo|[0-9]+,[0-9]+,[0-9]+,[0-9.]+)$ ]] || { echo "--spec-augment 는 off|light|nemo|Fm,Fw,Tm,Tw" >&2; exit 2; }
[[ -z $speed_perturb || $speed_perturb =~ ^[0-9.]+(,[0-9.]+)*$ ]] || { echo "--speed-perturb 는 배율 쉼표 목록" >&2; exit 2; }
[[ -z $asr_list || -f $asr_list ]] || { echo "--asr-list 파일 없음: $asr_list" >&2; exit 1; }
[[ -z $noise_bank || ( -f $noise_bank/noise.jsonl && -f $noise_bank/rir.npy ) ]] || { echo "--noise-bank 뱅크 불완전: $noise_bank" >&2; exit 1; }
snap=${snap:-$(ls -d "$LABELS"/snapshots/*/ 2>/dev/null | sort | tail -1)}; snap=${snap%/}
for f in main-words.list main-labels.list short-words.list short-labels.list summary.json; do
  [[ -f $snap/$f ]] || { echo "스냅숏에 없음: $snap/$f" >&2; exit 1; }
done
if [[ -n $qwen_dir ]]; then [[ -f $qwen_dir/config.json ]] || { echo "Qwen3-ASR 없음: $qwen_dir" >&2; exit 1; }; init=qwen
elif [[ -n $afterok ]]; then [[ -f $init/config.json ]] || echo "init 은 job $afterok 가 만들 예정: $init" >&2
else [[ -f $init/config.json ]] || { echo "init 없음: $init" >&2; exit 1; }; fi
[[ $offline_frac =~ ^(0|1|0?\.[0-9]+|1\.0+)$ ]] || { echo "--offline-frac 은 [0, 1]" >&2; exit 2; }
out=/soundai/Model/VAPASR/semcommit-$run
args="--delays $delays --batch-max-tokens $batch_tokens --pack-max-bs $max_bs --epochs $epochs --lr $lr --warmup $warmup --save-every $save_every --max-steps $max_steps --num-workers 8"
(( train_encoder )) && args+=" --train-encoder --lr-encoder 1e-5"
[[ $spec_augment != off ]] && args+=" --spec-augment $spec_augment"
[[ -n $speed_perturb ]] && args+=" --speed-perturb $speed_perturb"
[[ -n $asr_list ]] && args+=" --asr-words @$asr_list" && [[ -n $asr_max_ratio ]] && args+=" --asr-max-ratio $asr_max_ratio"
(( varlen )) && args+=" --attn-impl varlen"
[[ -n $noise_bank ]] && args+=" --noise-bank $noise_bank"
[[ -n $qwen_dir ]] && args+=" --qwen-dir $qwen_dir"
(( asr_only )) && args+=" --asr-only"
[[ $offline_frac != 0 ]] && args+=" --offline-frac $offline_frac"
(( freeze_thinker )) && args+=" --freeze-thinker"
args+=" $extra"
echo "ASR 옵션: 인코더 $([[ $train_encoder = 1 ]] && echo 학습 || echo 동결) · SpecAugment $spec_augment · 속도 ${speed_perturb:-끔} · ASR 목록 ${asr_list:-없음} · varlen $varlen · 잡음 뱅크 ${noise_bank:-없음} · init ${qwen_dir:-$init} · ASR 전용 $asr_only · final 비율 $offline_frac · thinker 동결 $freeze_thinker" >&2
echo "δ $delays · 스냅숏 $snap ($(python3 -c "import json,sys;s=json.load(open(sys.argv[1]));print(s['parts'],'파트')" "$snap/summary.json")) · init $init · 산출물 $out · $nodes 노드 × 8 GPU" >&2
[[ -d $out ]] && echo "이어서: $out 에 checkpoint $(ls -d "$out"/checkpoint-* 2>/dev/null | wc -l) 개" >&2
dep=""; [[ -n $after ]] && dep="--dependency=afterany:$after"; [[ -n $afterok ]] && dep="--dependency=afterok:$afterok"
(( dry )) && sb="echo sbatch" || sb=sbatch
$sb --job-name="$job_name" --partition="$partition" $dep --nodes="$nodes" --time="$time_limit" \
  --export="ALL,OUT=$out,SNAP=$snap,INIT=$init,ARGS_EXTRA=${args//,/%2C}" slurm/semcommit-train-apex.sbatch   # --export 는 쉼표로 변수를 가른다(job 76482: '--delays 2,3,…' 가 '--delays 2' 로 잘림) → %2C 로 넘기고 sbatch 가 되돌린다
(( dry )) && echo "Submitted batch job 0"
exit 0
