#!/usr/bin/env bash
# semcommit 학습 제출 — 스케줄러 셸에서 `bash slurm/submit-semcommit-train-apex.sh --run <이름> [옵션]`.
# 라벨링 중간 스냅숏(끝난 파트만, semcommit_collect_done.py)으로 다중 노드 DDP 학습. 입력은 이 스크립트의 지역 변수이고 sbatch --export 로만
# job 에 전달된다(제출 셸 환경에 남지 않는다). 같은 --run 으로 다시 제출하면 checkpoint-N 에서 이어서 한다.
#   --run NAME         산출물 /soundai/Model/VAPASR/semcommit-<NAME> (필수)
#   --snap DIR         스냅숏(기본: 라벨 루트의 가장 최근 snapshots/*)
#   --init DIR         초기 HF 체크포인트(기본 /soundai/Model/VAPASR/hf-E2/final — semcommit recipe v0.3 과 같다)
#   --nodes N          노드 수(노드당 8 GPU, 기본 1)   --time HH:MM:SS(기본 24:00:00)   --partition P(기본 apex)   --job-name NAME
#   --after JOBID      그 job 이 끝난 뒤 시작(afterany)
#   --batch-tokens N   GPU 당 동적 배치 예산(샘플 수 × 최장 길이 ≤ N, 기본 32768)   --max-bs N(기본 256)
#   --epochs E(기본 2)  --lr LR(기본 6e-5)  --warmup N(기본 50)  --save-every N(기본 200)  --max-steps N(기본 0 = epochs)
#   참고(2026-09-28 스냅숏 1,274 파트, 1 노드): 32k 예산이면 GPU 당 ≈110 스트림/step → 8 GPU ≈900 스트림/step, epoch ≈420 step.
#   KO 오디오(databricks tar) 읽기가 병목(단일 GPU 스모크 4.6 s/step, GPU 계산 ≈1.1 s) — 노드를 늘려도 저장소 처리량 이상은 빨라지지 않는다.
#   --delays LIST      학습 δ(지연 청크) 목록, 기본 2,3,4,6(hf-E2 와 같음). 예: 2,3,4,6,8 — <DELAY_1..8> 은 어휘에 있고, E2 가 배우지 않은 δ 는
#                      이 학습에서 처음 배운다. 샘플마다 목록에서 고르게 뽑고, 꼬리 패딩은 가장 큰 δ 에 맞춘다.
#   --extra "..."      semcommit_train.py 에 그대로 붙일 인자
set -euo pipefail
cd "$(dirname "$0")/.."
LABELS=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035
run=""; snap=""; init=/soundai/Model/VAPASR/hf-E2/final; nodes=1; time_limit=24:00:00; partition=apex; job_name=SA_SFT_FullDuplex
batch_tokens=32768; max_bs=256; epochs=2; lr=6e-5; warmup=50; save_every=200; max_steps=0; extra=""; delays=2,3,4,6; after=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) run=$2; shift 2 ;;            --snap) snap=$2; shift 2 ;;          --init) init=$2; shift 2 ;;
    --nodes) nodes=$2; shift 2 ;;        --time) time_limit=$2; shift 2 ;;    --partition) partition=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;  --batch-tokens) batch_tokens=$2; shift 2 ;; --max-bs) max_bs=$2; shift 2 ;;
    --epochs) epochs=$2; shift 2 ;;      --lr) lr=$2; shift 2 ;;              --warmup) warmup=$2; shift 2 ;;
    --save-every) save_every=$2; shift 2 ;; --max-steps) max_steps=$2; shift 2 ;; --extra) extra=$2; shift 2 ;;
    --delays) delays=$2; shift 2 ;;      --after) after=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--run 이름 필요(영숫자·._-)" >&2; exit 2; }
[[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo "--nodes 는 양의 정수" >&2; exit 2; }
[[ -z $after || $after =~ ^[0-9]+$ ]] || { echo "--after 는 job id" >&2; exit 2; }
[[ $delays =~ ^[1-8](,[1-8])*$ ]] || { echo "--delays 는 1–8 의 쉼표 목록(<DELAY_1..8>)" >&2; exit 2; }
snap=${snap:-$(ls -d "$LABELS"/snapshots/*/ 2>/dev/null | sort | tail -1)}; snap=${snap%/}
for f in main-words.list main-labels.list short-words.list short-labels.list summary.json; do
  [[ -f $snap/$f ]] || { echo "스냅숏에 없음: $snap/$f" >&2; exit 1; }
done
[[ -f $init/config.json ]] || { echo "init 없음: $init" >&2; exit 1; }
out=/soundai/Model/VAPASR/semcommit-$run
args="--delays $delays --batch-max-tokens $batch_tokens --pack-max-bs $max_bs --epochs $epochs --lr $lr --warmup $warmup --save-every $save_every --max-steps $max_steps --num-workers 8 $extra"
echo "δ $delays · 스냅숏 $snap ($(python3 -c "import json,sys;s=json.load(open(sys.argv[1]));print(s['parts'],'파트')" "$snap/summary.json")) · init $init · 산출물 $out · $nodes 노드 × 8 GPU" >&2
[[ -d $out ]] && echo "이어서: $out 에 checkpoint $(ls -d "$out"/checkpoint-* 2>/dev/null | wc -l) 개" >&2
sbatch --job-name="$job_name" --partition="$partition" ${after:+--dependency=afterany:$after} --nodes="$nodes" --time="$time_limit" \
  --export="ALL,OUT=$out,SNAP=$snap,INIT=$init,ARGS_EXTRA=$args" slurm/semcommit-train-apex.sbatch
