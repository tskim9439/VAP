#!/usr/bin/env bash
# q06-s3-ast(2026-10-11, 사용자 결정): q06-s2-ast/final 에서 이어 번역 모드 학습량을 늘리고 영어 음량 증강을 넣는다.
#   근거: raw/sources/experiments/2026-10-11-q06-s2-ast-fleurs-ceiling
#     - FLEURS 교사 상한 대비 손실이 교사→학생(−2.7~−7.2 BLEU)과 스트리밍(−4.3~−6.6)에서 크다 → 번역 모드 비율 0.5 → 0.8.
#     - FLEURS 영어처럼 아주 작은 입력(RMS 약 −60 dBFS)에서 스트리밍이 침묵한다 → 영어 항목의 50 % 를 RMS [−70, −20] dBFS 로 맞춘다
#       (vapasr/data/ast_dataset.apply_gain, 잡음·잔향 뒤 이득만).
#   데이터·나머지 레시피는 q06-s2-ast(slurm/submit-q06-s2-ast.sh)와 같다. 번역 모드에서도 전사 흐름을 함께 내므로 ASR 학습은 계속된다.
# 추가 인자는 submit-semcommit-train-apex.sh 에 그대로 넘어가고 같은 옵션은 뒤의 값이 이긴다. 예:
#   bash slurm/submit-q06-s3-ast.sh --dry-run                                   # sbatch 없이 명령만 출력
#   bash slurm/submit-q06-s3-ast.sh --epochs 2 --nodes 2 --partition hpc       # 제출
#   같은 --run 으로 다시 내면 마지막 checkpoint 에서 이어서 한다.
#   주의: --extra 를 다시 주면 아래 --extra(인코더 고정·worker 수·AST·음량 증강 옵션)가 통째로 바뀐다.
set -euo pipefail
cd "$(dirname "$0")/.."
SNAP=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
NB=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
DE=/soundai/users/tskim/VAPKT-data/data/yodas-de-v1/lists/de-words-train-tn2.list
AST=/soundai/users/tskim/VAPKT-data/data/ast-mono-v1/targets-fix-gen-v3-bf16
INIT=/soundai/Model/VAPASR/semcommit-q06-s2-ast/final
[[ -f "$AST/pack/en-de/meta.json" && -f "$AST/pack/ko-en/meta.json" ]] || { echo "번역 타깃 묶음 없음: $AST/pack (ast_build_targets.py --pack-only)" >&2; exit 1; }
exec bash slurm/submit-semcommit-train-apex.sh --run q06-s3-ast --snap "$SNAP" --init "$INIT" \
  --asr-only --offline-frac 0.3 --asr-list "$ASR" --de-list "$DE" \
  --train-encoder --spec-augment 2,27,2,0.03 --speed-perturb 0.9,1.0,1.1 --noise-bank "$NB" --varlen \
  --delays 4,5,6,7,8,9,10 --batch-tokens 16384 --lr 6e-5 --warmup 500 --epochs 2 --save-every 1000 \
  --nodes 2 --extra "--encoder-right-context 3 --encoder-context-sampling fixed --num-workers 4 --ast-targets $AST --ast-frac 0.8 --gain-augment 0.5,-70,-20 --gain-augment-langs English" "$@"
