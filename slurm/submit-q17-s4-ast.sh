#!/usr/bin/env bash
# q17-s4-ast(2026-10-22, 사용자 요청 — Qwen3-ASR 1.7B 체인 slurm/submit-q17-chain.sh): q06-s4-ast 와 같은 레시피를 1.7B 로. init 은 q17-s3-ast/final, 데이터·하이퍼파라미터는 q06 판과 같다(run 이름·init 만 다르다).
# ── 아래는 q06 판 설명 그대로 ──
# q06-s4-ast(2026-10-12): q06-s3-ast/final 에서 이어 Common Voice 4 영어 낭독(정렬 351 h 중 train)과 그 단조 번역 타깃(영→독·영→한)을 더한다.
#   근거: s3 평가(wiki output-q06-s3-ast-eval-20261012)와 베이스라인 비교에서 영→독·영→한 스트리밍 번역이 가장 약하다
#     (SeamlessStreaming 대비 −6~7 BLEU, 짧은 문장의 번역 누락). 영어 원천을 늘려 영→X 번역 항목을 약 50 % 더한다.
#   더하는 것:
#     - ASR words: cv4-en-v1/lists/cv4-en-words-train.list(232,845 행, experiments/cv_en_align.py) → 영어 셋 'en' 에 SEM 중립 행으로
#     - 번역 타깃: ast-mono-cv4en-v1/targets-fix-gen-v3-bf16(영→독 230,083 · 영→한 228,474 행, 묶은 형태 pack/)
#   나머지(데이터·번역 모드 0.8·영어 음량 증강·레시피)는 q06-s3-ast(slurm/submit-q06-s3-ast.sh)와 같다.
# 추가 인자는 submit-semcommit-train-apex.sh 에 그대로 넘어가고 같은 옵션은 뒤의 값이 이긴다. 예:
#   bash slurm/submit-q17-s4-ast.sh --dry-run                                   # sbatch 없이 명령만 출력
#   bash slurm/submit-q17-s4-ast.sh --epochs 2 --nodes 2 --partition hpc       # 제출
#   같은 --run 으로 다시 내면 마지막 checkpoint 에서 이어서 한다.
#   주의: --extra 를 다시 주면 아래 --extra(인코더 고정·worker 수·AST·음량 증강·CV4 옵션)가 통째로 바뀐다.
set -euo pipefail
cd "$(dirname "$0")/.."
SNAP=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
NB=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
DE=/soundai/users/tskim/VAPKT-data/data/yodas-de-v1/lists/de-words-train-tn2.list
AST=/soundai/users/tskim/VAPKT-data/data/ast-mono-v1/targets-fix-gen-v3-bf16
CV4=/soundai/users/tskim/VAPKT-data/data/cv4-en-v1/lists/cv4-en-words-train.list
AST_CV4=/soundai/users/tskim/VAPKT-data/data/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16
INIT=/soundai/Model/VAPASR/semcommit-q17-s3-ast/final
[[ -f "$AST/pack/en-de/meta.json" && -f "$AST/pack/ko-en/meta.json" ]] || { echo "번역 타깃 묶음 없음: $AST/pack (ast_build_targets.py --pack-only)" >&2; exit 1; }
[[ -f "$AST_CV4/pack/en-de/meta.json" && -f "$AST_CV4/pack/en-ko/meta.json" ]] || { echo "CV4 번역 타깃 묶음 없음: $AST_CV4/pack" >&2; exit 1; }
[[ -s "$CV4" ]] || { echo "CV4 영어 words 목록 없음·비어 있음: $CV4" >&2; exit 1; }
exec bash slurm/submit-semcommit-train-apex.sh --run q17-s4-ast --snap "$SNAP" --init "$INIT" \
  --asr-only --offline-frac 0.3 --asr-list "$ASR" --de-list "$DE" \
  --train-encoder --spec-augment 2,27,2,0.03 --speed-perturb 0.9,1.0,1.1 --noise-bank "$NB" --varlen \
  --delays 4,5,6,7,8,9,10 --batch-tokens 16384 --lr 6e-5 --warmup 500 --epochs 2 --save-every 1000 \
  --nodes 2 --extra "--encoder-right-context 3 --encoder-context-sampling fixed --num-workers 4 --asr-words @$CV4 --ast-targets $AST,$AST_CV4 --ast-frac 0.8 --gain-augment 0.5,-70,-20 --gain-augment-langs English" "$@"
