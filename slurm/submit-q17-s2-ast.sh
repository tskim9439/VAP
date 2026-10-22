#!/usr/bin/env bash
# q17-s2-ast(2026-10-22, 사용자 요청 — Qwen3-ASR 1.7B 체인 slurm/submit-q17-chain.sh): q06-s2-ast 와 같은 레시피를 1.7B 로. init 은 q17-s1m3-d10/final, 데이터·하이퍼파라미터는 q06 판과 같다(run 이름·init 만 다르다).
# ── 아래는 q06 판 설명 그대로 ──
# 다음 단계(ast-v1): 화자·언어 prefix + 전사·번역 함께 출력 — 0.6B Stage 1 최종(q06-s1m3-d10/final)에서 시작(2026-10-10, 사용자 결정).
#   데이터: Stage 1 과 같은 영·한·독 ASR(speechlm main·short + ASR words + YODAS-de tn2) + 단조 번역 타깃
#           (ast-mono-v1/targets-fix-gen-v3-bf16, 묶은 형태 pack/ — 노드 안 프로세스가 mmap 으로 공유).
#   번역 타깃이 있는 항목은 --ast-frac(기본 0.5) 확률로 번역 모드(prefix 에 translate 줄, 출력에 <TRANS> 구간), 나머지는 화자·언어
#   prefix 의 ASR. 화자 태그 A/B 는 항목마다 무작위. <TRANS>(151726)·태그 decode 차단 해제는 학습 스크립트가 한다(semcommit_tokens.add_ast_tokens).
#   레시피(인코더 학습·[56,3] 고정·SpecAugment·속도·잡음·final 30 %·δ4–10·worker 4)는 Stage 1(slurm/submit-q06-s1m3-d10.sh)과 같다.
# 추가 인자는 submit-semcommit-train-apex.sh 에 그대로 넘어가고 같은 옵션은 뒤의 값이 이긴다. 예:
#   bash slurm/submit-q17-s2-ast.sh --dry-run                                   # sbatch 없이 명령만 출력
#   bash slurm/submit-q17-s2-ast.sh --epochs 2 --nodes 4 --partition hpc       # 제출
#   같은 --run 으로 다시 내면 마지막 checkpoint 에서 이어서 한다.
#   주의: --extra 를 다시 주면 아래 --extra(인코더 고정·worker 수·AST 옵션)가 통째로 바뀐다.
set -euo pipefail
cd "$(dirname "$0")/.."
SNAP=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
NB=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
DE=/soundai/users/tskim/VAPKT-data/data/yodas-de-v1/lists/de-words-train-tn2.list
AST=/soundai/users/tskim/VAPKT-data/data/ast-mono-v1/targets-fix-gen-v3-bf16
INIT=/soundai/Model/VAPASR/semcommit-q17-s1m3-d10/final
[[ -f "$AST/pack/en-de/meta.json" && -f "$AST/pack/ko-en/meta.json" ]] || { echo "번역 타깃 묶음 없음: $AST/pack (ast_build_targets.py --pack-only)" >&2; exit 1; }
exec bash slurm/submit-semcommit-train-apex.sh --run q17-s2-ast --snap "$SNAP" --init "$INIT" \
  --asr-only --offline-frac 0.3 --asr-list "$ASR" --de-list "$DE" \
  --train-encoder --spec-augment 2,27,2,0.03 --speed-perturb 0.9,1.0,1.1 --noise-bank "$NB" --varlen \
  --delays 4,5,6,7,8,9,10 --batch-tokens 16384 --lr 6e-5 --warmup 500 --epochs 2 --save-every 1000 \
  --nodes 2 --extra "--encoder-right-context 3 --encoder-context-sampling fixed --num-workers 4 --ast-targets $AST --ast-frac 0.5" "$@"
