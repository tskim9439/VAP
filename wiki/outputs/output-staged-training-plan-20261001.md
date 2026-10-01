---
type: output
status: active
created: 2026-10-01
updated: 2026-10-01
summary: 학습 단계 재설계 — Stage 0 adapter 워밍업 → Stage 1 스트리밍+final 겸용 ASR → Stage 2 <SEM_END>, 0.6B·1.7B thinker 파일럿(Nemotron 인코더 유지)
sources:
  - '[[output-asr-accuracy-levers-20260929]]'
  - '[[decision-asr-backbone]]'
  - '[[output-encoder-causality-audit]]'
related:
  - '[[output-semcommit-v035-d8-eval]]'
  - '[[output-semcommit-two-pass-plan]]'
raw_authors:
  - tskim
---

# 학습 단계 재설계와 1.7B 파일럿 (2026-10-01)

## 사용자 요구와 결정 근거

- WER/CER 이 가장 중요하다. 스트리밍 부분 전사와 최종 전사 둘 다 중요하다.
- 사용자가 세 가지를 물었다.
  1. thinker 를 Qwen3-ASR-1.7B 로 바꿀까?
  2. 인코더를 Qwen AuT 로 바꿀까?
  3. 학습 단계를 나눌까?
- 분석 결과:
  1. **1.7B 는 파일럿으로 확인한다.**
     - 오프라인에서 0.6B → 1.7B 는 −17~21 % 다.
     - 영어에서는 같은 인코더 RNN-T 가 우리보다 나아서, 디코더 쪽 여유가 보인다([[output-asr-accuracy-levers-20260929]] §2).
  2. **인코더는 Nemotron 을 유지한다.**
     - AuT 는 기본 경로가 비인과이고, 블록 모드도 평균 420 ms 를 미리 본다. 프레임 인과로 바꾸면 재학습 없이 WER 23.5 % 이고, 잡음에도 약하다([[decision-asr-backbone]], [[output-encoder-causality-audit]]).
     - AuT 품질은 2-pass 최종 전사로 얻는다.
  3. **단계를 나눈다.** WER 관문을 따로 두고, 라벨이 바뀌면 Stage 2 만 다시 돌린다.
- 사용자 아이디어로 **δ 스트리밍과 final(오프라인) 전사를 한 모델에서 함께 학습**한다. 인과 인코더라 스트리밍 중 특징을 그대로 써서 같은 모델로 문장 단위 final 재디코드(self 2-pass)를 할 수 있다.

## 단계

| 단계 | 학습 대상 | 데이터·모드 | 관문 |
|---|---|---|---|
| Stage 0 | adapter 만(thinker 동결) | final 모드만(Qwen3-ASR 원 형식이라 새 특수 토큰 불필요), ASR 전용 | adapter 수렴(원 형식 손실) |
| Stage 1 | thinker + adapter + 인코더 | 스트리밍 δ 2–8 + final(--offline-frac 0.3), ASR 전용(라벨 3.8k h + SEM 중립 4.3k h), SpecAugment·속도·잡음·잔향 | single-turn WER/CER(δ 별 + final), Kspon + VoxPopuli-AA |
| Stage 2 | 전체(낮은 LR, 짧게) | <SEM_END> 라벨 + ASR 재생(sem_mask) | Stage 1 대비 WER 유의 악화 없음(짝 bootstrap) + 골드 commit F1 |

## 구현 (커밋 df19d9a + 배포)

- final 모드:
  - 형식 `vapasr/data/offline_seq.py` 는 U0.5 브리지와 같은 Qwen3-ASR 원 형식이다.
  - 디코드 `vapasr/hf/offline_decode.py`
  - 평가 `eval_single_turn_asr.py run --final`
  - 2-pass `semcommit_twopass_eval.py --decoder self:<모델>`
- `from_qwen`: adapter 출력 = thinker hidden(1.7B 2048).
- 학습 쪽 옵션:
  - 학습 스크립트(mxc 배포): `--asr-only`, `--offline-frac`, `--freeze-thinker`
  - 제출 스크립트: `--qwen-dir`, `--asr-only`, `--offline-frac`, `--freeze-thinker`
  - 데이터셋·학습 스크립트 쪽은 다른 세션의 미커밋 short 풀 헝크와 같은 파일이라 커밋하지 않았다(결정 대기).

## 1.7B 스모크 (mxc GPU 1·2, 30 step, 예산 16,384, 부분 목록)

| | Stage 0 | Stage 1 |
|---|---|---|
| 학습 파라미터 | adapter 6.3 M | 2,336 M(thinker 1.7 B + 인코더 0.6 B + adapter) |
| 손실(30 step) | 4.62 → 3.62 | 12.1 → 3.0(첫 step grad norm 839) |
| 최대 메모리 | 약 19 GB | 약 74 GB(H200 141 GB) |
| 저장·재로드 | 955/955 키 일치 | 955/955 키 일치 |

Stage 1 을 random adapter 로 바로 시작하면 첫 기울기가 매우 크다. 그래서 Stage 0 워밍업을 둔다(U0.5 가 adapter 를 증류 초기화한 것과 같은 이유).

## 파일럿 제출 명령(사용자) — 0.6B 와 1.7B 를 같은 예산으로

```bash
SNAP=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
NB=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
for Q in 1.7B 0.6B; do T=q$(echo $Q | tr -d .B | tr 'A-Z' 'a-z')
  # Stage 0: adapter 워밍업
  bash slurm/submit-semcommit-train-apex.sh --run $T-s0 --snap $SNAP --qwen-dir /soundai/Model/Qwen3-ASR-$Q --asr-only --offline-frac 1.0 --freeze-thinker \
    --asr-list $ASR --batch-tokens 16384 --max-steps 2000 --warmup 100 --nodes 1 --varlen
done
# Stage 0 이 끝나면(--after <job>) Stage 1 — Q·T 를 같게
bash slurm/submit-semcommit-train-apex.sh --run $T-s1 --snap $SNAP --init /soundai/Model/VAPASR/semcommit-$T-s0/final --asr-only --offline-frac 0.3 \
  --asr-list $ASR --train-encoder --spec-augment 2,27,2,0.03 --speed-perturb 0.9,1.0,1.1 --noise-bank $NB --varlen \
  --delays 2,3,4,6,8 --batch-tokens 16384 --lr 6e-5 --warmup 500 --epochs 1 --nodes 2
```

- 출력은 `/soundai/Model/VAPASR/semcommit-{q17,q06}-{s0,s1}` 이다.
- 두 크기를 같은 데이터·step·증강으로 비교하는 것이 목적이라 E2 이력(0.6B 전용)은 쓰지 않는다. 둘 다 Qwen3-ASR 원 모델에서 시작한다.
- 평가: `eval_single_turn_asr.py run --deltas 2 4 6 --final`(Kspon + VoxPopuli-AA, 짝 bootstrap). 기준은 v035-d8·E2 이고, 2-pass(Qwen 1.7B iso)와 self 2-pass 도 함께 본다.
