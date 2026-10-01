---
type: output
status: active
created: 2026-10-01
updated: 2026-10-01
summary: Stage 1 파일럿(0.6B·1.7B thinker, 1 epoch) 평가 — δ4/6 에서 E2·v035-d8 대비 10~24 % 개선, 1.7B 는 한국어에서만 유의하게 낫고 final 모드는 Kspon 에서 오프라인 Qwen3-ASR-1.7B 를 넘음
sources:
  - '[[2026-10-01-stage1-pilot-eval-mxc]]'
related:
  - '[[output-staged-training-plan-20261001]]'
  - '[[output-semcommit-v035-d8-eval]]'
  - '[[output-asr-accuracy-levers-20260929]]'
raw_authors:
  - tskim
---

# Stage 1 파일럿 평가 (2026-10-01)

## 설정
- 학습([[output-staged-training-plan-20261001]]):
  - Stage 0: adapter 만, final 모드, 2,000 step
  - Stage 1: 스트리밍 δ 2–8 + final 30 %, 인코더 포함 전체 학습, SpecAugment·속도·잡음·잔향, ASR 전용, 2 노드 1 epoch
- 평가: `single-turn-vpaa-kspon-v1`. 영어 기본 세트가 LibriSpeech 에서 VoxPopuli-AA 로 바뀌었다.
- 기준선:
  - VoxPopuli-AA 는 이번에 새로 쟀다.
  - Kspon 은 v1 결과를 재사용한다(같은 행).
- 원자료: `raw/sources/experiments/2026-10-01-stage1-pilot-eval-mxc/`

## 오류율 (영어 WER/AA-WER, 한국어 CER 공백 제외, %)

| 모델 | VoxPopuli-AA δ4 | δ6 | final | Kspon clean δ4 | δ6 | final | Kspon other δ4 | δ6 | final |
|---|---|---|---|---|---|---|---|---|---|
| **q17-s1 (1.7B)** | 7.70/5.73 | 7.18/5.20 | **6.84/4.77** | **9.70** | **9.40** | **8.21** | **9.95** | **9.63** | **8.42** |
| q06-s1 (0.6B) | 7.73/5.73 | 7.18/5.09 | 7.14/5.09 | 10.55 | 10.24 | 9.27 | 10.87 | 10.47 | 9.22 |
| E2 | 8.94/7.18 | 8.01/6.16 | – | 12.32 | 11.76 | – | 12.86 | 12.67 | – |
| v035-d8 | 9.88/8.02 | 9.15/7.26 | – | 12.26 | 12.04 | – | 12.26 | 12.19 | – |
| Qwen3-ASR-1.7B 오프라인 | | | 5.10/3.22 | | | 8.92 | | | 8.71 |
| Nemotron RNN-T [56,0] | | | 6.88/4.88 | | | 19.46 | | | 16.80 |

δ2: q17 11.75 / 14.80 / 14.81, q06 11.86 / 15.72 / 16.13, v035-d8 14.26 / 13.80 / 14.07.

## 결론 (짝 bootstrap 95 %)
1. **새 방식이 확실히 낫다.**
   - q17-s1 은 δ4/6 에서 E2 대비 −10~24 %, v035-d8 대비 −19~22 % 다. 12 칸 모두 유의하다.
   - 예외는 Kspon δ2 로, v035-d8 이 1.7B 보다 0.7~1.0 %p 낫다(유의).
2. **1.7B 는 한국어에서만 이득이다.**
   - Kspon 은 모든 칸이 −6~11 % 로 유의하다.
   - VoxPopuli 스트리밍은 차이가 없다. final 만 −4 % 로 유의하다.
   - 오프라인 상한과 같은 양상이다(영어 3.25 → 3.22, 한국어 10.8 → 8.9).
3. **final 모드(인과 인코더)가 Kspon 에서 오프라인 Qwen3-ASR-1.7B 를 넘는다.**
   - eval_clean 8.21 vs 8.92(유의), eval_other 8.42 vs 8.71(n.s.)
   - Kspon train 이 학습에 포함된 영역 내 효과일 수 있다.
   - 영어 final 은 아직 +1.7 %p(+34 %) 뒤진다.
4. **영어에서 같은 인코더의 RNN-T 와 격차가 사라졌다.**
   - 이전 E2 δ6 는 RNN-T 보다 +1.1 %p 나빴다.
   - q17 δ6 는 +0.30(n.s.), final 은 −0.04(n.s.) 다.

## 다음
- 크기: 1.7B 로 간다(한국어 이득, final 품질). 영어만 쓰는 환경이면 0.6B 도 충분하다.
- Stage 2(<SEM_END>) 는 q17-s1 에서 시작한다. Stage 1 대비 WER 이 유의하게 나빠지지 않는지 함께 본다.
- 개선 여지:
  - Stage 1 은 1 epoch 뿐이다. 마지막 손실 0.27 이 계속 내려가는 중이었다.
  - Kspon δ2 열세
  - 영어 final 과 오프라인의 격차
- self 2-pass(`--decoder self:`)로 스트리밍 + final 결합 지연·품질을 아직 재지 않았다.
- 운영:
  - Stage 1 은 NFS 정체로 실제 2.5 s/step 이었다(계산만 0.6–1.0).
  - checkpoint 200 step 마다 27 GB(1.7B)는 과하다. 다음에는 500–1000 step 간격이 낫다.
