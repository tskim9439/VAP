---
type: decision
status: active
decision_status: proposed
owner: tskim
review: 2026-10-22
created: 2026-09-04
updated: 2026-09-04
summary: IS-SLM(Interleaved Streaming SLM)이 Paper 1·2 모두의 단일 주력 모델. 이중 프레임율+RNN-T 안(A)은 2026-09-04 사용자 결정으로 기각
sources:
  - [[output-unified-slm-architecture-plan]]
  - [[output-model-architecture-proposal]]
---

# 결정: 목표 모델 구조

## 맥락

사용자 방향(2026-09-04): 전사를 RNN-T 로 독립 수행하는 구조보다 **하나의 통합 SLM** 이 streamable encoder 출력을 실시간으로 받아
전사와 turn 예측을 함께 내는 모델을 원한다 (Muse Voice Transcribe 와 같은 실시간성). 이전 스텝의 text token 임베딩을 다음 speech
프레임과 합산해 LLM 입력으로 넣는 구조.

## 선택지

| 선택지 | 장점 | 단점 |
|---|---|---|
| A. 이중 프레임율 + RNN-T ([[output-model-architecture-proposal]]) | 실측 근거 위에 서 있음, 50 Hz 타이밍, 전사 품질 보장(RNN-T 그대로) | 두 하위 시스템, "통합" 이 아님 |
| B. 시간 동기 합산 SLM ([[output-unified-slm-architecture-plan]] v0) | 1 forward/chunk, 고정 길이 시퀀스 | **1 프레임 = 1 토큰 상한**(한국어 BPE), 합산은 KV 가 이미 주는 정보와 중복 |
| **B′. Interleaved SLM** ([[output-interleaved-streaming-slm-architecture]]) | `<NEXT_AUDIO>` 가변 방출로 토큰율 상한 해소, 모달리티 분리, Muse·Qwen3-ASR 생태계와 정합, gated residual 로 융합을 안전하게 | tick 당 (2+M) forward, 겹침 텍스트 직렬화 규약 필요, **12.5 Hz turn 헤드 한계(보고서 미반영)** |
| C. Muse 재현(interleave + 결정 토큰) | 검증된 형태 | closed weights, VAP 미래 투사 없음 — 우리 기여가 아님 |

## 결정 (2026-09-04 사용자 확정)

**IS-SLM(B′)이 단일 주력 모델이다.** 이중 프레임율 + RNN-T 안(A)은 **완전 기각** — 대조군으로도 개발하지 않는다.
B′ 는 50 Hz 음향 사이드 브랜치 하이브리드(U3 ablation) + joint chunk token 기본 + 겹침 텍스트 직렬화 규약 + U1 WER 관문을 포함한다.
합산 융합(B v0)은 gated residual ablation 으로만 남긴다.

대조군은 외부 것으로 한다: VAP(oto fine-tune, 공식 예측 재현 완료), TurnBench 동봉 baseline(rms_vad·dualturn·wavlm_causal 등),
Nemotron RNN-T(전사 WER 기준), Stage 1 frozen probe(표현별 turn 상한). 즉 "통합의 가치" 는 A 가 아니라
**같은 encoder 위의 encoder-only probe 대 IS-SLM 상태** 로 판정한다 (보고서 성공 기준 4).

## 결과 / 파급

- Paper 1 의 내용이 바뀐다: Stage 1 probing(표현 비교, 완료 직전) + **IS-SLM U0–U3** (interleaved ASR → self-conditioned → 멀티태스크 turn 헤드).
  기존 Phase 2 "Fusion (A 구현)" 태스크 2건은 폐기·병합 — `task-stage2-multitask-vap-with-wer-guardrail`, `task-stage3-linguistic-state-fusion` → U3.
- fallback 이 사라진 위험: U1 이 WER 관문(≤10 %)을 못 넘으면 남는 것은 "Nemotron RNN-T + encoder probe" 라는 기존 baseline 스택뿐이다.
  그래서 U0(토큰율·정렬)과 U1 을 **가장 먼저, 짧게** 돌려 조기에 판정한다.
- 12.5 Hz 한계(Stage 1 실측)는 IS-SLM 안에서 50 Hz 사이드 브랜치로 흡수한다 — U3 의 첫 ablation.
- [[decision-asr-backbone]] 최종 결정: Nemotron `[56,0]` → new adapter → Qwen3-ASR thinker를 U1부터 최종 모델까지 유지한다.
- [[output-model-architecture-proposal]] 은 superseded.

## 재검토

2026-10-22 — Stage 1 최종 + U0/U1 결과. U1 관문 실패 시 재논의.
