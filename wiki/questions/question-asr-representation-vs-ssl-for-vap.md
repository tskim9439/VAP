---
type: question
status: seed
created: 2026-09-03
updated: 2026-09-03
summary: ASR pretrained streaming representation이 SSL/생성형 pretraining보다 turn projection에 유리한가 — 연구의 중심 질문
sources:
  - [[source-turn-taking-related-work-2026]]
---

# ASR representation 이 turn projection 에 정말 유리한가?

## 질문

> 대규모 ASR pretraining 으로 얻은 streaming representation 자체가
> conversational future prediction 능력을 담고 있는가?

## 왜 중요한가

이 연구의 **첫 번째 논문 가치가 여기에 있다.** 답이 "그렇다" 면 이후 모든 단계에
근거가 생긴다. "아니다" 면 backbone 전략([[decision-asr-backbone]])을 전면 재검토해야 한다.

## 경쟁 가설이 이미 존재한다

[[source-turn-taking-related-work-2026]] 의 **DualTurn** 은 정반대를 주장한다 —
ASR 이 아니라 **dual-channel generative pretraining** 이 답이며, VAP 를 크게 앞섰다
(weighted F1 0.633 vs 0.389). 따라서 이 질문은 **ASR vs SSL** 이 아니라
**ASR vs SSL vs generative dual-channel** 의 3자 비교여야 한다.

## 실험 설계 — 교란 변수를 반드시 통제

encoder 만 바꾸는 단순 비교는 위험하다. 후보들이 frame rate, 차원, causality,
pretraining 시간, 모델 크기에서 모두 다르기 때문에 "ASR pretraining 이 좋다" 와
"파라미터·데이터가 많다" 를 구분할 수 없다.

| 비교 대상 | pretraining | frame rate |
|-----------|-------------|-----------:|
| CPC (원 VAP) | SSL | 50 Hz |
| WavLM Base / Large | SSL | 50 Hz |
| Nemotron FastConformer | **ASR** | 12.5 Hz |
| Qwen3 AuT | **ASR** | 12.5 Hz |
| DualTurn encoder | **generative dual-channel** | — |
| random-init (floor) | 없음 | — |

통제 항목:

1. 모든 표현을 **공통 frame rate 로 리샘플** 하고, 원 해상도 조건도 함께 보고.
2. probe **용량 고정** (동일한 작은 VAP head).
3. **크기 대조군** 포함 (WavLM Base vs Large) → 크기 효과 분리.
4. **random-init floor** 로 "probe 가 스스로 학습한 양" 측정.
5. encoder 별 **pretraining 시간과 실효 lookahead** 를 표에 명시
   ([[streaming-causality-and-latency-budget]]).

## 상태

미해결. → [[task-stage1-encoder-probing]]
