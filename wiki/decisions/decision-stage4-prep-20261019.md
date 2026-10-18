---
type: decision
status: active
created: 2026-10-19
updated: 2026-10-19
summary: Stage 4 준비 결정 — 실제 같은 언어 대화를 지금 번역, 교차 언어는 이어 붙인 합성 대화만(TTS 합성은 병행 준비), <EOT> 는 말이 끝날 때, 지연 δ4–10, 한국어–영어 평가셋은 보류
decision_status: accepted
owner: tskim
review: 2026-10-26
sources:
  - '[[decision-cst-training-stages-and-speaker-slots]]'
  - '[[decision-ast-monotonic-data]]'
  - '[[decision-cst-paper-v1-scope]]'
  - '[[output-stage3-prep-plan-20261014]]'
  - '[[output-ast-mono-generation-20261009]]'
---

# Stage 4 준비 결정 (2026-10-19)

## 맥락
- Stage 3 최종을 p11 checkpoint-3000 으로 정했다([[output-stage3-prep-plan-20261014]]).
- Stage 4 는 교차 언어 2 인 대화와 번역이다. 같은 언어 대화의 번역도 함께 섞는다([[decision-cst-training-stages-and-speaker-slots]]). 주 언어쌍은 EN↔DE, 둘째는 KO↔EN 이다([[decision-cst-paper-v1-scope]]).
- 2026-10-19 조사: 최소 경로로 본 학습 제출까지 약 4–6 일로 추정했다. 미정 사항은 여섯 개였다.

## 결정 (사용자, 2026-10-19)
1. **실제 같은 언어 대화의 번역 타깃을 지금 만든다**(한국어 약 940 h 포함).
2. **교차 언어 대화는 우선 실제 발화를 이어 붙인 합성 대화만 학습에 쓴다.** TTS 합성은 병행해서 준비한다. 학습 투입은 나중에 정한다.
3. **`<EOT>` 는 말이 끝날 때 낸다.** 번역이 끝나는 시점과 상관없이 Stage 3 규칙을 그대로 쓴다.
4. **학습 지연은 Stage 3 와 같다**(δ4–10).
5. **한국어–영어 평가셋은 결정을 미룬다.**
- 답하지 않은 것: KO↔EN 을 논문 본문에 넣을지([[decision-cst-paper-v1-scope]] 재검토일 10-19)

## 결과
- 대화 번역은 Stage 2 방식([[output-ast-mono-generation-20261009]])을 대화 발화에 그대로 쓴다.
  - Qwen3.8-27B, 단조 번역, 판정 관문, 원문 접두 번역으로 정한 단어별 확정 시각(stable prefix)
  - 방향도 Stage 2 와 같다: KO→EN, DE→EN, EN→DE·KO
- 격리 발화는 원문이 의심스러워 번역을 학습 타깃으로 쓰지 않는다.
  - 구현 메모(2026-10-19): Stage 3 에서 되살린 발화는 번역을 따로 만들어 입력 문맥으로만 둔다(label −100). 번역 모드에서 한국어 행의 약 42 % 가 번역 없이 빠져 태그·`<EOT>` 구조가 끊기지 않게 하려는 것이다.
- 진행과 산출은 [[output-stage4-prep-plan-20261019]] 에 적는다.

## 재검토
- 2026-10-26: TTS 합성을 학습에 넣을지, 한국어–영어 평가셋
