---
type: concept
status: active
created: 2026-10-02
updated: 2026-10-02
summary: 대화형 동시 음성→텍스트 번역(CST-S2TT) — 2인 교차언어 대화에서 번역과 언제·어느 방향으로 쓸지를 함께 판단하는 과제
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
---

# Conversational Simultaneous S2TT (CST-S2TT)

## 정의

- 서로 다른 언어를 쓰는 두 화자 A(언어 X)와 B(언어 Y)가 같은 대화 문맥에서 번갈아 말한다.
- 시스템은 두 사람의 음성을 실시간으로 들으며 상대 언어의 **텍스트 번역**을 낸다.
- 번역 품질에 더해 **상호작용 제어**(interaction control)를 함께 푼다. [[source-cst-s2tt-project-spec-v0-1]]

매 시점의 판단:

1. 지금 누가 말하는가(화자·턴 상태 S_t)
2. 번역 방향: A→B / B→A / 없음
3. 진짜 턴인가, 맞장구(backchannel)인가
4. 의미가 충분히 모였는가 → 쓰기 / 기다리기(WRITE / WAIT)
5. 계속 / 중단 / 전환(CONTINUE / STOP / SWITCH)

개념 모형은 P(Y_t, A_t, S_t | X_≤t, H_t)다. Y 는 번역 토큰, A 는 상호작용 행동, H 는 대화 이력이다.

핵심 가설: **번역 정책과 turn-taking 정책은 독립이 아니다.** 그래서 둘을 따로 붙인 파이프라인(2×SimulST + VAD/VAP)은 상호작용 사건에서 실패하고, 이를 함께 모델링해야 한다는 주장이다.

## 기존 동시통역(SimulST)과의 차이

- 기존 SimulST 는 원천 화자와 번역 방향이 주어진 단일 스트림을 번역한다. 쓰기/기다리기 정책(wait-k·AlignAtt·local agreement 등)만 다룬다.
- CST-S2TT 는 원천 화자·방향을 모델이 판단한다. 맞장구·끼어들기·빠른 턴 전환을 명시적으로 처리한다.
- **semantic commitment**(확정 텍스트는 수정 불가)를 핵심 문제로 둔다.
- 첫 단계는 TTS 를 뺀 **텍스트 출력**이다. 음성 생성 품질·보코더 지연 같은 부가 요인을 제거하고 상호작용만 깨끗이 평가하려는 것이다.

## 이 볼트의 기존 자산과의 연결

이전 목표(스트리밍 ASR + turn-taking 투사, [[streaming-conversational-projection-asr]])에서 만든 것 중 이어지는 것:

| CST-S2TT 의 요구 | 기존 자산 | 비고 |
|---|---|---|
| 스트리밍 입력 → 토큰 방출 시퀀스 | Nemotron `[56,0]`/`[56,3]` 인과 인코더 + Qwen3-ASR thinker, δ 지연 interleave([[decision-asr-backbone]], [[output-stage1-pilot-eval-20261001]]) | 원천 언어 이해부. 번역 출력·방향 토큰은 새로 붙여야 한다 |
| semantic commit | `<SEM_END>` 의미 완결 신호, 골드 commit 주석([[output-semcommit-gold-v1]]), 레시피([[output-semcommit-recipe-v0.3]]), 창 없는 commit 지연 지표([[output-semcommit-v035-d8-eval]]) | 원천 쪽 commit 이다. 번역 쪽 revision-free commit 과의 대응은 미정 |
| final(오프라인) 전사·2-pass | final 모드, self 2-pass([[output-staged-training-plan-20261001]], [[output-semcommit-two-pass-plan]]) | B0 오프라인 참조, 확정 구간 재번역에 쓸 수 있다 |
| VAP 기준선(B3) | VAP TurnBench 재현([[output-vap-turnbench-baseline-reproduction]]), [[voice-activity-projection]] | B3 의 turn-taking 예측기로 바로 쓸 수 있다 |
| 지연 규약 | [[streaming-causality-and-latency-budget]], 인코더 lookahead 실측([[output-encoder-causality-audit]]) | SimulST 지표(AL·LAAL)와 맞춰야 한다 |
| ASR 품질 가드레일 | single-turn 평가(VoxPopuli-AA + Kspon), 짝 bootstrap | cascade 기준선의 ASR 부분 품질 관리 |

## 열린 문제

- **입력 채널**: 기존 결정은 mono 혼합 입력이었다([[decision-mono-input]]). 명세는 “양쪽 음성 스트림”을 관찰하고 화자 분리(diarization)를 범위에서 뺐다. 그래서 화자별 채널 입력이 자연스럽다. 재결정이 필요하다. → [[decision-project-goal-cst-s2tt]]
- **원천 commit 과 번역 단위의 관계**: 현 `<SEM_END>` 는 문장급(EN 11 s·KO 8 s/단위)이라 WRITE 단위로는 크다. 번역 기준 의미 단위(MU·문맥 정렬)를 따로 정의할 것을 권고했다. 한국어 연결어미 규칙 음성은 KO→EN 청크 경계와 충돌한다. → [[output-simulst-translation-unit-survey-20261002]]
- **번역 데이터**: 실제 교차언어 대화 + 참조 번역이 필요하다. 지금 확보된 것은 없다. → [[task-cst-data-access]]

## 관련
- [[cst-bench]] — 이 과제의 평가 체계
- [[voice-activity-projection]] — B3 기준선의 turn-taking 예측기
