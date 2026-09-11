---
type: question
status: open
created: 2026-09-11
updated: 2026-09-11
summary: <HOLD>·<BC> 와 의미 기반 turn 라벨(완결/미완결/맞장구)을 한국어 구어체·영어 lexical 전사에서 신뢰성 있게 만들 수 있는가 — Phase 2 에서 제외, Stage 3 전 κ 파일럿으로 판단
sources:
  - '[[output-phase2-turn-token-proposal]]'
---

# 질문: `<HOLD>`·`<BC>` 의미 라벨의 신뢰성

## 배경
[[output-phase2-turn-token-proposal]] 은 turn 토큰 4 종을 제안했으나, `<HOLD>`(미완결)·`<BC>`(맞장구)는 텍스트의 의미 판정(LLM + 사람 검증)에 기대야 하고, 한국어 실발화의 머뭇거림·어미의 비정형 사용, 영어 구두점 없는 전사에서 판정이 흔들릴 수 있다. 2026-09-11 사용자 결정으로 Phase 2 는 `<ONSET>`·`<EOT>`(타이밍 라벨)만 쓰고 나머지는 Stage 3 으로 미뤘다.

## 답을 내려면
1. Q0 파일럿(제안서 §3f.3): KO/EN 각 300 멈춤 지점(발화 안 / 경계-상대 / 경계-같은 화자 각 100)을 3 명이 독립 판정 → 사람-사람 κ, LLM-사람 κ 를 등급별로.
2. 기준: 사람-사람 κ ≥ 0.7, LLM-사람 κ ≥ 0.6 인 등급만 학습에 사용. 미달 시 축소 사다리(soft 라벨 → 2 등급 → 타이밍 라벨).
3. `<HOLD>` 없이도 정체 문제가 추론 정책(타임아웃·hazard 승격)으로 충분히 처리되는지 Phase 2 결과에서 확인 — 충분하면 `<HOLD>` 의 필요성 자체가 줄어든다.

## 관련
- [[output-phase2-turn-token-proposal]] §3b–§3f (이월된 설계)
- [[task-event-label-heuristics-validation]] — VAD 휴리스틱의 TurnBench gold 검증
