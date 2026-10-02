---
type: overview
status: active
created: 2026-09-03
updated: 2026-10-02
summary: 2인 교차언어 대화의 동시 음성→텍스트 번역(CST-S2TT)과 상호작용 제어, CST-Bench 를 연구하는 볼트 (2026-10-02 목표 전환)
---

# 개요

## 이 볼트가 다루는 것 (2026-10-02 부터)

**대화형 동시 음성→텍스트 번역 — [[conversational-simultaneous-s2tt]] (CST-S2TT).**

서로 다른 언어를 쓰는 두 사람이 한 대화에서 번갈아 말할 때, AI 통역기가 상대 언어의 텍스트 번역을 실시간으로 낸다. 번역에 더해 다음을 함께 판단한다.
- 언제 듣고 언제 쓰기 시작할지
- 언제 기다리고 멈출지
- 어느 방향(A→B / B→A)으로 전환할지

핵심 가설은 **번역 정책과 turn-taking 정책은 독립이 아니다** 이다. [[source-cst-s2tt-project-spec-v0-1]]

이전 목표(스트리밍 ASR + 미래 2 초 turn-taking 투사, [[streaming-conversational-projection-asr]])에서 전환했다. 전환 이유와 이어지는 자산은 [[decision-project-goal-cst-s2tt]].

## 첫 논문 범위

- 포함:
  - 2 인, 서로 다른 언어, 실제 multi-turn 대화
  - 스트리밍 입력, 양방향 동시 S2TT
  - semantic commitment(확정 번역은 수정 불가), 턴 전환·맞장구·끼어들기
- 제외: TTS/S2ST, 음성 자연성, 3 인 이상, 화자 분리, code-switching 중심 과제
- S2ST 는 후속 확장이다.

## 현재 최우선

모델보다 **[[cst-bench]] v0 와 strong pipeline 기준선**이 먼저다. 첫 질문은 “상호작용을 다루지 않는 기존 SimulST 파이프라인은 어디서, 얼마나 실패하는가?”다. joint 모델은 kill criterion 을 통과한 뒤에 만든다.

| 단계 | 내용 | 태스크 |
|---|---|---|
| Phase 0 | 교차언어 대화 코퍼스 확보, 공통 schema 파서 | [[task-cst-data-access]], [[task-cst-conversation-schema-parser]] |
| Phase 1 | 턴 재구성·번역 정규화·통제 타임라인 생성기 | [[task-cst-timeline-generator]] |
| Phase 2 | CST-Bench v0 (두 세트·지표·revision-free commit) | [[task-cst-bench-v0-metrics]] |
| Phase 3 | 기준선 B0–B4, kill criterion | [[task-cst-strong-baselines]], [[task-cst-kill-criterion-check]] |
| Phase 4 | joint 모델(B5) + ablation | — |
| Phase 5 | 한국어 확장(ETRI 또는 KO↔EN 수집) | — |
| Phase 6 | S2ST 확장 | — |

## 주제 영역

### 과제와 평가
- [[conversational-simultaneous-s2tt]]: 과제 정의, 기존 SimulST 와의 차이, 기존 자산 연결
- [[cst-bench]]: 두 세트, 합성 상호작용, 지표, 기준선, kill criterion

### 이전 목표에서 이어지는 자산
- 스트리밍 백본: 인과 인코더 Nemotron + Qwen3-ASR thinker, δ 지연 → [[decision-asr-backbone]], [[output-stage1-pilot-eval-20261001]]
- semantic commit: `<SEM_END>` 라벨, 골드 commit, commit 지연 지표 → [[output-semcommit-recipe-v0.3]], [[output-semcommit-gold-v1]], [[output-semcommit-v035-d8-eval]]
- turn-taking 기준선: VAP 재현 → [[voice-activity-projection]], [[output-vap-turnbench-baseline-reproduction]]
- 지연·인과성 규약: [[streaming-causality-and-latency-budget]], [[output-encoder-causality-audit]]

### 이전 목표의 기록 (참고)
- [[streaming-conversational-projection-asr]], [[output-streaming-vap-research-plan]], [[turn-taking-objectives]], [[korean-turn-taking-cues]]

## 다음 관문

- 즉시 접근 가능한 교차언어 대화 코퍼스 1 개 확보(라이선스 확인 포함)
- 입력 채널 재결정(mono 혼합 → 화자별 스트림) → [[decision-mono-input]] 재검토
