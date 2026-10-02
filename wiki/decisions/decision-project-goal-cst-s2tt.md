---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-10-16
created: 2026-10-02
updated: 2026-10-02
summary: 프로젝트 목표를 스트리밍 ASR+turn-taking 투사에서 대화형 동시 음성→텍스트 번역(CST-S2TT)과 CST-Bench 로 전환
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
---

# 프로젝트 목표를 CST-S2TT 로 전환

## 맥락

- 2026-09 까지의 목표는 두 가지였다.
  - 한 스트리밍 speech LM 이 80 ms 마다 전사와 “미래 2 초의 대화 역학(VAP)”을 함께 내는 것([[streaming-conversational-projection-asr]], [[output-streaming-vap-research-plan]])
  - 그 하위 단계인 `<SEM_END>` 의미 commit 스트리밍 ASR
- 그동안 갖춘 것:
  - 인과 인코더 + Qwen3-ASR thinker 백본, δ 지연 interleave, final 모드
  - VoxPopuli-AA·Kspon 평가 체계
  - 골드 commit 주석, 교사 LLM 라벨링 파이프라인
- 2026-10-02 사용자가 새 명세를 제시하고 목표를 그 문서대로 바꾸라고 지시했다. [[source-cst-s2tt-project-spec-v0-1]]

## 검토한 선택지

| 선택지 | 장점 | 단점 |
|---|---|---|
| A. 기존 목표 유지 | 진행 중 학습·라벨링 그대로 | 사용자 지시와 다름. turn-taking 단독·Muse 류는 경쟁이 많다 |
| B. CST-S2TT 로 전환 | 과제·벤치마크·지표가 새롭다. 기존 자산(스트리밍 백본·commit·VAP)이 기준선과 구성요소로 이어진다 | 번역 데이터·교차언어 대화 코퍼스가 0 에서 시작한다 |

## 결정

**프로젝트 목표를 CST-S2TT(대화형 동시 음성→텍스트 번역)로 바꾼다.** 첫 논문 범위는 명세 v0.1 을 따른다.
- 2 인 교차언어 multi-turn 대화
- 양방향 동시 S2TT
- semantic commitment
- 턴 전환·맞장구·끼어들기

TTS/S2ST·다자·화자 분리·code-switching 은 제외한다. 당장의 최우선은 모델 개발이 아니라 **CST-Bench v0 와 strong pipeline 기준선**이다. → [[conversational-simultaneous-s2tt]], [[cst-bench]]

## 근거

- 사용자 지시다(2026-10-02).
- 명세의 핵심 질문(“상호작용을 다루지 않는 SimulST 파이프라인은 어디서 얼마나 실패하는가”)은 모델 없이도 벤치마크와 기준선으로 먼저 답할 수 있다. kill criterion 으로 방향 재검토 시점이 정의돼 있다.

## 결과 (이어지는 것 / 멈추는 것 / 다시 볼 것)

이어지는 자산:
- 스트리밍 인과 백본과 δ 지연 시퀀스 → SimulST 원천 이해부, joint 모델(B5)의 출발점
- `<SEM_END>` 라벨·골드 commit·commit 지연 지표 → semantic chunk 경계, 끼어들기 난이도 설계, commit 지연
- final 모드·2-pass → 오프라인 참조(B0)와 확정 구간 재번역
- VAP 재현 → B3 기준선
- 평가 습관(짝 bootstrap, 지연 맞춤 비교)

목표에서 내려가는 것:
- “미래 2 초 투사 헤드를 한 모델에 넣는다”는 최종 목표 → B3 기준선 구성요소로 격하
- AA 순위표(영어 ASR) 순위 목표 → ASR 은 cascade 기준선의 부품 품질 관리로만

진행 중 작업(2026-10-02 기준, 사용자 판단 필요):
- semcommit 라벨링 job 79205: 원천 쪽 semantic chunk 경계와 commit 라벨로 쓸 수 있어 계속할 가치가 있다
- Stage 1 본학습(q17-s1m-r0·r3) 결과: 원천 ASR 품질 기준선으로 보존한다. 영어 악화 대응·Stage 2 진행은 CST 우선순위에 맞춰 재판단한다

재검토가 필요한 기존 결정(각 결정 페이지는 이 결정으로 자동 변경하지 않는다):
- [[decision-mono-input]]: 명세는 두 화자의 스트림을 따로 관찰하고 화자 분리를 범위에서 뺀다 → 화자별 채널 입력으로 바꿀지
- [[decision-target-architecture]]: IS-SLM 목표 구조 → 번역 출력·방향 토큰을 포함한 joint 구조로
- [[decision-multi-speaker-scope]]: 2 인 고정
- [[decision-semcommit-turn-eot-scope]]: `<EOT>` 계획 → 턴 전환·STOP/SWITCH 행동으로 흡수할지
- [[decision-korean-benchmark-release-scope]]: 한국어는 Phase 5(ETRI 또는 소규모 KO↔EN 수집)로 이동

## 다음 행동

- [[task-cst-data-access]]
- [[task-cst-conversation-schema-parser]]
- [[task-cst-timeline-generator]]
- [[task-cst-bench-v0-metrics]]
- [[task-cst-strong-baselines]]
- [[task-cst-kill-criterion-check]]

이 결정 페이지는 병합 전에 다른 팀원 검토가 필요하다(AGENTS.md).
