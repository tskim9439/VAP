---
type: concept
status: active
created: 2026-10-02
updated: 2026-10-02
summary: CST-Bench — 실제 교차언어 대화 내용에 통제된 상호작용 타이밍을 합성해 번역·지연·방향·끼어들기 대응을 함께 재는 벤치마크
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
---

# CST-Bench

[[conversational-simultaneous-s2tt]] 를 평가하는 벤치마크(가칭)다. 질문은 이것이다: “두 사람이 서로 다른 언어로 대화할 때, 시스템이 번역 품질을 유지하면서 적절한 시점과 방향으로 번역하고 상호작용 사건에 대응하는가?” [[source-cst-s2tt-project-spec-v0-1]]

## 두 세트

| 세트 | 구성 | 주로 재는 것 |
|---|---|---|
| Natural Conversation Set | 실제 대화 그대로 | 번역 품질, semantic commit, 지연, 방향 전환, 대화 문맥 활용 |
| Controlled Interaction Set | 실제 대화의 내용·화자·턴 의존성은 그대로, **타이밍만 합성** | 맞장구·끼어들기·빠른 전환에서의 행동 |

통제 세트는 같은 내용과 화자에서 상호작용 조건만 바꾼다. 그래서 “어디서 실패하는가”를 조건별로 분리할 수 있다.

## 합성 상호작용 유형

- **Normal**: A 가 끝난 뒤 B 가 시작
- **Early Turn**: A 가 끝나기 직전 B 가 시작
- **Interruption**: A 의 발화 중간에 B 가 시작(겹침)
- **Backchannel**: 짧은 맞장구를 삽입한다. KO “네/응/맞아요/음/그렇군요”, EN “yeah/right/uh-huh/okay/I see”. 턴이 아니므로 방향 전환을 하면 안 된다
- **Explicit Barge-in**: 문맥 조건 LLM + TTS 로 “잠시만요/Wait/Hold on/But what about…?” 같은 개입 발화를 생성

끼어들기 시점은 semantic chunk 경계를 기준으로 고르고, 난이도를 셋으로 나눈다.

| 난이도 | 끼어드는 시점 |
|---|---|
| Easy | 턴 끝 직전 |
| Medium | chunk 경계 부근 |
| Hard | 핵심 의미가 아직 완성되지 않은 시점 |

chunk 경계는 이 볼트의 `<SEM_END>` 의미 완결 라벨([[output-semcommit-recipe-v0.3]])로 만들 수 있을 것으로 본다(미검증).

## 지표

- **번역 품질**: COMET 계열·BLEU·chrF. 메인은 revision-free committed translation, 즉 확정된 텍스트만 채점한다.
- **지연**: AL·LAAL·첫 토큰 지연·semantic commit 지연
- **상호작용**:
  - 턴 방향 정확도(A→B / B→A / NONE)
  - 잘못된 중단·전환 비율
  - 놓친 끼어들기 비율, 잘못된 방향 번역 비율
  - Stop Latency = 이전 방향 마지막 토큰 시각 − 끼어들기 시작
  - Switch Latency = 새 방향 첫 토큰 시각 − 새 턴 시작

벤치마크는 구조를 강제하지 않는다. 무엇을, 언제, 어느 방향으로 냈는지만 본다.

## 기준선

| | 구성 | 목적 |
|---|---|---|
| B0 | 오프라인 ST | 품질 참조 |
| B1 | 2×SimulST(A→B, B→A) | 가장 단순한 양방향 |
| B2 | VAD + 2×SimulST | 음성 활동 기반 라우팅 |
| B3 | VAP + 2×SimulST | turn-taking 예측 결합([[voice-activity-projection]]) |
| B4 | Oracle 턴 경계 + 2×SimulST | 상호작용 병목의 상한 |
| B5 | Joint Conversational SimulST | 제안 모델 |

기대 결과(가설):
- B1·B2 는 정상 턴에서는 좋지만 맞장구·끼어들기·전환 지연에서 나쁘다.
- B3 는 일부 개선한다.
- joint 모델은 oracle 이 메우는 격차의 상당 부분을 외부 턴 라벨 없이 되찾는다.

## Kill criterion

joint 모델을 만들기 전에 **B3(VAP + 2× strong SimulST)가 통제 세트의 상호작용 지표를 거의 다 푸는지** 먼저 확인한다. 다 풀면 방향을 재검토한다. → [[task-cst-kill-criterion-check]]

## 원본 데이터 조건

- 필수:
  - 같은 세션에 실제 화자 2 명 이상, 세션 안에서 서로 다른 언어
  - multi-turn, 공유 문맥
  - 턴별 음성, 화자 ID·언어, 턴 순서, 전사
- 권장: 참조 번역, 턴 시각, 세션 단위 음성, 화자별 채널
- 배제: 따로 낭독한 병렬 음성, 화자별로 따로 녹음한 텍스트 대화, 단순 합친 다언어 코퍼스, 순서·문맥을 복원할 수 없는 발화 모음

후보와 확보 상태는 [[task-cst-data-access]].

합성음 세트(실제 대화 확보 전 주 세트) 구축 계획: [[output-cst-bench-synthetic-plan-20261003]].
