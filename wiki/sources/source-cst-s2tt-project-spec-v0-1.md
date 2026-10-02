---
type: source
status: active
created: 2026-10-02
updated: 2026-10-02
summary: CST-S2TT 프로젝트 명세 v0.1 — 2인 교차언어 대화의 동시 음성→텍스트 번역 + 상호작용 제어, CST-Bench·기준선·kill criterion
raw_path: raw/sources/CST_S2TT_Project_Spec_v0.1.pdf
observed: 2026-10-02
raw_authors:
  - tskim
---

# CST-S2TT Project Specification v0.1

## 무엇인가

- 사용자가 2026-10-02 에 올린 연구 프로젝트 명세서다(9 쪽, LibreOffice 로 만든 PDF).
- 이 볼트의 **새 프로젝트 목표**를 정의한다. 이전 목표(스트리밍 ASR + turn-taking 투사)를 대체한다. → [[decision-project-goal-cst-s2tt]]
- 원본 sha256 `2c334b61…76185`. 텍스트 추출(pypdf)로 읽었다. 그림은 없고 타임라인은 텍스트 도식이다.

## 핵심 내용

### 과제 정의
- 이름(가칭): **Conversational Simultaneous Speech-to-Text Translation (CST-S2TT)**. 벤치마크 가칭 **CST-Bench**.
- 상황: 서로 다른 언어(X, Y)를 쓰는 두 화자가 **하나의 대화 문맥**에서 multi-turn 으로 대화한다. AI 통역기가 양쪽 음성 스트림을 실시간으로 보고, 적절한 시점에 상대 언어의 **텍스트 번역**을 낸다.
- 번역만이 아니라 다음을 실시간으로 판단하는 것이 과제다.
  - 지금 누가 말하는가
  - 번역 방향(A→B / B→A)
  - 진짜 턴인가 맞장구(backchannel)인가
  - 의미가 충분히 모였는가
  - 쓰기/기다리기(WRITE/WAIT)
  - 계속/중단/전환(CONTINUE/STOP/SWITCH)
- 개념 모형: P(Y_t, A_t, S_t | X_≤t, H_t). Y = 번역 토큰, A = 상호작용 행동, S = 화자·턴 상태, X = 지금까지의 음성, H = 대화 이력.
- 핵심 메시지: 기존 동시통역(SimulST)은 **음성 스트림**을 번역하고, 이 연구는 **대화**를 동시에 번역한다.
- 핵심 가설: 대화형 동시통역에서 **번역 정책과 turn-taking 정책은 독립이 아니다**.

### 범위
- 포함:
  - 2 인, 서로 다른 언어, 같은 대화 문맥, 실제 multi-turn
  - 스트리밍 입력, 양방향 동시 S2TT
  - semantic commitment, 턴 전환, 맞장구, 끼어들기(early turn)
- 첫 논문에서 제외:
  - TTS·보코더·S2ST, 음성 자연성(MOS·화자 유사도)
  - 3 인 이상, 화자 분리(diarization)
  - code-switching 을 핵심 과제로 다루는 것
- S2ST 는 후속 확장 또는 데모로 둔다.

### 기존 SimulST 와의 차이

| | 기존 SimulST | CST-S2TT |
|---|---|---|
| 원천 화자·번역 방향 | 주어짐·고정 | 실시간 판단·전환 |
| 입력 | 단일 스트림·발화 | 대화 전체 |
| semantic commit | 일부 연구 | 핵심 문제 |
| 맞장구·끼어들기 | 보통 무시 | 명시적 처리 |
| 턴 전환 | 외부 제어 | 모델·시스템이 판단 |

### CST-Bench
- **Natural Conversation Set**: 실제 대화 그대로 둔다. 번역 품질, semantic commit, 지연, 방향 전환, 대화 문맥 활용을 본다.
- **Controlled Interaction Set**:
  - 실제 대화의 내용·화자·턴 의존성은 유지하고 **상호작용 타이밍만 합성**한다.
  - 합성 유형: Normal, Early Turn, Interruption, Backchannel(“네/응/yeah/uh-huh” 등 삽입), Explicit Barge-in(문맥 조건 LLM + TTS 로 “잠시만요/Wait” 생성).
  - 끼어들기 시점은 무작위가 아니라 semantic chunk 경계 주변에서 고른다.
  - 난이도: Easy = 턴 끝 직전, Medium = chunk 경계 부근, Hard = 핵심 의미 미완성 시점.
- **원본 데이터 조건**:
  - 필수: 같은 세션, 실제 화자 2 명 이상, 세션 안에서 서로 다른 언어, multi-turn, 공유 문맥, 턴별 음성, 화자 ID·언어, 턴 순서, 전사
  - 권장: 참조 번역, 턴 시각, 세션 단위 음성, 화자별 채널
  - 배제: 남의 문장을 따로 낭독한 병렬 음성, 텍스트 대화를 화자별로 따로 녹음한 것, 서로 다른 언어 코퍼스를 단순 합친 것, 대화 순서·문맥을 복원할 수 없는 발화 모음

### 데이터 확보 후보 (명세 시점)

| 후보 | 언어 | 특징 | 상태 |
|---|---|---|---|
| BAS TAXI | DE↔EN | 실제 전화 대화, 턴 기반, 전사·번역 | 우선 확보 |
| Verbmobil II Multilingual | DE↔EN, DE↔JA | 실제 multi-turn, 사람 통역사 부분집합 | 우선 확보 |
| BAAI CS-Dialogue | ZH/EN | 약 104 h 실제 2 인 자연 대화, 공개 | 프로토타입 |
| Bangor bilingual corpora | ES/EN, CY/EN, CY/ES | 실제 자연 대화 | 보조 프로토타입 |
| ETRI/SiTEC | KO↔EN | 한국어에 가장 직접적 | 배포 경로 확인 필요 |

- 한국어를 자체 수집할 때는 끼어들기까지 녹음할 필요 없이 자연스러운 턴 기반 대화만 있으면 된다. 상호작용은 합성한다.

### Semantic commitment
- 너무 일찍 확정하면 되돌리기 어렵다. 예: “내일 오전에… 아니 오후 세 시에 갈게요.”
- 미확정 텍스트는 수정할 수 있고, 확정 텍스트는 수정할 수 없다.
- 메인 벤치마크는 **revision-free committed translation** 을 기준으로 한다.
- 함께 볼 것: prefix 번역 안정성, 미래 문맥 민감도, semantic chunk 경계, 참조 번역 정렬.

### 모델 방향
- 기존 스트리밍 음성 시퀀스 모델을 Conversational SimulST 로 확장한다.
- 특수 토큰 예: `<NEXT_AUDIO>` `<WAIT>` `<A_TO_B>` `<B_TO_A>` `<CONTINUE>` `<BACKCHANNEL>` `<INTERRUPT>` `<STOP>`. 실제 수는 최소화한다.
- 벤치마크는 구조를 강제하지 않는다. 무엇을, 언제, 어느 방향으로 냈는지와 상호작용 사건에서의 행동만 평가한다.

### 기준선과 지표
- 기준선:
  - B0 오프라인 ST(품질 참조)
  - B1 2×SimulST(A→B, B→A)
  - B2 VAD + 2×SimulST
  - B3 VAP + 2×SimulST
  - B4 Oracle 턴 경계 + 2×SimulST(상한)
  - B5 제안 joint 모델
- 지표:
  - 번역 품질: COMET·BLEU·chrF
  - 지연: AL·LAAL·첫 토큰 지연·semantic commit 지연
  - 상호작용:
    - 턴 방향 정확도(A→B/B→A/NONE)
    - 잘못된 중단·전환 비율
    - 놓친 끼어들기 비율, 잘못된 방향 번역 비율
    - Stop Latency = 이전 방향 마지막 토큰 시각 − 끼어들기 시작
    - Switch Latency = 새 방향 첫 토큰 시각 − 새 턴 시작
- 목표: 기존 시스템이 번역은 잘하지만 **대화 수준 상호작용에서 실패한다**는 것을 정량으로 보이는 것.

### Kill criterion
- joint 구조를 만들기 전에 “VAP + 2× strong SimulST 파이프라인만으로 Controlled Interaction Set 의 거의 모든 상호작용 지표가 풀리는가”를 먼저 검증한다.
- **계속할 근거**:
  - 맞장구에서 잘못된 전환이 남는다
  - 끼어들기 대응이 늦다
  - 의미 정정에서 너무 이른 commit 오류가 난다
  - 빠른 턴 전환에서 방향 오류가 는다
  - oracle 턴 정보로 크게 개선된다
- **재검토할 근거**: strong pipeline 이 거의 다 풀거나, 상호작용 인지 모델링이 번역·지연에 실질 이득이 없는 경우다.

### 단계

| 단계 | 내용 |
|---|---|
| Phase 0 | 접근 가능한 교차언어 코퍼스 확보, 공통 schema parser |
| Phase 1 | 턴 재구성, 화자·언어 매핑, 번역 정규화, 합성 타임라인 생성기 |
| Phase 2 | CST-Bench v0: 두 세트, 지표, revision-free commit 프로토콜, 모델 독립 평가 도구 |
| Phase 3 | strong 기준선(2×SimulST·VAD·VAP·Oracle)과 한계 분석 |
| Phase 4 | joint 모델(턴·commit·번역)과 ablation |
| Phase 5 | 한국어 확장(ETRI 또는 KO↔EN 소규모 수집) |
| Phase 6 | S2ST 확장(스트리밍 TTS 연결) |

- **현재 최우선**: 모델 개발보다 CST-Bench v0 와 strong pipeline 기준선을 먼저 완성한다.
- 첫 질문은 “대화 수준 상호작용을 명시적으로 다루지 않는 기존 SimulST 파이프라인은 실제로 어디서, 얼마나 실패하는가?”다.

### 예상 기여와 성공 기준
- 기여:
  1. 과제 정의
  2. 벤치마크(실제 대화 내용 + 통제된 상호작용 타이밍)
  3. 지표 프로토콜
  4. 번역 정책과 turn-taking·commit 정책을 함께 모델링하는 스트리밍 구조
- 성공 기준:
  - 2×SimulST 가 자연 대화에서는 잘 되지만 통제 상호작용에서 뚜렷이 떨어진다
  - VAP 류 모듈이 일부 개선하되 격차가 남는다
  - joint 모델이 번역 품질을 크게 희생하지 않고 상호작용 지표를 유의하게 개선한다
  - 여러 언어쌍에서 반복된다

## 이 볼트에 준 영향

- 새 개념: [[conversational-simultaneous-s2tt]], [[cst-bench]]
- 새 결정: [[decision-project-goal-cst-s2tt]]
- 갱신: [[overview]], `README.md`, `PLAN.md`, `TODO.md`, [[status]]
- 새 태스크: [[task-cst-data-access]], [[task-cst-conversation-schema-parser]], [[task-cst-timeline-generator]], [[task-cst-bench-v0-metrics]], [[task-cst-strong-baselines]], [[task-cst-kill-criterion-check]]

## 불확실성 / 미확인

- 명세의 데이터 후보 상태(“우선 확보”, “배포 경로 확인 필요”)는 작성자 판단이다. 이 볼트에서 접근 가능 여부·라이선스를 아직 확인하지 않았다.
- 특수 토큰 목록은 예시이고, 명세 스스로 “실제 수는 최소화”라고 적었다.
- 9 절의 예시 문장(“3B 모델을 사용하고 / …”)은 semantic chunk 분할 예시로 보인다. 앞뒤 맥락이 잘려 있어 해석에 불확실성이 있다.

## 출처
- 원본: `raw/sources/CST_S2TT_Project_Spec_v0.1.pdf` (관측일 2026-10-02, 사용자 제공)
