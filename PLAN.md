# PLAN — CST-S2TT 단계별 실행 계획

> README 의 단계를 **실행 단위**로 푼 문서다.
> 단계마다 목표 · 입력 · 산출물 · 관문 · 연결 태스크를 둔다.
> - 근거: `raw/sources/CST_S2TT_Project_Spec_v0.1.pdf` → `wiki/sources/source-cst-s2tt-project-spec-v0-1.md`
> - 전환 결정: `wiki/decisions/decision-project-goal-cst-s2tt.md`
> - 이전 계획(스트리밍 ASR + turn-taking 투사): `plans/PLAN-vap-asr-20260905.md`
>
> 갱신 규칙: 관문을 통과·실패하거나 데이터·평가 정의가 바뀔 때 갱신한다. 일일 진행은 `wiki/tasks/` 와 `wiki/status.md` 에 쓴다.

**최종 갱신: 2026-10-02** · 현재 단계: **Phase 0 시작**

---

## 원칙

1. **벤치마크와 기준선이 모델보다 먼저다.** 첫 질문은 “상호작용을 다루지 않는 SimulST 파이프라인은 어디서, 얼마나 실패하는가”다.
2. **실제 대화 내용 + 통제된 타이밍.** 내용·화자·턴 의존성은 실제 대화에서 가져오고, 상호작용 조건(끼어들기·맞장구·빠른 전환)만 합성해 조건별 실패를 분리한다.
3. **확정 번역만 채점한다(revision-free commit).** 수정 가능한 미확정 출력은 품질 점수에 넣지 않는다.
4. **구조를 강제하지 않는 평가.** 무엇을, 언제, 어느 방향으로 냈는지만 본다. 기준선과 제안 모델이 같은 도구로 평가된다.
5. **kill criterion 을 지킨다.** strong pipeline 이 거의 다 풀면 joint 모델로 가지 않고 방향을 재검토한다.
6. **기존 운영 규칙 유지.**
   - 서버 삭제 금지
   - 외부 데이터는 T5 에만 받고 업로드는 사용자가 한다
   - SLURM 제출은 사용자가 한다
   - 짝 bootstrap 으로 유의성을 판단한다

---

## Phase 0 — Feasibility / Data Access

- **목표**: 실제 2 인 교차언어 multi-turn 대화 코퍼스를 확보하고 접근 경로·라이선스를 정리한다.
- **후보**: BAS TAXI(DE↔EN) · Verbmobil II(DE↔EN, DE↔JA) · BAAI CS-Dialogue(ZH/EN) · Bangor(ES/EN, CY/EN, CY/ES) · ETRI/SiTEC(KO↔EN)
- **판정 기준**: 명세 6절의 필수 조건(같은 세션·다른 언어·multi-turn·턴별 음성·화자 ID·언어·턴 순서·전사)과 배제 조건
- **산출물**: 코퍼스별 접근·라이선스 표, 첫 코퍼스 로컬 확보, 공통 conversation schema 초안
- **관문**: 필수 조건을 만족하는 코퍼스 1 개 이상 확보. 못 하면 Phase 1 은 그 코퍼스 없이 schema·생성기만 만든다
- **태스크**: `task-cst-data-access`, `task-cst-conversation-schema-parser`

## Phase 1 — CST Data Pipeline

- **목표**: 원본 → 공통 schema, 그리고 통제 상호작용 샘플 생성
- **작업**:
  - 턴 재구성, 화자·언어 매핑, 번역 정규화
  - 턴 단위 음성 정규화(16 kHz)
  - semantic chunk 분할
  - 타임라인 생성기: Normal · Early Turn · Interruption · Backchannel · Explicit Barge-in. 난이도 Easy/Medium/Hard 는 chunk 경계 기준
- **이어 쓰는 자산**:
  - `<SEM_END>` 의미 완결 라벨(`output-semcommit-recipe-v0.3`) → chunk 경계 후보
  - 교사 LLM 라벨링 파이프라인 → 번역 정규화·Barge-in 문장 생성. 판정 LLM 은 EXAONE-4.0·Qwen3.8-27B 만 쓴다
- **결정할 것**: 입력 채널. 기존 결정은 mono 혼합(`decision-mono-input`)이다. 명세는 두 스트림 관찰·화자 분리 제외라 화자별 채널이 자연스럽다
- **관문**: 생성 표본 청취 점검. 사건 라벨(시각·유형·난이도)과 음성이 일치해야 한다
- **태스크**: `task-cst-timeline-generator`

## Phase 2 — CST-Bench v0

- **목표**: Natural Set · Controlled Interaction Set 과 모델 독립 평가 도구
- **지표**:
  - 번역 품질: COMET·BLEU·chrF(확정 텍스트만)
  - 지연: AL·LAAL·첫 토큰·semantic commit 지연(창 없는 commit 지연 지표 확장)
  - 상호작용: 방향 정확도·잘못된 중단/전환·놓친 끼어들기·잘못된 방향 번역·Stop/Switch Latency
- **관문**: 오라클 출력(정답 번역을 정답 시각에 낸 가상 시스템)이 지표 상한을 내고, 의도적으로 망가뜨린 출력이 해당 지표만 떨어뜨리는지 확인한다(지표 sanity)
- **태스크**: `task-cst-bench-v0-metrics`

## Phase 3 — Strong Baselines

- **목표**: B0–B4 구성과 파이프라인 한계 분석
- **구성**:

  | 기준선 | 구성 |
  |---|---|
  | B0 | 오프라인 ST |
  | B1 | 2×SimulST |
  | B2 | VAD 라우팅 |
  | B3 | VAP 라우팅(재현 VAP) |
  | B4 | Oracle 턴 경계 |

- **원천 인식 후보**: 이 볼트의 스트리밍 백본(`output-stage1-pilot-eval-20261001`, Stage 1 본학습 q17-s1m-*)과 공개 SimulST 모델
- **관문 = kill criterion**: B3 가 통제 세트 상호작용 지표를 거의 다 풀면 재검토한다. 다음 실패가 남고 oracle 로 크게 개선되면 Phase 4 로 간다.
  - 맞장구 오전환
  - 끼어들기 지연
  - 조기 commit 오류
  - 빠른 전환 방향 오류
- **태스크**: `task-cst-strong-baselines`, `task-cst-kill-criterion-check`

## Phase 4 — Joint Model

- **목표**: 번역·turn-taking·commit 정책을 함께 모델링하는 스트리밍 모델(B5)과 ablation
- **출발점**:
  - 인과 인코더 + Qwen3-ASR thinker 의 δ 지연 interleave 시퀀스
  - 번역 출력과 소수의 상호작용 토큰을 더한다(예: `<WAIT>` `<A_TO_B>` `<B_TO_A>` `<STOP>`, 수는 최소화)
  - 원천 `<SEM_END>` commit 을 번역 WRITE 신호로 쓰는 변형을 비교한다
- **성공 기준**: 번역 품질을 크게 희생하지 않고 상호작용 지표를 유의하게 개선한다. oracle 격차의 상당 부분을 외부 턴 라벨 없이 회복한다

## Phase 5 — Korean Extension

- ETRI KO↔EN 확보 또는 소규모 자체 수집. 끼어들기는 녹음하지 않고 자연스러운 턴 기반 대화만 모은다(상호작용은 합성)
- 이전 목표의 한국어 ASR 자산(Kspon 평가·한국어 학습 데이터)을 원천 인식에 쓴다

## Phase 6 — S2ST Extension

- 스트리밍 TTS 연결 데모·후속 연구. 첫 논문 범위 밖

---

## 관문 요약

| 관문 | 조건 | 통과 못 하면 |
|---|---|---|
| 데이터 | 필수 조건을 만족하는 교차언어 대화 1 개 이상 | 접근 경로 재탐색, 소규모 수집 검토 |
| 지표 sanity | 오라클 상한·의도적 손상 출력의 지표 반응 | 지표 정의 수정 |
| Kill criterion | B3 가 상호작용 지표를 거의 다 풂 | joint 모델 보류, 방향 재검토 |
| Joint 성공 | 품질 유지 + 상호작용 유의 개선 + 여러 언어쌍 반복 | ablation 으로 원인 분리 |
