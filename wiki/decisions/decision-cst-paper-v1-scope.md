---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-10-19
created: 2026-10-05
updated: 2026-10-05
summary: CST 첫 논문 범위 — mono 단일 입력, 끼어들기·맞장구·정보 인과성 제외, 실제 세트는 TAXI 만, S2TT 베이스라인, 모델은 ASR·2인 diarization·양방향 번역 동시 수행
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
related:
  - '[[decision-project-goal-cst-s2tt]]'
  - '[[cst-bench]]'
  - '[[output-cst-bench-syn-final-plan-20261003]]'
  - '[[output-paper-draft-cst-s2tt-icml-20261005]]'
---

# CST 첫 논문(v1) 범위 결정 (2026-10-04~05)

## 맥락
- 명세 v0.1([[source-cst-s2tt-project-spec-v0-1]])과 합성 세트 계획 v1.1([[output-cst-bench-syn-final-plan-20261003]])은 끼어들기·맞장구·정보 인과적 사건 생성까지 포함했고, 화자 분리(diarization)는 범위에서 뺐다.
- 2026-10-04~05 사용자와 관련 연구(JSTAR, Seed LiveInterpret 2.0, DiariST, CS-Dialogue 등)를 검토하면서 범위를 다시 정했다. 근거 대화는 이 세션 기록이며, 아래 항목은 모두 사용자가 직접 정했거나 동의한 것이다.

## 결정

| 항목 | v1 결정 | 이전 |
|---|---|---|
| 입력 | 두 화자가 섞인 **단일 채널(mono) 스트림** 하나 | 명세: 화자별 스트림 관찰 |
| 상호작용 사건 | **끼어들기·barge-in·맞장구·정보 인과성 제외**(향후 과제). 조건은 L0 간격(natural·mediated)과 L1 턴 끝 겹침(다음 턴이 앞 턴 끝 0.2–0.8 s 전에 시작). 맞장구(L1b)는 부록으로도 넣지 않는다(2026-10-05 사용자 결정) | 명세: Early Turn·Interruption·Backchannel·Barge-in, 정보 인과적 생성 |
| 실제 데이터 | **TAXI 만**. 실제 겹침이 없어도 된다(리뷰어가 이해할 수준의 한계). 겹침 주장은 합성 세트와, TAXI 원음에 턴 끝 겹침을 준 변형으로 한다 | VM2·AI Hub 71686 등 후보 |
| 언어 | 영↔독(주, 양방향 S2TT 베이스라인이 가장 많음) + 한↔영(어순 차이). 중국어(CS-Dialogue 등)는 넣지 않는다 | – |
| 베이스라인 | **S2TT 를 지원하는 시스템**(음성 생성 전용 모델은 주 베이스라인이 아님). 단방향 시스템은 정답 분할·실제 조건(VAD + 언어 판별) 래퍼로 비교 | 명세 B0–B5(VAP 라우팅 포함) |
| 모델 출력 | 하나의 모델이 **스트리밍 ASR + 2인 화자 귀속(diarization) + 양방향 동시 번역**을 함께 낸다. 평가는 2 인 중심 | 명세: diarization 제외 |
| 차별점 | 선행 시스템(JSTAR, Seed LiveInterpret 2.0)이 시스템 수준에서 같은 방향을 보였으므로, 주 기여는 공개 벤치마크·평가 프로토콜·공개 베이스라인과 공개 통합 모델. "단일 채널"은 Seed 대비 차별점이 아니다 | – |

## 결과
- 논문 초안 구성: [[output-paper-draft-cst-s2tt-icml-20261005]]
- 합성 세트 계획 v1.1 의 4 절(정보 인과적 사건)과 끼어들기 관련 부분은 보류된다. v1.2 갱신 필요.
- VAP 라우팅 베이스라인(B3)은 재현해 둔 VAP 가 스테레오 입력이라 v1 에서 돌리지 않는다.
- [[decision-project-goal-cst-s2tt]] 의 "재검토가 필요한 기존 결정" 중 입력 채널(mono 유지)과 다화자 범위(2 인 고정)가 이 결정으로 정리됐다.

## 재검토
- 2026-10-19: 합성 세트 1차 결과와 TAXI 베이스라인 결과를 보고 한↔영 위치(본문/부록)를 다시 본다.
