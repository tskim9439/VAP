---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-11-08
created: 2026-10-08
updated: 2026-10-08
summary: 평가 녹음은 학습에 넣지 않는다 — VoxPopuli 는 학습 허용(평가셋과 겹치는 문장 제거), Europarl-ST 는 학습 제외, 영→독 평가는 겹침 없는 부분집합
sources:
  - '[[output-eval-set-plan-asr-simulst-20261007]]'
related:
  - '[[decision-cst-paper-v1-scope]]'
  - '[[output-paper-draft-cst-s2tt-icml-20261005]]'
  - '[[task-semcommit-main-labeling]]'
---

# 평가-학습 데이터 겹침 방지 (2026-10-08)

## 맥락
- 표준 평가셋 계획([[output-eval-set-plan-asr-simulst-20261007]])에 따라 사용자가 Europarl-ST v1.1(영↔독)과 YODAS-Granary 독일어를 mxc `VAPKT-data/data/corpora/` 에 받았다(2026-10-08).
- Europarl-ST 와 VoxPopuli 는 같은 유럽의회 본회의 녹음에서 나왔다. 스트리밍 ASR 백본(1 단계)은 VoxPopuli 영어 학습 분할(본회의 580 일, 2009-01-13 ~ 2020-11-12)로 학습됐다.
- 대조 결과(2026-10-08, mxc, 이 세션):

| 평가 분할 | 연설 | VoxPopuli 영어 학습과 같은 날짜 | 문장 대조 |
|---|---|---|---|
| Europarl-ST 영→독 test | 126 | 106 개(날짜 76 일 중 64 일) | 8 단어 이상 문장 1,169 개 중 **329 개(28.1 %)** 가 같은 날짜 VoxPopuli 학습 문장과 8 단어 연속 구절을 공유. 날짜를 바꾼 대조군은 **0 개**. 문장 절반 이상이 겹치는 연설 33 개 |
| Europarl-ST 영→독 dev | 134 | 106 개 | 1,233 개 중 279 개(22.6 %) |
| Europarl-ST 독→영 test | 226 | 207 개(날짜 120 일 중 106 일) | 독일어 음성은 우리 학습에 없음 → 지금은 겹침 없음 |
| Europarl-ST 독→영 dev | 218 | 197 개 | 같음 |

- 대조군이 0 이므로 같은 날짜의 겹침은 상투 구절이 아니라 **같은 연설**이다. 즉 Europarl-ST 영→독 평가의 영어 음성 일부를 백본이 학습 때 이미 들었다.

## 원칙
**평가에 쓰는 녹음(같은 연설·같은 발화)은 어떤 학습 단계에도 넣지 않는다.** 데이터셋 이름 단위로 금지하지 않고, 겹치는 녹음 단위로 막는다. 모든 비교 시스템은 같은 평가 부분집합으로 채점한다.

## 결정

| 데이터 | 결정 |
|---|---|
| **VoxPopuli(영어)** | 학습에 **계속 쓴다.** 다음 학습 단계부터 평가셋(Europarl-ST dev/test 양방향 등)과 겹치는 문장을 학습 목록에서 뺀다 |
| **VoxPopuli(독일어)** | 독일어 학습에 넣을 경우, 넣기 **전에** Europarl-ST 독→영 dev/test 와 겹치는 문장을 뺀다 |
| **Europarl-ST** | **학습에 쓰지 않는다**(train·train-noisy 포함). 처음 보는 도메인 평가 세트로만 쓴다. 공식 분할은 서로 나뉘어 있어 규칙 위반은 아니지만, Hibiki-Zero 등과의 비교가 "같은 도메인 학습 여부"에 좌우되지 않게 한다 |
| **Europarl-ST 영→독 평가** | 백본이 이미 겹치는 연설을 들었으므로, **겹침 없는 부분집합**으로 평가한다. 기준: 같은 날짜 VoxPopuli 학습 문장과 8 단어 구절을 하나라도 공유하는 문장이 있는 연설은 통째로 뺀다. 남는 연설·문장 수를 논문에 적는다 |
| **Europarl-ST 독→영 평가** | 그대로 쓴다(독일어 VoxPopuli 를 넣게 되면 위 규칙으로 다시 확인) |
| **YODAS-Granary**(유튜브) | 겹칠 위험은 낮지만 학습 전에 같은 방법으로 평가셋(Europarl-ST, FLEURS 등 텍스트가 있는 세트)과 대조한다 |
| **VoxPopuli-AA**(영어 ASR) | 문장이 아니라 화자가 겹친다(207 명 중 80 명). 본문 대표 세트에서 빼고 부록에 주의와 함께 둔다(기존 결정 유지) |
| 기존 제외 목록 | Earnings-22, TurnBench dev/test, semcommit 골드 v1 스트림(LibriSpeech test-clean, GigaSpeech test, Kspon eval_clean·eval_other)은 계속 학습에서 뺀다([[task-semcommit-main-labeling]]) |

## 겹침 판정 방법
- 원천 텍스트를 소문자·영숫자만 남겨 단어로 나누고, 평가 문장의 **8 단어 연속 구절(8-gram)** 이 같은 날짜(유럽의회처럼 날짜·세션 ID 가 있으면) 또는 전체 학습 텍스트에 있는지 본다.
- 매번 **날짜를 바꾼 대조군**을 같이 돌려 우연 일치율을 확인한다(이번에는 0).
- 음성만 있고 텍스트가 없는 데이터는 원본 ID·URL·세션 ID 로 대조한다.

## 결과
- 1 단계 백본은 이미 겹치는 영어 연설을 학습했다 → 영어 원천 Europarl-ST 결과는 반드시 겹침 없는 부분집합으로 보고한다. 부분집합이 작아 신뢰구간이 넓어진다.
- 다음에 만들 것(사용자 요청 시):
  - VoxPopuli(영·독)에서 Europarl-ST dev/test 와 겹치는 문장 목록(학습 제외용)
  - Europarl-ST 영→독 겹침 없는 부분집합 목록과 통계
- 논문: 평가 데이터 절(부록)에 이 규칙과 부분집합 크기를 적는다.

## 재검토
- 2026-11-08 또는 새 학습 코퍼스를 추가할 때.
