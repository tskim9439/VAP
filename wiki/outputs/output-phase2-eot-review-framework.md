---
type: output
status: active
created: 2026-09-13
updated: 2026-09-14
summary: Phase 2 경량 EOT 검수 작업안 — TurnBench dev 대조·AMI 200경계; 규약은 정본, 전수 이중 검수·HTML 인프라는 이월
sources:
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[output-phase2-real-sequence-probe]]'
  - '[[output-phase2-sequence-replay]]'
  - '[[output-phase2-sequence-critique-response]]'
  - '[[source-turnbench]]'
---

# Phase 2 EOT 검수: 최소 실행 범위

## 역할과 변경

이 문서는 **검수 작업 안내**다. 사건 정의·P/C·registry·직렬화·loss mask는 [[output-phase2-streaming-asr-diarization-plan]] §4–6·9 한 곳만 따른다. 2026-09-14 비판을 반영해 종전 F0–F4 전용 라벨링 플랫폼 구축을 Q1 선행 조건에서 제외했다.

Q0/F2 범위는 **TurnBench dev gold 대조 + AMI 200경계**다. KO 실외 1대화의 실제 데이터 경로 QA는 별도 필수 구현 점검이다. 전 코퍼스 200개씩, 전수 2인 검수, 별도 HTML UI, LLM 판정, 독립 릴리스 도구는 지금 만들지 않는다. 의미 EOT/HOLD/BC 대규모 라벨링은 Stage 3에 남긴다.

아직 실제 검수를 수행하지 않았다. 기존 probe/replay는 과거 P-mode 후보와 독립 JSON의 실험 기록이며 검수된 C-mode 학습 데이터가 아니다.

## 1. 입력을 고정한다

- 원 대화 ID·원 화자 ID·오디오/주석 해시·split과 E2 노출 이력을 기록한다.
- 정본의 새 C-rule 버전, TN/registry/serializer 버전을 고정하고 이전 P-rule 결과와 구분한다.
- TurnBench dev는 학습에 넣지 않는다. 결과를 보고 규칙을 수정하면 이후 결과는 튜닝된 dev다. 기존 반복 사용 사실도 남기고 독립 test로 부르지 않는다.
- AMI ES2002a의 공개 진단 구간은 이미 노출됐으므로 오류 재현용이다. 신규 표본과 분리한다.
- source 음성/라벨은 수정하지 않는다. 새 결과 경로에 JSONL·요약만 저장한다.

## 2. TurnBench dev에서 자동 대조한다

공식 라벨 매핑·scorer를 재사용한다. 임의의 SRT 끝이나 화자 변경을 공식 EOT gold로 바꾸지 않는다. mono 입력 조건을 명시하고 비교 기준을 고정한다. [[source-turnbench]]

두 종류의 결과를 분리한다.

1. **라벨 후보 비교:** offset 기준 사건 매핑, 오귀속·누락·불확실·후보 coverage. 규칙을 검증하는 값이다.
2. **방출 시각 비교:** C의 실제 증거 가용 시각 이후 선언을 공식 scorer에 넣는다. 3초 horizon 대기가 공식 매칭 창을 넘기는 사례도 그대로 실패로 보고한다. offset으로 소급해 점수를 높이지 않는다.

교사 규칙의 점수는 모델 성능이 아니다. C의 저지연 한계와 P/head ablation의 필요성을 판단하는 기준선으로 사용한다. 그 결과로 학습 checkpoint 성능을 주장하지 않는다.

## 3. AMI 200경계를 점검한다

| 표본 | 수 | 목적 |
|---|---:|---|
| 자동 EOT 후보 무작위 | 100 | 후보 조건부 precision·오귀속·경계 오차 |
| 자동 no_eot 경계 무작위 | 50 | 놓친 사건/모호한 pause의 유형 |
| 재개·짧은 겹침·다자 충돌 위험 표본 | 50 | 실패 원인 탐색; 무작위 precision 분모와 분리 |

seed·대화별 개수·추출 집합을 기록하고 중복을 제거한다. C 규칙에서 positive가 부족하면 억지로 채우지 않고 가용 개수와 coverage를 보고한다. 200개가 **전체 EOT recall, 전 코퍼스 라벨 품질, 다자 floor 의미의 보증**은 아니다. no_eot 경계 밖에서 아예 누락된 사건은 이 설계만으로 셀 수 없다.

기존 오디오 재생 도구로 전후 문맥을 듣고 JSONL 또는 TSV에 응답한다. 기본 앞 8초·뒤 5초, 부족하면 원 대화 문맥을 더 확인한다. 자동 판정은 가능하면 첫 청취 때 숨긴다. 별도 브라우저/서버는 필수가 아니다.

기본 검수자 1명이 타이밍·귀속을 확인하고 모호한 사례만 추가 검토한다. 추가 검토자가 없거나 합의가 안 되면 uncertain으로 남긴다. 이 결과로 사람 간 κ/독립 일치도를 보고하지 않는다. 문법적 이어짐만으로 floor 종료를 반박하지 않는다.

## 4. 최소 기록 계약

`conversation_id, speaker_id, boundary_sample, candidate_kind, rule_version, label_observed_until_sample, target_emit_sample, sampling_stratum, review_status, verdict, reason, reviewer_id`를 보존한다. verdict는 `accept / reject / uncertain`이며 검수자가 직접 토큰 방출 위치를 수정하지 않는다. 경계 수정과 사건 정오를 따로 기록하고 재직렬화는 공용 모듈이 수행한다.

별도 유지할 필드는 `provenance=weak_auto/human_reviewed/source_human_mapped`, 관측/정렬 품질·horizon mask, `event_complete`다. unknown을 no_eot로 바꾸지 않는다. 새로운 raw 결과는 버전/해시를 기록하고 이전 후보 파일을 덮어쓰지 않는다.

검수한 경계 하나가 주변 창 전체의 완전 감독을 의미하지 않는다. 작은 표본 QC를 통과한 자동 라벨도 계속 weak label이며, 정본의 관측/라벨 completeness 조건을 통과한 창만 joint에 넣는다. unknown 창의 ASR-only 처리는 정본 §6.1을 따른다.

## 5. 결과와 중단 기준

보고서는 다음 한 장으로 끝낸다.

- 고정 rule/TN/registry 버전과 TurnBench dev 사용 이력.
- TurnBench 후보 비교와 C 방출 평가를 분리한 결과.
- AMI 3개 표본군의 판정 수, 무작위 positive precision과 표본 불확실성.
- 재개·overlap·오귀속·시각 오차·unknown 유형별 사례.
- 원 후보 수 대비 유효 EOT 수, 전체 시간 대비 joint 유효 시간.
- `진행 / 해당 규칙 보류 / 전체 EOT 학습 보류`의 판단과 근거.

source/split/타임스탬프 누출·귀속·serializer 오류는 0건이어야 한다. 후보가 거의 사라졌다면 높은 precision만으로 통과시키지 않는다. 표본이 적으면 통계 관문을 달성했다고 쓰지 않는다. 전용 95%/90% 합격 플랫폼을 먼저 만드는 대신, 이 결과와 사용자의 검토로 Q1 weak-label 범위를 확정한다.

## 6. 실행 순서와 이월

1. 정본 registry·TN·16개 serializer/state 테스트를 구현한다.
2. 71631 실외 1대화의 정렬·2채널 복원·VAD·480블록·head targets를 실제 Dataset으로 검사한다.
3. 위 TurnBench dev + AMI 200경계 QC와 언어/동시 화자별 cap 밀도를 측정한다.
4. Q1 C-mode 32창 overfit 및 HF save/load를 확인한다.

전용 검수 UI, 전수 이중 판정, 독립 라벨 릴리스 시스템, 모든 코퍼스의 대규모 경계/연속 창 gold 구축은 **필요성이 실측된 이후 또는 Stage 3**으로 이월한다. 이번 축소로 독립 라벨 품질 보증을 확보한 것처럼 주장하지 않는다.

## 실험 기록

- [[output-phase2-real-sequence-probe]] — 실제 AMI 38.4초, 과거 P-mode 후보 생성.
- [[output-phase2-sequence-replay]] — 같은 보존본 480블록 재직렬화. `event_complete=false`, 사람 검수 미실행.
- [[output-phase2-sequence-critique-response]] — 8개 지적의 대조 결과와 채택/보완 근거.
