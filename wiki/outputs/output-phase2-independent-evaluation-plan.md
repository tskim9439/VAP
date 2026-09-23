---
type: output
status: active
created: 2026-09-17
updated: 2026-09-17
summary: 독립 평가안 — 전사·화자 귀속·음향 경계·턴 종료 분리, 세션 순열 불변 채점, 미학습 DB 잠금과 검증 fixture·실행 순서
sources:
  - '[[output-phase2-lane-plan]]'
  - '[[output-phase2-d1-lane-eval]]'
  - '[[task-phase2-data-prep]]'
  - '[[output-phase2-data-inventory]]'
  - '[[output-phase2-training-db]]'
  - '[[source-turnbench]]'
---

# Phase 2 독립 평가 계획: 무엇을 말했는가, 누가 말했는가, 언제 시작하고 끝났는가

## 0. 제안의 범위와 결론

**전사 내용·화자 귀속·발화 경계·턴 종료를 별도 축으로 채점하고, 사람별 최종 전사 품질을 종합 지표로 추가한다.** A/B 이름만 통째로 바뀐 것은 오류로 세지 않는다. 반대로 대화 중간에 같은 사람의 말을 다른 사람에게 붙인 것은 전사와 별개로 화자 귀속 오류에 남긴다.

사용자의 요청에 따라 작성한 **새 독립 보고서**다. 기존 정본·보고서·평가 코드·학습 잡은 수정하지 않았다. 아래 지표·split·관문은 구현 제안이며, 아직 새 평가 결과를 얻었다는 뜻이 아니다. 수치는 별도 표시가 없으면 초기 규약 제안이다.

최신 상태도 구분한다. 2026-09-17 보고서에는 동결 E2 encoder의 저장 누락을 고친 `final-enc`와 D1b 준비가 추가됐다. 과거 잘못 재로드한 D1과 수정본을 같은 모델로 합산하지 않는다. **평가 v2의 첫 관문은 올바른 checkpoint 재로드 parity**다. [[output-phase2-d1-lane-eval]], [[task-phase2-data-prep]].

## 1. A/B 교환과 진짜 화자 혼동은 다르다

설명용 예시다. 아래 이름은 실제 사람이고 A/B는 시스템의 임의 라벨이다.

| 상황 | 전사 내용 점수 | 화자 점수 / 사람별 전사 점수 |
|---|---|---|
| 처음부터 끝까지 민수=A·지수=B 대신 민수=B·지수=A | 동일해야 함 | **동일해야 함**. 세션 전체 순열로 해결 |
| 첫 발화 하나를 놓친 뒤 나머지 번호가 일관되게 반대 | 놓친 내용만 deletion | 놓친 활동/사람은 오류. 이후의 일관된 이름 교환은 추가 오류 아님 |
| 처음에는 민수=A, 중간부터 민수의 발화를 B에 붙임 | 단어가 맞으면 내용 오류와 분리 | 중간 귀속 변화는 오류. 매 발화마다 정답에 재매핑해 지우지 않음 |
| 민수와 지수 전사를 모두 A에 합침 | 내용은 잘 맞을 수 있음 | 사람 병합·겹침 귀속 오류가 남아야 함 |
| lane 재사용으로 A의 generation이 바뀜 | 전사 흐름은 계속 평가 | lane A를 영구 사람 ID로 채점하지 않음. 외부 재매핑 결과로 평가 |

**화자 ID는 이름표이지 정답 단어가 아니다.** 세션 전체의 일관된 이름 교환을 허용하는 것은 정답을 모델에 제공하는 것이 아니라 평가 시 임의 라벨을 대응시키는 절차다. 이 대응은 추론 입력으로 되먹이지 않는다.

## 2. 현재 평가에서 바꿀 부분

[p2_eval_lanes.py](../../experiments/p2_eval_lanes.py)를 확인한 결과:

1. `lane_rate`는 같은 lane 번호끼리 비교한다. 명칭 교환에 취약하므로 주 지표에서 제외하고 구현 진단에만 남긴다.
2. `mapped_rate`는 이미 순열을 시도하지만 **시간 겹침 최대** 매핑이다. word edit distance를 최소화하는 cpWER가 아니다. 참조·가설 사람 수가 다를 때 dummy 대응까지 포함하는 검증된 평가기로 대체해야 한다.
3. `pooled_rate`는 화자들의 BPE를 시각순으로 섞어 decode한다. 겹친 단어 조각과 출력 순서가 섞일 수 있어 순수 내용 지표로 삼지 않는다.
4. ONSET/EOT 매칭은 화자 귀속과 의미 종료 label을 충분히 분리하지 않는다. 모든 EOT 후보를 gold 종료로 세면 p=0·미관측 후보의 의미가 사라진다.
5. 20–40초 창마다 초기화하면 긴 대화의 ID 유지·재배정 오류를 놓친다. 정렬 가능·다화자·무음 시작 창만 골라 평가하는 선택 편향도 제거해야 한다.

## 3. 최종 성적표: 네 축과 종합 점수

| 축 | 주 지표 | 함께 보여줄 것 | 의도 |
|---|---|---|---|
| A. 전사 내용 | **ORC-WER / ORC-CER** | S/D/I, 비중첩·겹침별, 분할/병합 민감도 진단 | 세션 화자 이름에 덜 의존하는 전사 품질 |
| B. 화자 구분 | **DER의 confusion / miss / false alarm 분해** | JER, ID 변경·병합·분절, unresolved 비율 | 텍스트 edit와 분리한 시간·사람 귀속 |
| C. 음향 경계·발화 구간 | **onset/offset P/R/F1**, segment F1 | 경계 오차, IoU, 과분할·과병합, 무음 오방출 | 실제 목소리의 시작과 끝을 잡는가 |
| D. 턴 종료 | **EOT Recall @ 고정 FPR + 지연** | hold/짧은 맞장구 오방출, calibration, 사건 귀속 | 의미·상호작용상 끝난 턴을 적시에 판단하는가 |
| 종합. 사람별 전사 | **cpWER / cpCER** | 신뢰 가능한 시각이 있으면 tcpWER, 최초/수정 후 결과 | 전사와 사람 연결이 합쳐진 최종 품질 |

네 축은 원인을 분리해 읽기 위한 것이지 수학적으로 완전히 독립인 값들은 아니다. 특히 DER에는 활동 검출 오류가, ORC에는 발화 분할 오류가 일부 포함된다. 점수 하나를 다른 점수에서 빼서 순수한 오류 개수로 단정하지 않는다.

### 3.1 내용: ORC를 기본으로, 완전한 귀속 불변은 별도 진단

ORC는 참조 발화를 출력 스트림들에 배정해 전사 edit를 최소화하므로 A/B 명칭이나 발화 단위 귀속 교환의 영향을 줄인다. **한 참조 발화를 여러 출력에 쪼개는 오류까지 모두 무시하는 것은 아니다.** 이 차이를 숨기지 않는다.

보조적으로 DI-cpWER를 계산해 가설 발화의 사람 연결을 재배정했을 때의 전사 하한을 본다. 이 값은 발화 분할에 민감하고 잘게 쪼개면 유리해질 수 있어 **모델 순위·best checkpoint의 단독 기준으로 쓰지 않는다.** ORC와 DI-cpWER의 차이를 활용해 전사 문제와 분할/병합 문제를 조사한다. word-level 완전 불변 정렬은 지나친 재배정으로 오류를 과소평가할 수 있으므로 작은 fixture/분석용에만 둔다. [WER 정의 비교 논문 §IV·VI](https://arxiv.org/html/2508.02112v1).

최소 구현은 ORC + cpWER다. DI 진단은 같은 출력이 준비된 뒤 추가한다. 라이브러리의 greedy 근사와 exact를 결과에서 구별하고, 작은 세션의 exact 대조로 차이를 검사한다. 표준 구현 후보는 [MeetEval](https://github.com/fgnt/meeteval)이다. 라이브러리·버전은 구현 시 고정하며 이번에는 설치하지 않는다.

### 3.2 사람별 전사: cpWER는 세션 전체에서 한 번 매핑

원 대화마다 각 사람의 전사를 모은 뒤, 참조·가설 사람 사이의 **일대일 순열 중 총 edit가 최소인 대응**을 쓴다. 수가 다르면 빈 스트림을 포함해 누락·가짜 사람의 내용을 빠뜨리지 않는다. A/B 전체 교환은 비용 0이고, 중간 ID switch는 일반적으로 하나의 순열로 고칠 수 없어 남는다. [MeetEval 논문](https://www.isca-archive.org/chime_2023/neumann23_chime.html).

- 외부 재매핑 완료 결과의 `speaker_id`를 사용한다. `lane_id`를 세션 사람 ID로 가정하지 않는다.
- 외부 모듈이 아직 없으면 **session cpWER는 미구현**으로 표시한다. N≤R·재배정 없음의 lane 순열 점수는 제한된 baseline으로만 낸다.
- N>R에서 한 lane을 여러 사람이 쓴 원시 결과에는 local ASR 지표를 내고, 외부 연결까지 완성한 결과의 cpWER와 구별한다.
- 세션을 30초씩 자르더라도 scoring의 사람 대응을 매 창 다시 정하지 않는다. 계산 분할은 가능하지만 상태·ID·평가 단위는 원 세션으로 유지한다.

### 3.3 정규화·집계 계약

EN은 동결 TN 후 단어, KO는 동결 TN 후 NFC 한글 문자열의 공백 제외 문자 단위로 한다. KO를 BPE나 자모 단위로 바꾸지 않는다. ORC-CER/cpCER는 **문자 단위로 적용한 프로젝트 규약**임을 표시하고 표준 EN WER 구현과 문자 변환 fixture를 검증한다.

주 집계는 `Σ(S+D+I) / Σ(reference units)`인 micro 평균이다. 창별 비율 평균은 보조로만 두고 세션·코퍼스별 표를 함께 낸다. 참조가 비어 있는 구간의 삽입은 전량 집계의 분자에 포함하고, pure-silence 성능은 false words/chars per minute로 별도 보고한다. EN WER와 KO CER을 합쳐 한 숫자로 만들지 않는다.

공식 벤치마크와 비교할 때는 그 벤치마크 TN·마이크·UEM·scorer로 별도 표를 낸다. 내부 TN 점수를 공식 leaderboard 수치와 직접 비교하지 않는다.

## 4. 화자 구분: 텍스트를 보지 않는 축

참조의 speaker activity와 가설의 activity/세그먼트에서 RTTM을 만들고, **시간 기반 세션 일대일 대응**으로 DER를 계산한다. 이는 cpWER의 텍스트 최적 매핑과 목적이 다른 대응이다. 두 매핑을 서로 재사용하지 않는다. DER는 speaker confusion·miss·false alarm을 각각 표시하고, 짧게 말한 사람의 실패가 묻히지 않도록 JER도 보조로 둔다. [dscore 공식 설명](https://github.com/nryant/dscore).

초기 규약은 **overlap 포함, collar=0초가 주 평가**, collar=0.25초가 보조다. 주 평가에서 겹침을 제외하면 Phase 2 핵심 난도를 숨긴다. UEM 밖/라벨 결손만 제외하며 제외 음성 시간과 이유를 고정한다. overlap 제외 수치는 비교용으로만 붙인다.

추가 진단은 다음과 같다.

- ID switch: 세션 대응을 고정한 뒤 같은 참조 사람이 다른 가설 ID로 바뀐 횟수. 잘못된 사람 병합과 여러 ID로 쪼개짐을 별도 집계한다.
- 외부 ID가 미확정인 구간을 삭제하지 않는다. 고유 provisional ID로 보존하고 coverage와 확정 지연을 함께 낸다. 임의로 모든 unknown을 같은 사람으로 합치지 않는다.
- ASR 내부의 local 구간 품질은 비중첩 유효 구간의 화자 순도와 누락 coverage로 보조 평가한다. 순도만 높이기 위해 어려운 음성을 내지 않는 모델을 좋게 평가하지 않는다.
- 실시간 ID 최초 배정과 세션 종료 후 수정된 배정을 각각 채점한다. 사후 clustering 개선이 실시간 성능으로 둔갑하지 않게 한다.

## 5. 발화 시작·끝·구간: EOT와 분리한다

### 5.1 먼저 세 개의 시간을 출력 계약에 넣는다

| 시간 | 뜻 | 평가 용도 |
|---|---|---|
| `estimated_start/end` | 모델/활동 경로가 추정한 실제 음향 경계 | boundary error·IoU·DER |
| `audio_observed_until` | 판단이 실제로 사용한 마지막 오디오 시각 | 인과성·알고리즘 지연 |
| `emitted_at` / wall-clock | 토큰·이벤트가 소비자에게 가용해진 시각 | 검출·사용자 체감 지연 |

현재 parser의 `(k+1)×80ms`는 **방출 청크 시각**이지 정밀한 음향 경계 추정치가 아니다. 별도 추정이 없으면 경계 정확도는 `미지원`으로 표시하고 검출 지연만 낸다. 정답 시각이나 정답 δ를 빼 경계를 맞춘 것처럼 보고하지 않는다. causal timestamp adapter를 추가한다면 산출 방식·지연·신뢰도를 기록한다.

### 5.2 경계와 segment의 채점

- 참조는 **음향 발화 구간**이다. 발화 단위 텍스트 라벨/forced alignment는 후보 근거이고, 짧은 맞장구·겹침·무음은 별도 검수한다. semantic turn과 동일하다고 가정하지 않는다.
- onset과 offset 각각 오차 허용 **100/250/500ms**, 주 값 250ms에서 P/R/F1. 허용 범위 안 maximum-cardinality 일대일 매칭 후 시간 오차가 최소인 대응을 선택한다. 중복 예측은 FP, 미예측은 FN이다.
- 경계의 signed error와 absolute error p50/p90/p99는 매칭된 사건만 계산하므로, 항상 P/R·매칭 coverage를 붙인다.
- interval IoU의 일대일 매칭으로 segment F1@IoU 0.5/0.75를 낸다. 하나의 예측을 여러 참조에 중복 TP로 주지 않고 과분할·과병합을 별도로 센다. 평균 IoU만으로 평가하지 않는다.
- 먼저 화자 이름을 무시한 geometry-only 결과를 내고, 세션 화자 대응을 요구한 owner-aware 결과를 병기한다. 겹침의 동시 onset은 서로 다른 사건이며 임의로 하나로 합치지 않는다.

방출 시각밖에 없는 초기 구현에는 별도의 **onset 검출 Recall@250/500/1000ms**, 조기 방출률, 무음 분당 false ONSET을 낸다. 이 값은 위 경계 F1의 대체 이름이 아니다. EOT가 출력되지 않은 음향 offset도 활동 경로에서 평가해야 한다.

## 6. 턴 구간·EOT: 무엇이 끝났는지를 따로 정의한다

**숨을 쉬거나 잠깐 멈춘 것은 음향 구간 종료일 수 있지만 턴 종료 정답은 아닐 수 있다.** 학습의 p_end=0.8·0.5 같은 규칙값은 사람이 검수한 종료 확률 gold가 아니다. 그 규칙을 그대로 재사용해 잘 맞았다고 하면 weak-target agreement일 뿐이다.

### 6.1 두 가지 정답 수준

1. **semantic gold:** 검수된 turn 시작/끝, 계속 말하려는 pause, 짧은 맞장구, uncertain 구간. 종료점만 있는 데이터에서는 EOT만 평가하고 full turn interval IoU를 만들지 않는다.
2. **behavioral weak:** 참조 발화의 이후 재개/교대에서 유도한 후보. QA·학습 추세에만 사용하며 semantic gold와 표를 분리한다.

TurnBench에는 공식 gold/scorer를 우선 사용한다. 공식 제출은 사용한 오디오가 모두 가용해진 시각을 요구하고, EOT/INT별 recall·negative span FPR·지연을 채점한다. **dev에서 FPR≤0.10 operating point를 고정**하고 locked 평가에는 같은 정책을 적용한다. 양성 매칭은 공식 window·다음 사건 제한·negative span당 FP 규약 그대로 사용한다. [공식 제출·채점 규약](https://github.com/SesameAILabs/turnbench/blob/main/docs/SUBMISSION_FORMAT.md).

우리 모델은 mono 입력을 유지하고 참조 채널은 label/평가에만 쓴다. 공식 baseline과 입력 조건 차이를 명시한다. 번호 교환에 강건한 owner-oracle 분석과 실제 온라인 외부 귀속 결과를 구분하며, 공식 제출에는 사후 정답 매핑을 시스템 출력처럼 넣지 않는다. 외부 제출은 별도 사용자 요청 시에만 한다.

### 6.2 프로젝트 내부 정량표

- gold EOT Recall, negative span FPR, hold/맞장구 내 오방출, 검출 지연 p10/p50/p90/p99.
- full turn span gold가 있는 부분만 turn F1@IoU 0.5/0.75와 분할/병합 오류. 실제 겹친 turn을 강제로 배타적인 한 사람 floor로 바꾸지 않는다.
- 확률은 emitted EOT에만 기록하지 않고, 약속한 모든 후보/negative span의 모델 점수를 저장한다. raw 점수·제약 후 점수·정책 방출을 구분한다. Brier score/reliability는 검수된 binary gold와 비교하고, weak soft target MSE와 별도다.
- **candidate recall**과 후보 내 분류 성능을 나눠 낸다. 시작/끝 후보를 놓친 사건을 calibration 분모에서만 지워 좋은 모델처럼 만들지 않는다.
- 모델 EOT, 활동 닫힘, timeout-policy를 `emission_source`로 분리한다. EOT를 맞힌 시각과 사람 ID가 확정된 시각 중 늦은 시각도 end-to-end 지연으로 보고한다.

## 7. 학습하지 않은 DB: 평가용으로 보유하는 것과 진짜 독립성은 다르다

### 7.1 평가셋을 세 등급으로 잠근다

| 등급 | 용도 | 금지 사항 |
|---|---|---|
| `debug_seen` | 과적합·파서·정합성 확인 | 일반화 성능으로 보고하지 않음 |
| `dev_unseen_session` / `dev_ood` | 모델 선택·threshold·학습 recipe 조정 | 같은 결과를 untouched test라고 부르지 않음 |
| `locked_ood_test` | 학습 DB 밖의 최종 성능 | 학습·증강 원료·증류·threshold 선택에 사용 금지 |

미학습은 Phase 2만의 조건이 아니다. **Phase 1, E2, adapter 증류, replay, 합성 원료, 이전 checkpoint 계보 전체**를 감사한다. 이미 학습한 세션을 지금 test로 옮겨도 현재 모델에서 unseen이 되지 않는다. 그 세션을 본 D1을 이어 학습한 모델도 마찬가지다.

기초 Qwen/Nemotron의 공개되지 않은 pretraining까지 “절대 미노출”이라고 보장할 수는 없다. 보고 문구는 **프로젝트 학습·개발 이력 기준 미노출, backbone pretraining 노출은 공개 정보 범위의 한계 있음**으로 한다.

### 7.2 권장 구성과 현재 상태

| 자료 | 배치 제안 | 선정 이유와 통과할 조건 |
|---|---|---|
| **DiPCo eval** | EN `locked_ood_test` 1순위 후보 | 로컬 보유 기록이 있고 D1 학습 7코퍼스에 없음. 원본 파티션·실제 사용 이력·화자 중복 감사 후 잠금. 근접 채널은 gold 보조, 주 입력은 사전 선택한 mono |
| **CHiME-6 eval** | EN 별도 강건성 locked 후보 | dev는 이미 디코더 비교에 사용됐으므로 dev에 남김. eval의 노출 감사 및 원본/CHiME-7 재편 split·정규화 구분 필요 |
| **NIA23 002_Meeting의 미노출 세션** | KO `locked_ood_test` 후보 | 현재 D1 7코퍼스 밖이지만 E2/방송 데이터 원본 중복 여부 미확인. B등급 조각의 원 시간축·겹침 라벨을 검증해야 함. audit 실패 시 새로운 KO DB를 확보 |
| NIKL 2020 | KO `dev_ood` | Phase 1의 2021–2025와 연도는 다르지만 같은 DB 계열이고 이미 점수를 보며 개발함. 엄격한 “새 DB의 untouched test”를 충족하지 않음 |
| TurnBench dev / 공식 test | EOT dev / 별도 locked test 경로 | dev는 이미 사용. 공식 test gold는 비공개이므로 승인된 제출 절차 필요. 일반 회의 ASR benchmark와 역할 분리 |
| AMI·ICSI·NOTSOFAR·71631·134 계열 | debug 또는 노출 감사 후 future dev | 현재 D1의 학습 포함 가능성이 큼. 공식 test라는 파일명만으로 독립성을 인정하지 않음 |

DiPCo는 공식적으로 개발/평가 파티션을 가진 4인 대화 자원이다. CHiME-7 버전은 CHiME-6 파티션과 정규화를 바꾸므로 원본 CHiME-6와 혼용하지 않는다. [CHiME 공식 데이터 설명](https://www.chimechallenge.org/challenges/chime7/task1/data). 보유 경로·KO 후보의 근거는 [[output-phase2-data-inventory]]와 [[task-phase2-data-prep]]이며, 이번에 서버 실물을 새로 검증하지 않았다.

**KO의 엄격한 DB-disjoint locked test는 아직 확보 확정이 아니다.** 71631의 미반입분을 새로 확보하면 session-disjoint에는 도움이 되지만 같은 DB이므로 DB-disjoint 요건의 대체로 쓰지 않는다. NIKL proxy 시각을 정렬로 바꿔도 semantic turn gold가 자동으로 생기지 않는다.

### 7.3 평가셋 생성 절차

1. 모든 프로젝트 학습 manifest의 원 대화·원 녹음·화자·파생 crop 관계를 모아 노출표를 만든다. 현재 `p2_train_hf.py`는 corpus JSONL을 읽고, 코드상 `split=train` 필터가 보이지 않으므로 파일이 이미 분리됐는지 run별로 검사한다.
2. 동일 파일 hash, 재인코딩·crop·채널 차이의 원본 ID 연결, 의심 자료의 음향 중복 검사로 교집합을 확인한다. byte hash만 다른 녹음 사본을 놓치지 않는다.
3. 대화·가능한 경우 사람/가구·회의 계열을 묶어서 분리한다. 익명 ID가 세션 안에서만 유효하면 전역 화자 분리를 검증했다고 쓰지 않는다.
4. DB별 `debug/dev/locked` manifest와 UEM, TN, 참조 버전을 고정한다. 어떤 모델도 평가셋을 합성 원료로 쓰지 못하도록 train 시작 시 교집합 검사를 넣는다.
5. 성능을 보지 않고 언어·화자 수·겹침률·근접/원거리·긴 무음으로 층화한다. 잠금 기준과 제외 사유를 먼저 정하고 어려운 창을 나중에 제거하지 않는다.

초기 규모 제안은 **EN은 확보된 DiPCo/CHiME eval 전체 세션**, KO는 독립성이 확인된 **10–20개 원 세션·5–10시간 목표**다. 부족하면 실제 확보량과 한계를 보고하고 시간을 인위적으로 채우지 않는다. 작은 경계/turn gold pilot는 평가 모델 출력을 보기 전에 KO/EN 각 200–300개 후보·negative를 선정해 검수하고, 의미가 모호한 부분은 별도 표시한다. 이 표본은 검수 비용 산정용이며 이후 final에 쓸지 dev로 쓸지 모델 결과를 보기 전에 고정한다.

## 8. 실행 조건: 좋은 부분만 뽑는 평가를 피한다

- 최종은 **원 대화 전체를 순차 스트리밍**한다. 30초 간격으로 KV·lane·외부 ID를 정답으로 초기화하지 않는다. 긴 문맥은 제품의 bounded cache/carry 규칙을 그대로 적용한다.
- 개발용 짧은 창은 고정된 manifest로 유지하고 `cold_start`·중간 발화 시작·carry를 구분한다. 이를 세션 평가와 합산하지 않는다.
- 마이크/mono 혼합·리샘플링·gain을 사전 고정한다. 참조로 화자별 분리 음원을 만들어 모델에 제공하거나 가장 잘 나온 마이크를 사후 선택하지 않는다.
- 실제 EOF와 crop 끝을 구분하고, 패딩을 미래 무음 증거로 사용하지 않는다. 시작 무음·무발화·한 사람·S≥3·N>R·실패 세션을 모두 coverage에 포함한다.
- forced alignment 실패는 해당 timing 지표의 제외 사유일 수 있지만 전사 정답이 있으면 ASR 평가에서 자동 제외하지 않는다. 누락/오염 참조는 별도 UEM으로 관리한다.
- 같은 TN·입력·세션에서 E2, 올바르게 복원한 D1, D1b를 비교한다. E2가 못 내는 화자/EOT 기능은 미지원으로 표시하고 원시 ASR 내용만 비교한다.
- 전사 최초 출력·수정 후 최종 출력, 외부 ID 최초/수정 후, RTF·tick p99·backlog·ID 확정 지연을 분리한다. 미래를 보지 않는 timestamp 규약을 fixture로 검사한다.
- checkpoint·encoder hash, reload parity, tokenizer registry, scorer 버전, decoder 옵션, TN fingerprint, manifest hash를 결과에 묶는다. 인코더 저장 사고가 재발하면 그 run은 순위표에 넣지 않는다.

## 9. 구현 산출물과 순서

기존 평가를 덮어쓰지 않고 v2 모듈/출력을 별도 두는 제안이다. 실제 경로는 구현 착수 때 확정한다.

| 단계 | 산출물 | 통과 기준 |
|---|---|---|
| E0. 입력·독립성 감사 | `eval-v2/exposure-ledger`, 모델 parity 결과, dev/locked manifest | 실제 모델 계보의 노출 표시와 제외 사유 완비 |
| E1. 텍스트 평가 | export adapter + ORC/cp evaluator + KO 문자 규약 | A/B 교환·사람 수 불일치·누락 fixture 통과 |
| E2. 화자·시간 평가 | RTTM/UEM adapter, boundary/segment scorer | 텍스트 최적 매핑과 분리, 전체 세션·겹침·unknown 보존 |
| E3. 턴 평가 | TurnBench adapter, 내부 gold pilot와 사건 로그 | weak/semantic 분리, false EOT·candidate miss·지연 노출 |
| E4. 개발 실험 | E2/D1/D1b의 동일 dev 세션 결과 | checkpoint/threshold 선택 가능, paired 비교 |
| E5. locked 평가 | 동결한 recipe의 최종 독립 DB 결과 | test로 재튜닝하지 않고 불확실성·실패 포함 보고 |

공통 출력은 기존 stream JSON에서 다음을 보존·추가한다: `session_id, lane_id, generation, episode_id, speaker_id/provisional, text/token_ids, estimated_start/end, audio_observed_until, emitted_at, assignment_revision, emission_source, raw_event_score, policy_event_score`.

없는 값은 null/미지원으로 둔다. 정답 generation·화자·경계로 채워 넣지 않는다. `text`, `diarization`, `boundary`, `turn`, `system` 결과 파일을 따로 만들어 지표 한 축을 바꾸어도 원 추론을 다시 실행할 필요가 없도록 한다. decoder를 바꾸면 추론 결과 자체가 바뀌므로 새 run으로 남긴다.

## 10. 평가기가 먼저 통과해야 할 fixture

1. **세션 전체 A↔B 교환:** ORC·cp·DER·owner-aware 경계 점수 불변.
2. **첫 발화만 누락 + 이후 전체 번호 교환:** 실제 누락은 남고 번호 교환 추가 페널티는 없음.
3. **중간부터 전체 발화의 사람 라벨만 변경:** 내용 지표는 안정적이고 cp/DER는 오류를 잡음.
4. **한 발화를 여러 lane에 분할:** ORC의 분할 민감도와 DI 진단 차이를 확인. 무조건 모두 0을 요구하지 않음.
5. **두 사람을 한 ID로 병합 / 한 사람을 여러 ID로 분절:** 사람 오류를 평가에서 숨기지 않음.
6. **참조·가설 사람 수 불일치:** 빈 스트림·미대응 사람·삽입·삭제를 모두 포함.
7. **lane generation 재사용:** 외부 ID가 올바르면 과거/신규 사람의 전사를 섞지 않음.
8. **동일 단어가 겹친 두 화자:** 중복 참조를 하나로 합치거나 출력 하나를 두 TP로 쓰지 않음.
9. **음향 경계만 ±80/160/400ms 이동:** 내용 지표는 그대로, 시간 지표만 변함.
10. **동일 episode에서 이벤트 중복:** boundary FP와 공식 EOT negative-span 규칙이 각각 정의대로 작동.
11. **무음에 ONSET·단어 환각:** 텍스트 없는 참조를 0점/skip하지 않음.
12. **EOT 없음·timeout만 있음·잘못된 사람 EOT:** EOT miss·정책 출처·귀속 오류를 구별.
13. **문장만 여러 JSON 행으로 나눔:** 전사 정렬 단위를 임의 변경해 지표가 좋아지지 않는지 검사. 허용된 segmentation 영향은 명시.
14. **KO 띄어쓰기·Unicode 조합·특수 토큰:** 동결 CER 규약과 일치.
15. **crop/원본이 서로 다른 split:** 데이터 누출 검사가 차단. 이미 노출된 checkpoint 계보도 표시.
16. **같은 오디오 prefix·다른 미래 suffix:** 과거에 가용했던 출력·시각이 바뀌지 않음.

## 11. 보고와 의사결정

주 표는 **코퍼스별 내용 ORC / 종합 cp / DER 분해 / 경계 F1 / EOT recall·FPR·latency**로 고정한다. 부가 표에는 N, S, 겹침률, 발화 길이, 무음 길이, 입력 조건, 유효/제외 시간을 낸다.

모델 비교의 신뢰구간은 창이 아니라 **원 세션 단위 paired bootstrap**으로 계산한다. dev/test 수가 작으면 폭을 그대로 보고하고, “0.x% 개선”만으로 우열을 확정하지 않는다. test에서 가장 좋은 metric·threshold·마이크를 골라 대표값으로 삼지 않는다.

best checkpoint는 dev의 내용 성능을 기본으로 보되 화자·EOT 오방출·지연의 guardrail을 함께 만족해야 한다. 복합 점수 하나로 상쇄시키지 않고 trade-off를 남긴다. 모든 임계·허용 회귀폭은 E4에서 고정한 후 E5에 적용한다.

**가장 먼저 할 일은 E0와 E1이다.** 번호 교환에 강건한 전사 점수와 실제 미노출 세션을 확보하지 않으면, 새 학습이 인식을 개선한 것인지 라벨 순서만 바뀐 것인지 판단할 수 없다. 그 뒤 E2/E3를 붙여 경계와 턴을 객관적으로 평가한다. 이 순서가 정본 모델 구조를 성급히 바꾸지 않고 원인을 분리하는 방법이다.

## 외부 근거 — 확인일 2026-09-17

- [MeetEval 공식 구현](https://github.com/fgnt/meeteval) — ORC/cp/timing 계열 및 입출력 형식.
- [MeetEval 논문](https://www.isca-archive.org/chime_2023/neumann23_chime.html) — 회의 전사 평가의 표준 지표 묶음.
- [WER 정의 비교 논문](https://arxiv.org/html/2508.02112v1) — ORC·DI-cp의 다른 민감도와 진단 한계.
- [dscore](https://github.com/nryant/dscore) — DER 분해·JER.
- [TurnBench 공식 규약](https://github.com/SesameAILabs/turnbench/blob/main/docs/SUBMISSION_FORMAT.md) — 인과적 timestamp·EOT/INT 평가.
- [CHiME 데이터 설명](https://www.chimechallenge.org/challenges/chime7/task1/data) — DiPCo와 CHiME 버전별 평가 조건.
