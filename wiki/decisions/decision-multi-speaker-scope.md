---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-10-01
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 화자 수 제한 "최대 2 명" 을 제거한다(2026-09-11). 회의·다자 코퍼스를 자연 데이터로 쓰고, 화자 슬롯·활동 헤드·평가를 N 화자로 일반화한다
sources:
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[output-phase2-data-inventory]]'
---

# 결정: Phase 2 화자 수 제한 제거

## 맥락
정본 계획 [[output-phase2-streaming-asr-diarization-plan]] §1 은 범위를 0–2 화자로 두고 A/B 슬롯, 활동 헤드 2 sigmoid, VAP 256-class(2 화자 × 4 구간)를 전제했다. 자연 2 화자 대화가 약 290 h 로 부족해 회의 코퍼스(AMI·ICSI·CHiME-6·NOTSOFAR-1·DiPCo)를 후보에 넣으면서, 사용자가 2026-09-11 "최대 2 화자" 제약을 제거하기로 했다.

## 결정
Phase 2 모델은 화자 수를 2 로 제한하지 않는다. 다화자 회의·대화 코퍼스를 창 선택 없이 자연 데이터로 쓴다.

## 정본 계획에 미치는 영향 (반영 필요)
| 정본 절 | 현재 | 바뀌어야 하는 것 |
|---|---|---|
| §1 범위 | 0–2 화자, A/B 익명 ID | 0–K 화자(K 는 슬롯 상한, 초기 4–8 검토), 도착 순서 익명 ID `<SPK_1..K>` |
| §3 활동 헤드 | 2 sigmoid | K sigmoid(슬롯별), 새 화자 등장 시 다음 빈 슬롯 배정(Sortformer 식 도착 순서) |
| §3 VAP | 256-class(2 화자 × 4 bin) | 화자 슬롯별 4-bin 활동을 독립 예측(K × 4 sigmoid) 또는 "나 / 나머지" 2 채널 근사. 원 VAP 256-class 와의 비교는 dyadic 셋에서만 |
| §4.1 정체성 | 최초 식별 화자 = A, 다음 = B | 도착 순서로 슬롯 1..K, 슬롯 수 초과 화자는 오류로 계수. crop 안 재매핑도 도착 순서 |
| §4.2 직렬화 | 두 화자 교차 | K 화자 교차(규칙 동일), 태그 수 증가 → 청크 상한 재측정 |
| §5 데이터 | dyadic + 합성 2 화자 | + 회의 코퍼스(원거리 mono, 헤드셋 채널 라벨), 합성도 2–4 화자 |
| §6 hazard | 화자별 다음 onset(2) | 슬롯별(K) |
| §8 평가 | cpWER, DER(2), TurnBench(dyadic) | cpWER·DER 는 그대로 K 화자에 적용(meeteval), 화자 수 추정 오류 추가, TurnBench 는 dyadic 부분 평가로 유지 |
| §7 관문 | 비중첩 DER ≤10 % 등 | dyadic 셋과 회의 셋을 분리해 관문 설정 |

## 결과
- [[output-phase2-data-inventory]] §2b 의 회의 코퍼스는 전량 사용 대상이 된다. 한국어 다화자(NIA23 002_Meeting, AI Hub 464)도 후보.
- 정본 계획의 위 절은 사용자 확인 후 개정한다. 이 페이지가 개정 전까지의 근거다.

## 재검토
2026-10-01 — Q1 결과에서 K 화자 슬롯의 귀속 오류가 dyadic 대비 크게 나쁘면 K 를 줄이거나 dyadic 우선으로 되돌린다.
