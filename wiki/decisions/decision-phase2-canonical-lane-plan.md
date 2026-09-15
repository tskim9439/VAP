---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-10-15
created: 2026-09-15
updated: 2026-09-15
summary: Phase 2 정본을 lane 유지형 계획 [[output-phase2-lane-plan]] 으로 교체한다(2026-09-15). lazy-free R=6, EOT 즉시 soft label, 세션 정체성은 외부 재매핑, 모델 내 화자 메모리 범위 밖
sources:
  - '[[output-phase2-lane-plan]]'
  - '[[output-phase2-dynamic-speaker-memory-plan-v2]]'
  - '[[output-phase2-lane-proposal-report]]'
  - '[[decision-eot-immediate-soft-label]]'
---

# 결정: Phase 2 정본을 lane 유지형 계획으로 교체

## 맥락
이전 정본 [[output-phase2-streaming-asr-diarization-plan]](2026-09-14 개정)은 고정 K 슬롯·C-mode EOT 였다. 다른 세션이 동적 화자 메모리 대안([[output-phase2-dynamic-speaker-memory-plan]])을 제안했고, 이를 검토·실측해 lazy-free lane·R=6·병렬 학습으로 고친 실행안([[output-phase2-dynamic-speaker-memory-plan-v2]])이 나왔다. 이어 사용자가 "재등장 화자를 같은 lane 에 붙일 필요는 없고 외부 화자 인식기로 재매핑한다"는 조건과 "EOT 는 구간 끝 즉시, soft label"을 결정했다([[decision-eot-immediate-soft-label]]). 그림 보고서([[output-phase2-lane-proposal-report]])로 설명한 뒤 사용자가 2026-09-15 "이번 계획은 마음에 들어. 이걸 정본으로 설정해줘"라고 결정했다.

## 검토한 선택지
| 선택지 | 장점 | 단점 |
|---|---|---|
| 이전 정본 유지(고정 K, C-mode) | 검증된 규약, 라벨 확실 | 인원 상한 = K, EOT 3 s 지연 |
| 동적 화자 메모리(원안) | 인원 상한 제거 | 순차 학습·pointer sidecar·미검증 표현, 보유 DB 로 이득 구간이 좁음 |
| 즉시 해제 lane + 외부 재매핑(다른 세션 [[output-phase2-external-speaker-mapping-plan]]) | 단순 | 즉시 해제는 lane run 을 끊어 외부 재매핑 증거를 잃음(구간 단독 증거 부족 49–66 %) |
| **lane 유지형(lazy-free R=6) + 즉시 soft EOT + 외부 재매핑** | N≤6 에서 이전 정본과 동일 시퀀스, 새 구조 토큰 없음, 병렬 학습, EOT 320 ms | 겹침 화자 표현·soft EOT 정밀도는 D2/D1 에서 실측 필요 |

## 결정
[[output-phase2-lane-plan]] 을 Phase 2 정본으로 한다. 이전 정본은 `superseded` 로 두되, 새 정본이 참조로 지정한 절(직렬화 순서·블록 무음/flush·스키마·파이프라인·80 ms 정합·평가 원칙·구현 지도·QC·예산)은 계속 유효하다. 다른 세션의 외부 재매핑 두 페이지와 원안·비교 페이지는 분석 기록으로 남기며 규약을 정의하지 않는다.

## 근거
- lazy-free 는 N≤R 에서 이전 정본과 완전히 같은 시퀀스를 만들어 비교·이행 비용이 없다.
- R=6 은 AMI/ICSI/NOTSOFAR-1 실측(재배정 ≤2.4 %)으로 정했다.
- lane 유지가 외부 재매핑의 전제다(구간 단독 증거 부족 49–66 % → lane run 1–11 %).
- EOT 즉시·soft 는 사용자 요구 지연(≪3 s)과 오라벨 비율(본인 재개 42–66 %)을 동시에 다룬다.

## 재검토
D2 에서 겹침 화자 표현이 무너지거나, D3 외부 재매핑이 N≥7 에서 부족하면 모델 내 화자 메모리를 재검토한다. 2026-10-15.
