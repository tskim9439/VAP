---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-10-15
created: 2026-09-15
updated: 2026-09-15
summary: Phase 2 EOT 를 C-mode(3 s 관측 뒤)가 아니라 발화 구간 끝에서 즉시 낸다(2026-09-15). SEG_END 토큰은 두지 않고 EOT 가 그 자리를 맡으며, 라벨은 구간 종료 뒤 결과(교대/침묵/재개)로 만든 soft target 이다
sources:
  - '[[output-phase2-dynamic-speaker-memory-plan-v2]]'
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[output-phase2-lane-proposal-report]]'
---

# 결정: EOT 는 발화 구간 끝에서 즉시, soft label 로 학습

## 맥락

정본 [[output-phase2-streaming-asr-diarization-plan]] §6.3 은 Q1 의 `<EOT>` 를 C-mode 로 두었다. 종료 판정에 필요한 관측(offset 뒤 3 s 동안 본인 재개 없음)을 마친 뒤에 토큰을 내므로 라벨은 확실하지만 방출이 최소 3 s 늦다. lane 유지형 제안 [[output-phase2-dynamic-speaker-memory-plan-v2]] 는 그 위에 구간 종료 토큰 `<SEG_END>` 를 따로 두어 lane 을 닫았다. 사용자는 2026-09-15 "EOT 가 3 초 뒤에 붙는 것은 너무 길다. SEG_END 를 EOT 로 치환하고, 노이즈가 있더라도 soft label 로 학습해 lane 이 끊어지면 바로 EOT 로 인식하게 하자"고 결정했다.

## 검토한 선택지

| 선택지 | 장점 | 단점 |
|---|---|---|
| A. C-mode 유지(정본 Q1) | 라벨 확실, 조기 EOT 오류 없음 | 최소 3 s 지연. 대화 에이전트 용도에 부적합 |
| B. 구간 끝 hard EOT(모든 구간 끝 = EOT) | 즉시, 구현 단순 | 회의 자료에서 구간 끝 뒤 3 s 안 본인 재개가 42–62 % 라 절반이 오라벨 |
| **C. 구간 끝 EOT + soft target** | 즉시, 오라벨을 확률로 흡수, p(EOT) 가 그대로 신뢰도 | 부분 관측에서 예측하므로 precision 은 운율·의미 단서가 허용하는 만큼만 |

## 결정

`<EOT>` 는 각 발화 구간의 마지막 lexical 청크(offset+δ, 약 320 ms)에서 후보로 놓고, 그 위치의 학습 target 을 구간 종료 뒤 결과에 따른 확률 p_end 로 둔다. `<SEG_END>` 는 등록하지 않으며, lane 은 EOT 방출 또는 활동 헤드의 비활성으로 닫힌다(lazy-free 유지). C-mode 는 ablation 으로 내려간다.

## 근거

- 요구 지연이 3 s 보다 훨씬 짧다.
- 구간 끝이 턴 끝인지는 관측 시점에 불확실하지만, 그 불확실성은 soft target 으로 학습해 p(EOT) 를 신뢰도로 쓰면 된다. 임계값은 런타임에서 정한다.
- 즉시 방출이면 lane 재배정 뒤 늦은 EOT 의 귀속 문제(pointer head 필요성)가 아예 사라진다.

## 정본에 미치는 영향

| 정본 절 | 현재 | 바뀌는 것 |
|---|---|---|
| §6.3 | Q1 = C-mode, P 는 ablation | Q1 = 구간 끝 soft P-mode, C 는 ablation |
| §4.4 | EOT 는 C 증거 가용 이후 | EOT 후보는 구간 마지막 lexical 직후 |
| §6.1 손실 | EOT hard CE(가중 2) | 두 점 soft target CE(가중 2) |
| §8 평가 | C offset 지연 | 후보 위치 AUC·calibration·hold 내 오방출률, 지연은 고정 320 ms + 연산 |

상세 규칙·수치는 [[output-phase2-dynamic-speaker-memory-plan-v2]] §11.

## 재검토

D1 결과에서 hold 구간 내 오방출률이 제품 임계를 넘으면 후보 위치를 늦추는 변형(offset+0.5–1.0 s)을 같은 soft 규칙으로 비교한다. 2026-10-15.
