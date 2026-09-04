---
type: question
status: seed
created: 2026-09-03
updated: 2026-09-03
summary: VAD·타이밍 규칙으로 유도한 EOT/HOLD/INT/BACKCHANNEL 라벨이 사람 라벨과 얼마나 일치하는가
sources:
  - [[turn-taking-objectives]]
  - [[source-turnbench]]
---

# 유도 이벤트 라벨은 믿을 만한가?

## 질문

VAD + 타이밍 휴리스틱으로 자동 생성한 EOT / HOLD / INTERRUPTION / BACKCHANNEL
라벨이 사람 어노테이션과 얼마나 일치하는가?

## 왜 중요한가

[[turn-taking-objectives]] 의 `L_event` 는 이 라벨에 의존한다. 그런데
[[source-conversation-corpora]] 의 AI Hub·CANDOR 에는 이벤트 라벨이 **없다.**
휴리스틱 라벨의 정확도를 모르면:

- auxiliary head 의 성능 향상이 진짜인지 라벨 노이즈 적합인지 알 수 없다.
- 특히 **backchannel 과 interruption 의 구분**이 어렵다. 둘 다 상대 발화 중
  겹쳐 시작하며, 차이는 "이후 floor 를 가져가는가" 라는 사후 정보다.

## 검증 방법 — 정답지가 이미 있다

[[source-turnbench]] 의 **otoSpeech(104h)는 사람이 라벨했다.**
휴리스틱을 otoSpeech 에 적용해 사람 라벨과 비교하면 규칙별 precision/recall 을
직접 잴 수 있다.

측정할 것: 이벤트 종류별 P/R/F1, 혼동행렬(특히 BACKCHANNEL↔INTERRUPTION),
타이밍 오차 분포.

**수용 기준(잠정)**: 이벤트별 F1 ≥ 0.8. 미달하는 이벤트는 auxiliary supervision 에서
제외하거나 라벨 노이즈를 논문에 명시한다.

## 한국어로의 전이

otoSpeech 에서 튜닝한 규칙이 한국어에 그대로 통하리라는 보장이 없다
(backchannel 빈도와 형태가 언어마다 다르다 — "네", "아 예", "음"). 한국어
어노테이션 서브셋으로 다시 검증한다. → [[task-korean-benchmark-design]]

## 상태

미해결. → [[task-event-label-heuristics-validation]]
