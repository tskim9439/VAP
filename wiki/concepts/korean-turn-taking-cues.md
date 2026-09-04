---
type: concept
status: seed
created: 2026-09-03
updated: 2026-09-03
summary: 한국어 어미와 clause-final prosody가 turn 완결성 신호로 작동하는 방식 — 한국어 벤치마크의 언어학적 근거
sources:
  - [[source-chatgpt-research-plan]]
---

# 한국어 turn-taking 단서

## 왜 한국어가 흥미로운가

한국어는 **head-final** 이고 술어가 문장 끝에 온다. 따라서 turn 완결성 정보가
**발화 마지막 수백 ms 에 집중** 된다. 영어는 어순상 단서가 더 일찍 분산된다.

이 구조적 차이는 두 가지 예측을 낳는다 — 둘 다 검증 가능한 가설이다.

1. **acoustic-only 모델의 한국어 성능이 영어보다 상대적으로 나쁘다.**
   (단서가 늦게 오므로 미리 투사하기 어렵다)
2. 따라서 **[[acoustic-linguistic-fusion]] 의 이득이 한국어에서 더 크다.**

이것이 사실이면 **이 연구의 가장 좋은 발견 중 하나**가 된다.
bilingual 실험을 "성능 표 하나 더" 가 아니라 **가설 검증**으로 만들 수 있다.

## 관찰 대상 어미

```text
-요      비격식 종결 → EOT 강신호
-습니다  격식 종결   → EOT 강신호
-는데    연결/전환   → HOLD 신호이나 실제로는 turn 양보로도 쓰임 (모호)
-고      나열 연결   → HOLD
-면      조건 연결   → HOLD
-서      이유/순차   → HOLD
```

`-는데` 는 특히 흥미롭다. 형태상 연결어미지만 실제 대화에서는 말끝을 흐리며
floor 를 넘기는 데 자주 쓰인다. **형태만으로는 결정되지 않고 prosody 가 필요한
사례** 이므로 fusion 가설의 좋은 테스트 케이스다.

## 상태

현재는 **가설이며 근거 자료가 없다.** 한국어 대화 분석 문헌으로 뒷받침해야 한다.
→ [[question-korean-turn-cue-literature]]

한국어 benchmark 설계 시 이 어미들을 **층화 변수** 로 두면 분석이 풍부해진다.
→ [[task-korean-benchmark-design]]
