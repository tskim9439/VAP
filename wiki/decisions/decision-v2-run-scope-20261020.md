---
type: decision
status: active
created: 2026-10-20
updated: 2026-10-20
summary: V2 첫 학습 범위 — 두 스트림 입력(V2-a)·최대 4 화자·창 20–90 s 매 epoch 재추출·회의 번역 타깃 생성을 한 run 으로 묶는다
decision_status: accepted
owner: tskim
review: 2026-10-27
sources:
  - '[[output-v2-speaker-memory-plan-20261020]]'
  - '[[output-vibevoice-stream-comparison-20261020]]'
  - '[[decision-stage4-prep-20261019]]'
  - '[[decision-cst-training-stages-and-speaker-slots]]'
---

# V2 첫 학습 범위 (2026-10-20)

## 맥락
- V1 = Stage 4 최종 st4-m1 ck1500. 전화 대역 증강(st4-m2a)으로 TAXI cpWER 이 40.9 → 28.1 이 됐다. VibeVoice 스트리밍은 19.2 다([[output-vibevoice-stream-comparison-20261020]]).
- 화자 귀속 몫(cp − agn)은 VibeVoice 가 모든 코퍼스에서 0.8–5.2 pt 작다. M0: 우리 LLM 입력에는 화자 정보가 69 % 만 남는다([[output-v2-speaker-memory-plan-20261020]]).
- 학습 창은 20–40 s 로 고정이었고, 같은 창을 epoch 마다 그대로 다시 썼다(2,500 step 동안 영어 실제 대화 4.8 회, 독일어 3.3 회). 학습 화자 수는 2 명까지다([[decision-cst-training-stages-and-speaker-slots]]).

## 결정 (사용자, 2026-10-20)
1. **V2-a 두 스트림 입력을 쓴다.** 인코더 L00 → `adapter_tap`(0 초기화) → LLM 입력에 더한다. "그래 학습 전 점검까지 준비해."
2. **창을 epoch 마다 다시 뽑고(A) 최대 길이를 늘린다(B).** 길이는 20–90 s 다. "A 와 B 를 묶자."
3. **최대 화자 수를 4 명으로 늘린다.** "최대 화자 수를 4명으로 늘리고 싶어."
4. **3–4 인 창도 번역한다.** 영어 회의(AMI·ICSI·NOTSOFAR)의 번역 타깃을 S1 방식으로 새로 만든다.
5. **위 변경을 한 run 으로 묶는다.** V2-a 단독 파일럿은 하지 않는다. 각 변경의 효과를 따로 가르기 어렵다는 점은 받아들였다.

## 결과
- 슬롯 C·D 토큰은 Phase 2 예약 토큰 `<SPK_3>`·`<SPK_4>`(151717·151718)를 쓴다. 새 어휘는 없다. 예약 행은 평균 + 잡음으로 다시 초기화하고 decode 차단을 푼다.
- 실제 3–4 인 대화는 영어 회의뿐이다. 한국어·독일어 3–4 인은 합성 대화로 만든다(합성 생성기를 N 화자로 넓힌다).
- 진행과 산출은 [[output-v2-run-plan-20261020]] 에 적는다.
