---
type: source
status: active
created: 2026-09-06
updated: 2026-09-06
summary: Stage 1 mono 6,000-step sentinel에서 EN WER은 22–29%까지 하락했지만 KO 과소 방출과 큰 RNN-T 격차가 남은 중간 결과
raw_path: raw/sources/experiments/2026-09-06-stage1-mono-pilot-6000-sentinel-partial.md
observed: 2026-09-06
raw_authors:
  - tskim
---

# Stage 1 mono 6,000-step sentinel 중간 결과

## 무엇인가

LibriSpeech 100 h와 KsponSpeech 약 100 h로 random-init adapter·Qwen LoRA를
6,000 optimizer step 학습한 뒤, 작은 고정 sentinel에서 측정한 중간 결과다.
마지막 dev-other·kspon-dev 평가는 아직 진행 중이므로 최종 판정 자료가 아니다.

## 관측

- EN: @5000에서 dev-clean 22.1%, dev-other 28.9%까지 내려왔다. tok/chunk는
  참조와 비슷하고 p50 약 200 ms, `viol80` 0.5%라 방출 중심은 정상이다.
- KO: bias 0 CER은 85.1%에서 65.9%로 개선됐지만 @3000 이후 60%대에서 흔들렸다.
  @5000 tok/chunk 0.140은 참조 0.273의 51%이고 최적 bias는 주로 1이다.
- 대조군 RNN-T `[56,0]`은 더 큰 seed-7 표본에서 dev-clean 4.4%, dev-other 8.2%,
  kspon-dev 20.2%다. 표본이 다르므로 정확한 상대비는 큰 dev 선택 평가에서 계산한다.
- text loss는 2.05→0.96, train text top-1은 0.50→0.75로 끝까지 개선됐다.

## 왜 절대 WER이 높은가

현재 학습량은 한 epoch보다 훨씬 작다. 학습기는 EN·KO를 optimizer step마다 교대하므로
6,000 total step은 언어별 3,000 update다.

| 언어 | update × batch | 본 표본 | 전체 표본 | 대략적 epoch |
|---|---:|---:|---:|---:|
| EN | 3,000 × 2 | 6,000 stream | 13,182 | 0.46 |
| KO | 3,000 × 8 | 24,000 utterance | 62,000 | 0.39 |

또한 cosine scheduler는 `steps=6000`을 전체 주기로 써서 @6000에서 learning rate가
0이 된다. @5000에는 peak LR의 약 6.7%만 남는다. 손실·top-1이 계속 개선되는 가운데
LR이 먼저 끝났으므로, 현재 수치는 구조의 수렴 한계보다 **의도적으로 짧은 파일럿의
과소학습**을 더 강하게 반영한다.

동시에 격차 자체는 무시할 수 없다. 같은 Nemotron encoder의 RNN-T가 훨씬 낮은
WER/CER를 내므로 음향 정보가 사라진 것이 아니라, random-init adapter와 제한된 Qwen
LoRA가 그 표현을 interleaved text로 옮기는 학습이 아직 부족하거나 목적 함수·자유실행
경로가 비효율적인 것이다.

## KO 해석 수정

“짧은 KO 스트림이라 NEXT 비율이 더 높다”는 가설은 현재 aggregate와 맞지 않는다.
참조 text density는 KO 0.273 tok/chunk로 EN 약 0.226보다 높다. 즉 NEXT 하나당
참조 텍스트 수는 KO가 더 많다. 짧은 스트림의 앞뒤 무음 효과는 따로 측정할 수 있지만,
현재 과소 방출을 NEXT 표본 비율만으로 설명할 수 없다.

KO `next_weight` sweep은 합리적인 후보지만 5–11%p 정도의 bias 개선만으로 RNN-T와의
전체 격차를 닫지는 못한다. 먼저 같은 체크포인트에서 다음을 분해한다.

1. dev 교사강제 text top-1/top-5
2. 자유실행 substitution/deletion/insertion
3. text 위치에서 `<NEXT_AUDIO>`를 top-1으로 고른 비율
4. bias별 방출률·CER·evidence 위반 변화

교사강제 정확도도 낮으면 학습량·adapter mapping이 우선이고, 교사강제는 높지만
deletion이 많으면 방출 calibration·노출 편향이 우선이다.

## 다음 판단

- 진행 중인 @6000 sentinel 완료값을 보존한다.
- ckpt 4000/5000/6000을 동일한 큰 dev 표본에서 비교해 sentinel 표본 잡음을 제거한다.
- 선택은 현재 파일럿 안의 best일 뿐 수렴 checkpoint라는 뜻은 아니다.
- Stage 2는 고정 step이 아니라 언어별 본 audio hours 또는 effective epoch 기준으로
  스케줄을 잡고, LR이 최소 한 pass 전에 0이 되지 않게 한다.
- 1,900 h로 바로 확장하기 전에 현재 200 h에서 1 epoch 부근인 total 13k–16k까지의
  짧은 연장 곡선은 optimization 부족과 구조 한계를 싸게 구분하는 대조가 된다.

## 출처

- 원본: `raw/sources/experiments/2026-09-06-stage1-mono-pilot-6000-sentinel-partial.md`
- 대조군: `raw/sources/experiments/2026-09-06-s1-baselines/baselines-all.json`
- 학습 코드: `experiments/s1_train_mono.py`

