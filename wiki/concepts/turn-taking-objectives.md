---
type: concept
status: active
created: 2026-09-03
updated: 2026-09-03
summary: VAP 256-class + time-to-next-turn(생존분석) + semantic event의 다중 목표 설계와 손실 균형 문제
sources:
  - [[source-turn-taking-related-work-2026]]
  - [[source-chatgpt-research-plan]]
---

# Turn-taking objective 설계

## 전체 손실

```
L = λ_ASR·L_ASR + λ_VAP·L_VAP + λ_τ·L_next-turn + λ_event·L_event + λ_VAD·L_VAD
```

### 1. L_VAP — 미래 활동 투사

[[voice-activity-projection]] 의 256-class CE. 기반 목표로 유지한다.

### 2. L_next-turn — time-to-next-turn

```
τ_t = t_next_speaker_onset − t
```

를 discrete bucket 으로 분류: `<160 / 160–320 / 320–640 / 640–1280 / 1280–2560 / >2560 ms`.
근거는 [[source-turn-taking-related-work-2026]] 의 Next-Turn (320 ms 이내 endpoint
accuracy **+25.9%p**, binary EPD 와 **보완 관계**).

#### 초안보다 강화할 점 — 생존분석으로 정식화

초안의 τ 정의는 세 경우에 모호하다:

1. **어느 화자의 onset 인가?** (상대 화자만? 아무나?)
2. **overlap 중에는?**
3. **같은 화자가 다시 말하면?** (mid-turn pause 는 next turn 이 아니다)

또한 horizon 안에 onset 이 없는 구간은 `>2560` 이 아니라 **right-censored** 다.
따라서 τ head 를 **discrete-time hazard model** 로 정의할 것을 권한다:

```
h_k(t) = P(onset 이 구간 k 에서 발생 | 구간 k 이전에는 없었음)
```

- censoring 을 원리적으로 처리한다 (관측 종료 구간은 likelihood 에서 제외).
- `P(EOT before Δ)` 같은 임의 시점 확률을 hazard 로부터 바로 계산할 수 있어
  [[turn-taking-evaluation-protocol]] 의 "−600/−300/0 ms 시점 확률" 지표와 직결된다.
- bucket 분류보다 구현이 거의 안 늘고 정보 손실이 적다.

**대상 정의는 문서에 명시적으로 고정한다**: τ = "현재 floor 를 쥐지 않은 화자의
다음 onset 까지의 시간, overlap 시작 시점 기준, 동일 화자 재개는 제외."

### 3. L_event — semantic event auxiliary head

`P(EOT) / P(HOLD) / P(INTERRUPT) / P(BACKCHANNEL)`.

**라벨 출처가 문제다.** AI Hub·CANDOR 에는 이 라벨이 없다
([[source-conversation-corpora]]). VAD + 타이밍 규칙으로 **유도**해야 한다.
예: "짧고, 상대 발화 중에 겹치며, 이후 floor 를 가져가지 않는 발화 = backchannel".

이 규칙은 반드시 **사람이 라벨한 [[source-turnbench]] otoSpeech 에 대해 정확도를
측정** 한 뒤 사용한다. 검증 없이 쓰면 라벨 노이즈가 결론을 오염시킨다.
→ [[task-event-label-heuristics-validation]]

## 손실 균형 — 초안이 비워둔 부분

다섯 손실의 스케일이 크게 다르다 (RNNT 손실 ~수십, 256-class CE ~5.5 nats,
BCE ~0.7). 고정 λ 를 손으로 맞추면 **ASR 이 지배** 하거나 VAP 가 무시된다.

권장:

1. **1단계**: encoder·ASR 을 freeze 하고 turn head 만 학습 (λ_ASR = 0). 가장 안정적.
2. **2단계**: joint 시 **uncertainty weighting**(Kendall) 또는 GradNorm 으로 자동 균형.
3. **가드레일**: unfreeze 후 **WER/CER 회귀를 반드시 보고** 한다.
   VAP 가 좋아져도 ASR 이 망가지면 통합 모델의 존재 이유가 사라진다.
   → [[task-stage2-multitask-vap-with-wer-guardrail]]

## class imbalance

EOT·INT 는 희귀 이벤트다. 256-class CE 만 낮추면 다수 클래스에 수렴한다.
balanced sampling 또는 focal loss 를 검토하고, 평가는 반드시 **이벤트 단위**로 한다.
