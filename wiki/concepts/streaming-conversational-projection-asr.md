---
type: concept
status: seed
created: 2026-09-03
updated: 2026-09-04
summary: 하나의 streaming representation에서 transcription과 미래 대화 역학을 동시 예측하는 제안 모델의 정의와 가설
sources:
  - [[source-chatgpt-research-plan]]
---

# Streaming Conversational Projection ASR (제안 모델)

구체적인 구현 구조 제안:

- [[output-dual-rate-conversational-projection-architecture]] — RNN-T state를 사용하는 dual-rate 구조
- [[output-interleaved-streaming-slm-architecture]] — RNN-T 없이 audio/text를 교차 처리하는 통합 SLM 구조

> A bilingual streaming speech model for simultaneous transcription and
> predictive turn-taking.

## 정의

매 시점 `t` 의 streaming audio `x_≤t` 에서 **동시에** 예측한다:

```
P(Y_text | x_≤t)                    ← streaming transcription
P(A_{t:t+H} | x_≤t, Y_≤t)           ← 미래 H(=2s) 구간 대화 역학
```

두 번째 항은 단순 endpoint 가 아니라 **speaker activity / floor transition /
interruption / backchannel** 이다.

```text
                     Streaming Audio
                           │
                 Streamable Speech Encoder
                           │
                   shared representation
                  ┌────────┴─────────┐
                  ▼                  ▼
            Streaming ASR     Turn Projection
                  │                  ├─ Future VAP
           partial transcript        ├─ Time-to-next-turn
                  └──── semantic ────┤─ EOT / Hold
                        state        ├─ Interruption
                                     └─ Backchannel
```

구체 구조 제안 → [[output-model-architecture-proposal]] (이중 프레임율, 2026-09-04).

## 핵심 가설 (각각이 하나의 ablation)

1. **H1.** ASR 로 대규모 pretrain 된 streaming representation 은 CPC 기반보다
   turn-taking 에 유리하다. → [[question-asr-representation-vs-ssl-for-vap]]
2. **H2.** incremental linguistic state 를 acoustic VAP 와 결합하면 mid-turn pause 와
   true EOT 를 더 잘 구분한다. → [[acoustic-linguistic-fusion]]
3. **H3.** binary EOT 보다 future activity + time-to-next-turn 을 joint 예측하면
   더 빠르면서 FP 가 낮다. → [[turn-taking-objectives]]

## 조건부 항 `Y_≤t` 의 함정

수식이 `Y_≤t` 로 조건화하지만, 학습 시에는 gold transcript(teacher forcing)를,
추론 시에는 **모델 자신의 가설**(오류·revision 포함)을 쓰게 된다. 전형적인
exposure bias 다. 대응:

- gold transcript 조건 vs 자체 가설 조건의 **성능 격차를 반드시 ablation 으로 보고**한다.
  이 격차가 크면 H2 의 실용적 가치가 무너진다.
- 완화책: scheduled sampling, 또는 text token 이 아니라 decoder hidden state 만
  공유(→ [[acoustic-linguistic-fusion]]).

## Muse 와의 차이

| | Muse | 제안 모델 |
|---|------|-----------|
| 출력 | LISTEN / TEXT / SPEECH_ONSET / SPEECH_ENDPOINT / SPEAKER | 좌측 + **HOLD / YIELD / INTERRUPT / BACKCHANNEL / TIME_TO_NEXT_TURN / FUTURE_VOICE_ACTIVITY** |
| 질문 | "무엇을 듣고 있으며 언제 끝났는가" | **"무엇을 말하고 있으며 앞으로 누가 언제 말할 것인가"** |

이 차이는 Full-Duplex Speech Agent 에서 실질적이다. 다만
[[source-muse-voice-transcribe]] 는 **closed weights** 이므로 Muse 는 재현 대상이
아니라 아이디어 출처이자 API black-box 비교 대상이다.
