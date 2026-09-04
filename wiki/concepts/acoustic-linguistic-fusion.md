---
type: concept
status: active
created: 2026-09-03
updated: 2026-09-03
summary: ASR 내부 decoder state를 turn predictor와 공유해 cascade 없이 semantic 정보를 얻는 설계
sources:
  - [[source-turn-taking-related-work-2026]]
  - [[source-chatgpt-research-plan]]
---

# Acoustic-Linguistic Fusion

## 왜 필요한가

음향 정보만으로는 구분되지 않는 멈춤이 있다.

> "그런데 제가 말씀드리고 싶은 것은…"  ← 뒤의 침묵 = **HOLD**
> "네, 알겠습니다."                    ← 뒤의 침묵 = **EOT**

두 침묵은 음향적으로 비슷하지만 언어적으로 정반대다. 특히 한국어는 어미가
turn 완결성을 강하게 신호한다 → [[korean-turn-taking-cues]]

## 설계 원칙 — cascade 금지

```text
❌  Audio → ASR → text → TurnGPT      (텍스트를 다시 다른 모델에 넣음)
                                       latency 가 누적된다

✅              ┌─ ASR head
    audio → h_t ┤                      내부 state 만 공유
                └─ Turn prediction
```

```
z_t = F(h_t^audio, h_t^ling)
```

| backbone | h_audio | h_ling |
|----------|---------|--------|
| [[source-nemotron-3-5-asr-streaming]] | FastConformer encoder state | **RNNT predictor state** |
| [[source-qwen3-asr]] | AuT hidden state | **incremental Qwen decoder hidden state** |

Nemotron 쪽이 predictor state 가 명시적이라 실험이 단순하다. 먼저 여기서 검증한다.

## 선행 연구와의 차이

[[source-turn-taking-related-work-2026]] 의 **JAL-Turn** 이 동일한 acoustic-linguistic
cross-attention 을 이미 제안했고 ASR 과 병렬 실행해 추가 latency 가 없다.
따라서 fusion 자체는 novelty 가 아니다. **차별점은 그것을 hold/shift 이진 판정이
아니라 [[voice-activity-projection]] 의 future projection 으로 확장** 하는 것이다.
논문에서 이 구분을 흐리면 안 된다.

## 반드시 측정할 것 — cascade 를 실제로 재보기

초안은 cascade 를 "latency 가 커진다" 며 측정 없이 기각했다. 리뷰어는 숫자를 요구한다.

- cascade baseline (ASR → 텍스트 turn 모델) 을 **실제로 만들어 latency 와 정확도를 측정**한다.
- 그래야 "joint 가 cascade 대비 얼마를 벌었는가" 를 말할 수 있다.
→ [[task-add-missing-baselines]]

## ASR 오류 민감도

linguistic state 는 ASR 이 틀리면 함께 오염된다. 반드시 두 조건을 비교한다:

- **gold transcript 조건** (상한)
- **모델 자체 가설 조건** (실제)

격차가 크면 H2 의 실용적 가치가 제한된다.
한국어 CER 이 나쁜 chunk 크기(80–160 ms)에서 특히 위험하다.
