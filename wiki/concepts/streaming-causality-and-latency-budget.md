---
type: concept
status: active
created: 2026-09-03
updated: 2026-09-03
summary: 80ms frame ASR encoder로 VAP를 옮길 때의 causality 감사와 latency 회계 — 이 연구의 최대 방법론적 위험
sources:
  - [[source-qwen3-asr]]
  - [[source-nemotron-3-5-asr-streaming]]
  - [[source-turnbench]]
---

# Streaming causality 와 latency 예산

**초안이 다루지 않았지만 이 연구의 성패를 가르는 항목이다.**
리뷰어가 가장 먼저 공격할 지점이기도 하다.

## 문제 1 — causality leak

ASR encoder 는 보통 미래를 조금 본다.

- **FastConformer**: convolution subsampling 과 self-attention 에 right-context 가 있다.
  cache-aware streaming 설정이 이를 정의하지만, 설정값을 확인하지 않으면 모른다.
- **Qwen3 AuT**: 기술 보고서가 **causal 여부와 lookahead 를 명시하지 않는다.**
  "dynamic attention window 1–8s" 는 window *크기* 이지 right-context 가 0 이라는
  뜻이 아니다. offline 모드로 특징을 뽑으면 **미래 오디오가 `h_t` 에 새어 들어가
  VAP 결과 전체가 무의미해진다.**

### 감사 방법 (싸고 확실함)

```text
전체 오디오로 뽑은 h_t   vs   시점 t 에서 자른 오디오로 뽑은 h_t
                  둘이 비트 단위로 같아야 한다
```

`t` 를 뒤로 밀며 처음으로 일치하는 지점이 **실효 lookahead** 다.
모든 encoder 에 대해 이 값을 표로 기록한다. → [[task-audit-encoder-causality-lookahead]]

### 감사 결과 (2026-09-03)

→ [[output-encoder-causality-audit]]. Nemotron 80 ms chunk 는 ≤80 ms 로 통과.
**Qwen AuT 는 기본 경로가 비인과(구현 버그)이고 블록 모드도 평균 420 ms** 로 관문 초과.
CPC 대조군 0 ms 로 방법 검증.

## 문제 2 — latency 회계

보고하는 latency 는 반드시 세 항의 합이어야 한다.

```
latency = frame_period/2  +  effective_lookahead  +  compute_time
```

encoder lookahead 를 빼고 "우리 모델이 VAP 368 ms 보다 빠르다" 고 쓰면 비교가
성립하지 않는다. **encoder 별 lookahead 를 명시한 latency 표를 논문에 싣는다.**

## 문제 3 — frame rate 하락

| 모델 | frame rate | frame period |
|------|-----------:|-------------:|
| 원 VAP (CPC) | 50 Hz | **20 ms** |
| Nemotron FastConformer | 12.5 Hz | **80 ms** |
| Qwen3 AuT | 12.5 Hz | **80 ms** |

**backbone 을 바꾸면 시간 해상도가 4배 거칠어진다.** VAP 의 첫 구간 [0, 0.2s] 는
50 Hz 에서 10 frame 이지만 12.5 Hz 에서는 **2.5 frame** 에 불과하다.

### 예산 계산

목표는 [[source-turnbench]] 의 사람 기준 **−151 ms** 이고 VAP 는 368 ms 다.
80 ms frame 이면 양자화 오차만 ±40 ms 이므로 원리적으로는 여유가 있다.
그러나 lookahead 가 320 ms 라면 **구조적으로 VAP 를 이길 수 없다.**

따라서 **[[task-audit-encoder-causality-lookahead]] 는 다른 모든 모델링 작업의
선행 조건이다.** lookahead 가 크면 다음 중 하나로 대응한다:

1. chunk 를 80 ms 로 줄여 lookahead 를 최소화 (품질 손실 감수)
2. encoder 상위 layer 만 causal 로 재학습
3. 저해상도 semantic path + **고해상도 acoustic path(50 Hz VAD/CPC)** 를 병합해
   시간 해상도를 회복

3번은 그 자체로 기여가 될 수 있다.

## 문제 4 — encoder 를 채널마다 두 번 돌리는 비용

초안의 architecture 는 화자 A·B 각각에 shared encoder 를 통과시킨다. CPC(수 M
파라미터)에서는 값싸지만 **180M AuT 나 24층 FastConformer 를 80 ms 마다 2회**
돌리면 real-time factor 예산이 무너질 수 있다.

Stage 1 에서 **RTF 와 GPU memory 를 반드시 함께 측정**하고, 초과하면:
mixture 단일 encoder + speaker-conditioned readout, 또는 채널 concat 입력을 검토한다.
