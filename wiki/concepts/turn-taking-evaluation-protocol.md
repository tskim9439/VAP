---
type: concept
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: TurnBench 규약을 채택한 평가 프로토콜 — 매칭 윈도우, recall/FPR/latency 정의, 보고할 지표 축
sources:
  - [[source-turnbench]]
---

# 평가 프로토콜

## 채택 규약 — TurnBench 그대로

새 규약을 만들지 않는다. [[source-turnbench]] 와 숫자를 직접 비교하려면 동일해야 한다.

- 매칭 윈도우: gold 시점 `t` 기준 **[t−0.25 s, t+3.0 s]**
- 윈도우 내 **가장 이른 미청구 예측**이 TP
- negative span 내 발화는 몇 번이든 **FP 최대 1회**
- `Recall = TP/(TP+FN)`, `FPR = FP/(FP+TN)`, latency **p10 / p50 / p90**

## 보고 축

| 능력 | 지표 |
|------|------|
| EN ASR | WER |
| KO ASR | CER |
| Streaming ASR | TTFT, Time-to-Final, **revision rate** |
| EOT | Recall @ FPR ≤ 0.10 / 0.15 |
| Interruption | Recall @ FPR ≤ 0.10 / 0.15 |
| Timing | latency p10 / p50 / p90 |
| Prediction | **P(EOT \| x_≤t) at −600 / −300 / 0 ms** |
| Backchannel | Precision / Recall / F1 |
| Efficiency | **RTF, GPU memory, chunk latency, encoder lookahead** |

## 초안에 추가한 것

1. **latency-recall 곡선.** 단일 operating point 비교는 임계값 선택에 취약하다.
   임계값을 쓸며 (recall, FPR, p50 latency) 궤적을 그린다. 이것이 "빠르면서 정확한가"
   를 보여주는 유일한 정직한 방법이다.
2. **encoder lookahead 를 latency 표에 명시.**
   → [[streaming-causality-and-latency-budget]] 없이는 비교가 무효다.
3. **ASR 회귀 가드레일.** turn head 를 붙인 뒤의 WER/CER 을 항상 함께 싣는다.
4. **기준선 3종을 표에 고정 배치**: 사람 **−151 ms**, VAP(test) **368 ms / 0.845 @ 0.055** · VAP(dev, 재현) **463 ms / 0.841 @ 0.045**,
   DualTurn. split 을 반드시 명시한다. → [[output-vap-turnbench-baseline-reproduction]] → [[task-add-missing-baselines]]

## 목표 지점

```text
사람     −151 ms  ┤●
                  │
목표             ┤    ◆  (recall ≥ 0.845, FPR ≤ 0.055, p50 < 250 ms)
                  │
VAP      368 ms  ┤            ●
```

**최소 성공 기준**: 동일 FPR 에서 VAP 와 같은 recall 을 유지하며 p50 latency 를
유의하게 낮춘다. recall 을 올리면서 latency 를 못 낮추면 기여가 약하다.
