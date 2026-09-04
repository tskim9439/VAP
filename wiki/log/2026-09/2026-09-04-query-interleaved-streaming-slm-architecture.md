## [2026-09-04] query | Interleaved Streaming SLM 구조 계획과 비판적 평가

- Changed: `wiki/outputs/output-interleaved-streaming-slm-architecture.md`, `wiki/concepts/streaming-conversational-projection-asr.md`
- Reason: 독립 RNN-T 없이 streamable speech encoder의 soft token과 LLM text state를 반복 결합해 실시간 전사와 미래 대화 역학을 함께 예측하는 통합 SLM 구조를 설계하고, summation fusion·emission policy·causality·실시간 deadline의 실패 조건을 평가했다.
- Next: interleaving-only / raw sum / gated contextual residual 세 조건의 최소 ASR 실험과 80 ms p99 deadline 측정.
- By: tskim
