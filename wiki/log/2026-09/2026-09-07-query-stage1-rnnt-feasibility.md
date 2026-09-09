## [2026-09-07] query | Stage 1 RNN-T 동급 가능성 평가

- Changed: `wiki/sources/source-stage1-mono-run-ab.md`, `wiki/sources/source-stage1-mono-pilot-6000-sentinel-partial.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/status.md`
- Reason: Stage 1 개선 run A/B 결과를 근거로 IS-SLM이 RNN-T 동급 정확도와 예측 가능한 방출 타이밍을 동시에 달성할 가능성을 엄격히 평가하고, 단순 epoch 연장보다 데이터·정렬 목표·런타임 개선을 우선하는 관문을 기록했다.
- Next: B 기반 200/500/1,000/1,900 h scaling curve, full-dev 평가, windowed alignment·latency loss·KO next_weight Pareto sweep, end-to-end tick 측정.
- By: tskim
