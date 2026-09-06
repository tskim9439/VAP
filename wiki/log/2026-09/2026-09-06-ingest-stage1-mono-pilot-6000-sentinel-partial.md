## [2026-09-06] ingest | Stage 1 mono 6,000-step sentinel 중간 결과

- Changed: `raw/sources/experiments/2026-09-06-stage1-mono-pilot-6000-sentinel-partial.md`, `wiki/sources/source-stage1-mono-pilot-6000-sentinel-partial.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/status.md`
- Reason: 파일럿의 EN WER 하락과 정상 방출 타이밍, KO 과소 방출, RNN-T 대비 큰 절대 격차를 기록하고, 6,000 mixed step이 언어별 0.39–0.46 epoch에 불과하며 LR이 먼저 0이 되는 과소학습 조건임을 분리해 해석했다.
- Next: @6000 나머지 sentinel 완료, 같은 큰 dev에서 ckpt·bias 선택, 교사강제 정확도와 자유실행 S/D/I 진단, 필요 시 KO next-weight sweep, Stage 2 전 13k–16k one-epoch 연장 대조.
- By: tskim
