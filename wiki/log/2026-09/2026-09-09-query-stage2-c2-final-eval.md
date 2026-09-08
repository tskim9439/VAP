## [2026-09-09] query | run C2 최종 모델 WER/CER·타이밍 평가 보고서

- Changed: `wiki/outputs/output-stage2-c2-final-eval.md`, `raw/sources/experiments/2026-09-08-hf-C2-final-eval/`
- Reason: 사용자 요청 — 1,930 h full FT 최종 모델의 정확도와 방출 타이밍 평가. select 표본으로 checkpoint 선정(final), δ·next_bias 스윕으로 지연–정확도 트레이드오프, RNN-T 대조군·파일럿과 비교. δ=4 에서 dev-clean 0.056 / dev-other 0.122 / kspon-dev 0.156(대조군 0.044 / 0.082 / 0.202).
- Next: 전체 dev 수치(§4, slurm/s3_eval.sbatch), 디코더 tick 최적화, D run(66523) 결과와 비교.
- By: tskim
