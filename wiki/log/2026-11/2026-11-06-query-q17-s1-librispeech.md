## [2026-11-06] query | 1.7B Stage 1 LibriSpeech test-clean·other — δ10 2.34 / 5.66, final 1.83 / 4.45(0.6B 2.56 / 6.21 · 2.15 / 5.29, 같은 채점)

- Changed: [[output-stage1-q06-d10-eval-20261009]] LibriSpeech 절에 1.7B 표
- Reason: 사용자 "1.7 B 모델의 Librispeech test clean 과 others 에 대해 ASR 평가를 수행할 수 있는 명령어를 알려줘." → job 100589(`slurm/submit-eval-q17-bench.sh s1` — 다른 세션이 만든 스크립트를 mxc 에 올림, md5 대조)
- 결과(WER %, aa-wer-approx-v1): δ4 3.02 / 7.35 · δ6 2.54 / 6.28 · δ8 2.37 / 5.89 · δ10 2.34 / 5.66 · final 1.83 / 4.45. 0.6B 보다 모든 칸에서 낮다.
- Next: 1.7B Stage 2 번역(job 100590) → post-eval-q17-bench.sh
- By: tskim
