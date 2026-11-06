## [2026-11-06] query | 1.7B Stage 2(q17-s4-ast) 번역 — 0.6B 보다 모든 방향에서 BLEU 가 높다(δ10 +1.7 ~ +3.4, 지연 거의 같음)

- Changed: [[output-q06-s4-ast-eval-20261013]] "1.7B Stage 2 와 견줌" 절
- Reason: 사용자 "stage 2 모델의 번역 성능도 궁금해." → "번역 결과 나왔어?" — job 100590 의 FLEURS·CoVoST2 디코드가 끝나 번역만 먼저 채점(받아쓰기 셋은 아직 도는 중)
- 결과(δ10 BLEU, LAAL s): FLEURS 영→독 25.99(1.98) · 영→한 31.42(3.35) · 독→영 30.85(2.03) · 한→영 18.31(2.71), CoVoST 영→독 27.21(1.77) · 독→영 33.41(1.78). 0.6B 24.29 · 29.70 · 28.80 · 14.90 · 25.23 · 30.91.
- Next: 받아쓰기 셋(Kspon·YODAS·LibriSpeech)·영→한 peak05 끝나면 post-eval-q17-bench.sh, 필요하면 xCOMET·짝 bootstrap
- By: tskim
