## [2026-10-20] task | st4-m2a·m2b 학습 완료 — 전화 대역 증강으로 TAXI cpWER 40.9 → 28.1, VibeVoice 스트리밍 비교 페이지

- Changed:
  - raw [[2026-10-19-stage4-dialogue-translation]]: "st4-m2a·m2b 결과 — 중간" 절, `st4-m2a/`·`st4-m2b/`(TAXI·합성·대화 번역 요약, parity), `stage4_eval_resume.sh`, `vibevoice_cmp_m2.sh`
  - [[output-vibevoice-stream-comparison-20261020]](새), [[output-stage4-prep-plan-20261019]], [[status]]
  - `experiments/baseline_vibevoice.py`: score `--dest`(기존 score.json 을 덮지 않게)
- Reason: 사용자 "오늘 오전에 전화 증강해서 학습시킨거 학습이 완료된거 같은데?", "결과를 VibeVoice Streaming 과 비교할 수 있도록 작성해줘."
- 결과(δ10):
  - TAXI 받아쓰기 cpWER: st4-m1 ck1500 40.9 → m2a ck1500 29.4 · ck2000 28.1 · final 28.3, m2b 29.7–31.6. VibeVoice 스트리밍 19.2.
  - TAXI 독→영 빈 턴 26 → 2–3, chrF 독→영 42 → 51. 같은 언어 대화 번역은 ±1 BLEU.
  - CallHome 독일어를 더한 m2b 가 TAXI 에서 m2a 보다 1.6–2.6 pt 나쁘다(영어도 나쁨, 원인 모름).
  - final 의 대화 번역·합성·Stage 3 회귀는 호스트 GPU 공유 중 CUDA OOM 으로 중간에 끊겼다 → 큐가 끝나면 같은 폴더에 이어 쓴다.
- Next: Stage 3 회귀·단일 화자·CallHome 평가와 final 재평가 → 같은 창 VibeVoice 대비 대화 cpWER(`vibevoice_cmp_m2.sh`) → 체크포인트 선택(사용자)
- By: tskim
