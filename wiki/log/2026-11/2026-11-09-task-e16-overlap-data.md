## [2026-11-09] task | E16 — 겹침 데이터만 늘린 결과

- Changed: [[output-agn-reduction-20261104]] 12 절(신규)·summary. 코드·데이터: `SimConfig.p_full`·`--p-full-overlap`(완전 겹침), `sim_v7_trial.sh`·`sim_v7_build.sh`(→ `VAPKT-data/data/sim-dialogue-v7-ovl2`, sim_en-ovl2 2 만 세션 505 h), `slurm/submit-q17-e16.sh`, `auto_report.sh`
- Reason: 사용자 "화자 겹침 데이터를 더 만들어서 학습되도록 하는 게 낫지 않을까?" → "겹침 길이를 더 늘려도 될거 같아. 완전 겹치는 케이스도 추가하고" → "응 시작해."
- 발견
  - E16(E13·인코더 고정·겹침 비중 22 %)은 표준 cp 24.92 / 31.70(E13 24.39 / 31.33)이다.
  - 둘 겹침 놓침은 43.2 / 49.6 %(E13 42.7 / 48.9 %)로 줄지 않았다. 번역 BLEU 는 40.6(E13 41.3)이다.
- Next: 화자마다 조건 지은 소리를 LLM 주 입력으로 주는 구조(먼저 LLM 없이 인코더 단독 검증)
- By: tskim
