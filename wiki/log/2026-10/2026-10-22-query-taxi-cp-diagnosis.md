## [2026-10-22] query | TAXI cp 를 낮출 방안 — 남은 귀속 오류 진단과 1.7B 레시피 결정

- Changed: `experiments/sim_build_dialogues.py`(--level-offset-db 진단 옵션), `tests/test_tel_sim.py`, `raw/sources/experiments/2026-10-21-v3-attribution/`(README 19–21 절, v3_level_dev_build.sh, build_tel_v2.sh), [[status]]
- Reason: 사용자 "Taxi 에서 cp 를 낮출 수 있는 방안을 더 모색해봐", "진단 결과에 따라 1.7 B 의 레시피를 조정하고 제출 명령어를 알려줘".
  - R1x 남은 TAXI 오류는 놓친 영어 온셋(15/279)이다. 목소리·레벨이 비슷한 세션에만 있다.
  - 레벨(−12 dB)·채널(같은 채널 ±2 dB) 진단 dev 에서 R1x 는 놓침이 늘지 않는다. 그래서 합성 증강은 근거가 없다.
  - 같은 언어 교대 가중은 TAXI 귀속에 도움이 되지만 같은 언어 가드를 깬다. 1.7B 6 단계는 R1x 그대로다.
- Next: 1.7B 체인 제출(사용자) → 6 단계 뒤 평가·평균. 남은 실패(실제 비슷한 목소리)를 TAXI 없이 재현할 셋을 찾는다(예: 같은 성별 실제 2 회선 통화).
- By: tskim
