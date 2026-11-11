## [2026-11-11] task | E18 준비 — 인코더와 LLM 을 함께 학습(겹침 셋 75 %)

- Changed: `slurm/submit-q17-e18.sh` 추가, `slurm/submit-evals.sh` 에 묶음 `q17-e18*`, 스모크 `raw/sources/experiments/2026-11-11-e18/e18_smoke.sh`. [[output-cpwer-hypotheses-20261110]] 3 절에 준비 내용.
- Reason: 사용자가 "N3D 고정 + 인코더·LLM 함께 학습 + 겹침 늘림" 조합이 없었음을 확인하고 준비를 요청했다(low_p_hpc 로 제출 예정). E17 과 인코더 고정 여부만 다르게 했다. 스모크: 인코더 전체(609.1M) 학습으로 2 step·parity OK, GPU 메모리는 제출 값에서 약 72 GB 어림.
- Next: 사용자가 제출하면 checkpoint-5000 을 E17 의 같은 step 과 견준다(둘 겹침 놓침·혼자 말 놓침). low_p_hpc 는 재시작마다 약 39–45 분이 들어 느릴 수 있다 — 데이터 목록 캐시가 후보다.
- By: tskim
