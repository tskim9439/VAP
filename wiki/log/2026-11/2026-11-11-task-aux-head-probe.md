## [2026-11-11] task | 보조 헤드 진단 구현, P1·P2 학습량 정정

- Changed: `experiments/ssa_head_probe.py`·`tests/test_ssa_head_probe.py`·`slurm/ssa-head-probe.sbatch`·`slurm/submit-ssa-head.sh` 추가. [[output-cpwer-hypotheses-20261110]] 7 절(진단 설계·읽는 법), [[output-ssa-papers-review-20261106]] 7 절 정정.
- Reason: 사용자가 보조 헤드 진단을 요청했다(인코더 한 번 + LLM 한 스트림 안에서, 학습 때만 쓰는 헤드는 허용). 구현 중 두 가지를 확인했다. ① `ssa_probe.py` 는 DDP forward 를 거치지 않아 rank 사이 기울기가 합쳐지지 않았다 — P1·P2 는 GPU 1 장 분량(P1 약 600 h, P2 약 3,000 h)이다. ② 우리 모델(E13)의 인코더는 원 Nemotron 3.5 와 모든 텐서가 다르다 — 진단은 우리 인코더를 얹어 잰다.
- Next: 사용자가 `slurm/submit-ssa-head.sh` 로 제출하면 `ssa_probe_report.sh h1` 로 보고한다. 결과에 따라 인코더 보조 손실 학습(다음 단계)을 정한다.
- By: tskim
