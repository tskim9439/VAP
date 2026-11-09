## [2026-11-09] task | P1 화자 조건 인코더 단독 검증

- Changed: [[output-ssa-papers-review-20261106]] 7 절(신규)·summary. 코드 `experiments/ssa_probe.py`·`tests/test_ssa_probe.py`·`slurm/ssa-probe.sbatch`·`slurm/submit-ssa-probe.sh`·`raw/sources/experiments/2026-11-09-ssa-probe/ssa_probe_report.sh`
- Reason: 사용자 "먼저 LLM 없이 인코더만으로 화자를 가를 수 있는지 검증해보자"
- 발견
  - Nemotron 3.5 + Parakeet 식 화자 커널(0 층)을 6,000 step 대상 화자 학습했다.
  - AMI eval 조각 평가에서 둘 겹침 놓침이 42.7 / 51.5 % 로, E13 과 같고 Parakeet(같은 방식 22.8 / 33.0 %)의 두 배다.
  - 늘어난 몫은 삭제다. 귀속(0.36)·삽입(1.03)은 Parakeet 보다 적다.
  - 같은 조각 방식의 Parakeet 은 스트리밍 값과 0.3 / 2.3 pt 안이라, 평가 방식은 믿을 만하다.
- Next: 학습 분포(합성)에서도 겹친 말을 못 꺼내는지 진단하고, 겹침 비중·학습량을 올린 판을 검토한다.
- By: tskim
