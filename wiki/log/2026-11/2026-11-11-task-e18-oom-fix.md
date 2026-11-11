## [2026-11-11] task | E18 메모리 부족과 기울기 누적

- Changed: `experiments/stage3_train.py --grad-accum`, `vapasr/hf/trainer.py`(로그에 gpu_gb), `slurm/submit-q17-e18.sh`(배치 토큰 4,096 × 누적 4·저장 250 step·평가를 2 노드씩 차례로·`--evals-only`), `slurm/submit-evals.sh --after-any`, 스모크 `raw/sources/experiments/2026-11-11-e18/e18_smoke.sh`. [[output-cpwer-hypotheses-20261110]] 3 절.
- Reason: E18 첫 제출(103140)이 step 70 쯤 GPU 메모리 부족으로 죽고 같은 자리에서 되풀이됐다. 인코더 전체를 풀면 무거운 배치가 139 GB 를 넘는다. 배치를 4 분의 1 로 줄이고 4 번 모아 갱신하게 했다(둘째 스모크 60 step: GPU 최대 49.7 GB). 사용자가 평가를 한 번에 2 노드까지만 쓰도록 요청했다.
- Next: 사용자가 103140·103171–103174 를 취소하고 다시 제출하면 checkpoint-5000 을 E17 의 같은 step 과 견준다. 재시작 때 데이터 순서가 처음부터 다시 도는 점은 따로 보완 후보로 남긴다.
- By: tskim
