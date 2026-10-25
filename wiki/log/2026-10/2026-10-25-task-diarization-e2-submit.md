## [2026-10-25] task | 화자 분리 통합 E2 제출(low_p_hpc)

- Changed:
  - `slurm/submit-q17-e2-tgt.sh`: 기본 파티션을 apex 에서 low_p_hpc 로 바꿨다(학습·평가 노드 모두). MAX_RESTART 기본은 20(apex 면 5)이고, `--partition apex` 로 되돌릴 수 있다.
  - [[status]]: E1 진행, E2 제출 job.
- Reason:
  - 사용자 "low_p_hpc 로 제출할 수 있도록 해줘".
  - 선점 대비는 이미 있다. 학습은 선점 신호에 저장하고 requeue 뒤 마지막 체크포인트에서 잇는다(`slurm/stage3-train-apex.sbatch`). 평가 노드는 `--requeue` 이고, amitgt 는 끝난 몫을 완료 표시로 건너뛴다.
  - mxc dry-run 에서 학습·평가 sbatch 가 모두 `--partition=low_p_hpc` 인 것을 확인했다.
  - 사용자가 제출했다: 학습 95486(2 노드, 4,000 step), 평가 95487(95486 성공 뒤, amitgt 8 몫).
  - E1(95482)은 16:12 에 step 4,480/20,000, 0.44 s/step 이다.
- Next:
  - 95486 시작·진행을 본다. 선점이 없으면 학습 약 1.5 h, 평가 약 2 h(어림).
  - amitgt 결과를 E0(SDM eval cp 39.3)·창 평가(53.5)와 비교한다.
  - E1 이 끝나면 AMI DER 을 잰다.
- By: tskim
