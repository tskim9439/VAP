## [2026-10-18] task | Stage 3 파일럿 p11(격리 발화 되살리기) 제출 경과 — low_p_hpc 선점·랑데부 실패 뒤 멈춤, sbatch 감시자와 --exclude 추가

- Changed:
  - `slurm/stage3-train-apex.sbatch`: 실패 뒤 srun 이 멈추면 300 s 유예 뒤 requeue(한도 넘으면 취소)하는 감시자
  - `slurm/submit-stage3-train.sh`: `--exclude` (기본 제외 hpc-52·hpc-4 에 더함)
  - raw [[2026-10-18-stage3-ko-revive]]: 제출 경과 절, 평가·비교 스크립트(`pick_gpu_queue.sh`, `cmp_p11`·`cmp_attr_p11`·`boot_p11`·`mask_cmp_p11`)
- Reason: 사용자가 `low_p_hpc` 로 낸 job 91256 이 학습을 시작하지 못했다. 사용자: "low_p_hpc 만 사용할 수 있어 현재로썬"
- 결과:
  - 17:45 로딩 중 선점 → requeue
  - 18:13 hpc-60·102 재시작에서 `RendezvousTimeoutError` → hpc-102 작업이 끝나지 않아 sbatch 가 `wait` 에 멈춤(RUNNING 1 시간+, `squeue`·`sacct` 로 확인)
  - 감시자는 가짜 scontrol 로 5 경우(멈춤·정상·유예 안 종료·이전 시도 줄·한도 초과)를 mxc 에서 시험했다.
  - 제출 `--dry-run` 에서 `--exclude=…hpc-52,…hpc-4,…hpc-102` 가 넘어가는 것을 봤다.
  - 평가 큐 3 개(ck2000·ck3000·final)는 체크포인트를 기다린다.
  - 다시 낸 job 91438:
    - 시도 1: 환경 복사본에서 pyexpat 이 빠져 실패했다 → [[2026-10-18-task-stage3-env-staging-check]]
    - 시도 2(hpc-17·18): 19:41 학습 시작. 9 셋 점검 불일치 0, 데이터·섞기 비중은 dry-run·p10 과 같다.
- Next: 학습이 끝나면 평가 큐(ck2000·ck3000·final)가 돈다. 그 뒤 p10 과 같은 step 에서 비교한다(`mask_cmp_p11.py` 등).
- By: tskim
