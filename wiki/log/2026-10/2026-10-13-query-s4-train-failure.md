## [2026-10-13] query | q06-s4-ast 학습(job 85484)이 step 5,640 에서 멈춤

- Changed: 없음(기록만). 평가 job 85615 는 `--after` 로 학습 뒤에 걸려 있었고, final 이 없어 23:49 에 바로 끝났다.
- Reason: 사용자 진행 확인 요청에서 발견했다.
- 결과:
  - **경과:** hpc 에서 로딩 중 선점 3 번(19:46·19:56·20:22)을 맞았다. 20:26 네 번째 시작(hpc-83·92)부터 학습했다.
    - 0.55~0.63 s/step 이 평소 속도다.
    - 21:35~22:45 에는 2~4 s/step 으로 느려졌다(원인 미확인).
  - **실패 (23:45):**
    - rank 0 의 DataLoader worker 가 SIGKILL 로 죽었다(`DataLoader worker ... is killed by signal: Killed`). 나머지 rank 는 gloo 연결 끊김으로 따라 죽었다.
    - 로그에 OOM 문구는 없다. SIGKILL 이 노드 OOM killer 에서 왔을 가능성이 크지만, sacct(MaxRSS) 로 확인하지 못했다.
  - **재시도 없음:** sbatch 는 `SLURM_RESTART_COUNT < MAX_RESTART(3)` 일 때만 실패를 requeue 하는데, 선점 3 번이 카운트를 다 써서 "restart 한도 초과" 로 끝났다.
  - **남은 것:**
    - checkpoint-1000~5000 이 완전하다(5000 은 23:15 저장, trainer_state 포함). 잃은 학습은 5000 이후 640 step 이다.
    - 같은 `--run` 으로 다시 내면 checkpoint-5000 에서 이어 간다(지문 대조).
- Next:
  - 사용자: 원인 확인 `sacct -j 85484 --format=JobID,State,ExitCode,MaxRSS,NodeList%40 -P`
  - 재제출: apex(비선점) 또는 hpc + 서브셸 `MAX_RESTART` 상향. 평가는 새 job id 에 `--after` 로 다시 건다.
- By: tskim
