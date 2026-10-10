## [2026-10-11] task | q06-s2-ast(job 83866): 불완전 checkpoint-1000, step 1 부터 다시 학습

- Changed:
  - 코드·설정 변경 없음. 학습 상태만 기록한다.
- Reason:
  - 선점 뒤 checkpoint-1000 에서 이어 간다고 보고했는데 사실과 달랐다. 바로잡는다.
- 결과:
  - 첫 checkpoint-1000(2026-10-10 21:26–21:28 저장)은 저장 도중 선점됐다.
    - 남은 파일: `model.safetensors`, `optimizer.pt`
    - 빠진 파일: `trainer_state.json`, `scheduler.pt`, `rng_state_0.pth`
    - 그래서 `last_complete_checkpoint` 가 이를 건너뛰었다. restart=7(23:18, hpc-89·90)은 step 1 부터 다시 학습했다(run 기록 `resume_from` = None).
  - 같은 디렉토리에 새 checkpoint-1000 이 저장됐다(2026-10-11 00:06–00:08, trainer_state.json 포함 완전).
    - [[2026-10-10-ast-eval-smoke-ck1000]] 평가 시험은 첫 실행의 step-1000 가중치로 했다. 그 가중치는 이제 덮어써졌다.
  - 저장에 약 2 분이 걸린다(optimizer.pt 9.7 GB). 선점 신호가 저장 중에 오면 체크포인트가 불완전하게 남을 수 있다.
  - 00:08 기준 step 1,000 / 14,288, 0.65–0.94 s/step, 남은 시간 약 3.5 시간(선점이 없을 때).
- Next:
  - 학습이 끝날 때까지 지켜본다. 끝나면 final 로 전체 평가(번역·받아쓰기·LibriSpeech)를 한다.
- By: tskim
