## [2026-10-23] task | 평가 SLURM 묶음과 AMI IHM·SDM 평가

- Changed: `slurm/submit-evals.sh`(새 파일 — 묶음 r1x-ami·q17-s1·q17-v3), `raw/sources/experiments/2026-10-21-v3-attribution/v3_eval_task.sh`(`ami` 작업), `vapasr/hf/stage.py`(`stage_dir` 잠금·프로세스별 임시 파일), `tests/test_stage_dir.py`, `slurm/submit-q17-chain.sh`(시간 어림), `raw/sources/experiments/2026-10-22-q17-chain/README.md`, [[status]]
- Reason:
  - 사용자 "진행 상황 확인해줘" → "평가 작업도 SLURM 통해 수행할 수 있도록 명령어를 알려줘" → "AMI_IHM 이랑 AMI_SDM 에 대해서도 평가를 수행하고 싶어".
  - 1.7B 체인 평가를 학습 job 뒤에 자동으로 붙이고(afterok), 0.6B R1x 와 1.7B 를 AMI 공식 분할로도 비교한다.
  - AMI 는 V2 때 만든 `ami-bench-v1`·`ami_report.py` 와 같은 명령이다. 단일 GPU 로 4.5–5.7 h 걸려 4 몫으로 나눴다.
  - 1.7B Stage 1 실측 속도(2.86 s/step, 같은 step 구간 0.6B 의 2.3 배)로 체인 어림을 37–46 h 로 고쳤다.
- 확인:
  - 로컬·mxc 테스트 통과(`test_stage_dir`·`test_resume_checkpoint`).
  - 세 묶음 dry-run 이 나오고 참조 경로가 모두 있다.
  - AMI 명령 스모크(R1x final, 창 2 개)에서 디코드와 보고서가 나왔다.
- Next:
  - 사용자 제출(r1x-ami 바로, q17-s1·q17-v3 는 의존).
  - r1x-ami 결과를 V1 AMI 와 비교한다.
  - Stage 1 평가 뒤 q06-s1m3-d10·q17-s1m-r3 와 `cmp_eval.py` 로 비교한다.
- By: tskim
