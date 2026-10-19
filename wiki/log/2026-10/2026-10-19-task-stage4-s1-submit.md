## [2026-10-19] task | Stage 4 S1 대화 번역 전체 생성 제출 — 첫 연쇄는 hpc-50·51 munge 인증 오류로 실패, 제외하고 다시 제출(91697–91700)

- Changed:
  - `slurm/submit-ast-mono-generate.sh` — `--exclude N,…`(sbatch 기본 제외 hpc-52·hpc-4 에 더함, Stage 3 제출 스크립트와 같은 방식)
  - [[output-stage4-prep-plan-20261019]] S1 상태
- Reason: 사용자 "S1 대화 번역 타깃 전체 생성 제출 명령어를 전역변수 설정되지 않도록 해서 작성해줘." — 서브셸 안에서만 job 번호를 받아 `--after` 로 이음
- 결과:
  - 91684(apex, hpc-50·51)는 env 복사 `srun` 에서 `Munge encode failed: … munge.socket.2: Permission denied` 로 바로 끝났다. 생성 결과는 없다. 같은 시각 스케줄러 ssh 도 간헐적으로 연결이 거부됐다.
  - hpc-50·51 을 빼고 다시 제출: 일반 생성 91697(hpc-101·102, 09:30 시작) → 수리 91698 → 되살림 생성 91699 → 수리 91700
- Next: 생성·수리가 끝나면 두 루트 타깃 묶기(`ast_build_targets --pack`), 교차 언어 합성 셋·평가 셋. hpc-50·51 munge 문제는 관리자에게 알릴 것(사용자)
- By: tskim
