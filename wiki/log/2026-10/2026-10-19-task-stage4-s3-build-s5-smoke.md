## [2026-10-19] task | Stage 4 S3 본 생성·S5 스모크 — 교차 언어 합성 sim_en-de·sim_en-ko(1 만 세션씩)·영↔독 평가 300 세션, 전체 dry-run 문제 0, 스모크 PARITY OK, 평가 큐

- Changed:
  - `experiments/stage3_train.py` `--sim-data` 쉼표 목록, `slurm/submit-stage3-train.sh` Stage 4 옵션
  - raw [[2026-10-19-stage4-dialogue-translation]]: S3 본 생성·전체 dry-run·학습 준비 절, `xl_build.sh`·`xl_eval_build.sh`·`stage4_dry_train_full.sh`·`stage4_smoke_train.sh`·`stage4_eval_queue.sh`, 집계 JSON
  - [[output-stage4-prep-plan-20261019]] S3·S5·S6 상태
- Reason: S1 타깃 완료 뒤 계획대로(사용자 "그리고 Stage 4 준비를 이어서 수행해.")
- 결과:
  - `sim_en-de` 252.7 h · `sim_en-ko` 249.2 h, 발화 모두 번역 있음. 영↔독 held-out 평가 300 세션 7.7 h(한↔영 평가 셋은 사용자 결정 대기로 만들지 않음)
  - 전체 타깃 dry-run: 실제 대화 7 코퍼스·합성 셋 모두 문제 0, 번역 모드 길이 Stage 3 의 1.12–1.21 배, 학습 split 행 번역 비율 한 94.7 %·영 99.8 %·독 99.1 %
  - 학습 dry-run 11 셋 불일치 0, 스모크 20 step PARITY OK
- Next: 사용자 파일럿 st4-p1 제출, 평가 큐 시작
- By: tskim
