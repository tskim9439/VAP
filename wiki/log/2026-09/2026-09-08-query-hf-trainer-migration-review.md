## [2026-09-08] query | HF Trainer 마이그레이션 구현 검토

- Changed: `wiki/outputs/output-huggingface-training-framework-choice.md`
- Reason: [[source-hf-trainer-migration]]과 실제 구현을 대조해 legacy 패리티·Liger·2-GPU 분산 학습 검증을 통과로 판정하고, 8-node 실행 전 adapter 인자 누락과 정확한 data resume·recipe override·평가 이력·부분 checkpoint 위험을 기록했다.
- Next: `slurm/s3_train_hf.sbatch`의 초기화 변수 순서를 먼저 수정하고, 무중단/재개 parameter hash 시험과 8-node 짧은 smoke를 수행한다.
- By: tskim
