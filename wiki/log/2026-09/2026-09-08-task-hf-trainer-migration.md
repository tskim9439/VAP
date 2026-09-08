## [2026-09-08] task | 모델·학습 코드 HF 전환 (vapasr.hf + VapAsrTrainer) 구현·검증

- Changed: `vapasr/hf/{configuration_vapasr,modeling_vapasr,liger,data,trainer}.py`, `experiments/s3_train_hf.py`, `experiments/hf_parity.py`, `slurm/s3_train_hf.sbatch`, `scripts/activate-env.sh`, `wiki/sources/source-hf-trainer-migration.md`, `raw/sources/experiments/2026-09-08-hf-trainer-migration/`
- Reason: [[output-huggingface-training-framework-choice]] 의 권장(HF-native 모델 + Trainer, Liger 는 동등성 확인 후) 을 구현. 기존 ckpt 와 수치 패리티(손실·디코드 동일), Liger +24 % 처리량, 2-GPU 스모크에서 학습·분산 평가·저장·재개·선점 동작 확인.
- Next: 8-노드 SLURM 실전 run(hf-C3), select/final 평가 스크립트 HF 이관, DeepSpeed/FSDP 는 인코더 해동 시 검토.
- By: tskim
