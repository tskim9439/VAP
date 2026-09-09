## [2026-09-09] task | mxc 모델 로드 지연 해결 — 로컬 디스크 컨테이너·노드 스테이징·인코더 캐시

- Changed: `wiki/sources/source-mxc-model-load-latency.md`, `scripts/stage-env-local.sh`, `scripts/activate-env.sh`, `vapasr/hf/stage.py`, `vapasr/features/online.py`(인코더 캐시), `slurm/s3_train_hf.sbatch`·`s2_prep.sbatch`·`s3_eval.sbatch`, `.env`(sa_tskim_fd)
- Reason: NFS 위 conda env 때문에 추론 로드 16 분·학습 시작 25 분. 로컬 디스크로 옮겨 17 초.
- Next: 67293(D2) requeue 에서 노드 스테이징 동작 확인(`[stage]`·`로컬 env 사용` 로그)
- By: tskim
