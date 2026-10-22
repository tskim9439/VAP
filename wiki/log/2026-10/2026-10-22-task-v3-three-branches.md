## [2026-10-22] task | V3 다음 세 갈래 — R1x 쌍·R2m 제출, 체크포인트 평균

- Changed: `vapasr/hf/stage3.py`(decision_weights), `experiments/stage3_train.py`·`slurm/submit-stage3-train.sh`(--same-lang-change-weight·--same-lang-hold-weight), `experiments/avg_checkpoints.py`, `tests/test_v3_weights_avg.py`, 평가 스크립트 `MODEL_ROOT`, `v3_check.sh` r1x·r1x4·r2m, `raw/sources/experiments/2026-10-21-v3-attribution/README.md`(14–15 절), [[status]]
- Reason: 사용자 "셋다 진행해보자" — R1x `v3-p1-tel-xlw`(job 94613)·R1x-a `v3-p1-tel-xlw-arrival`(94614)·R2m `v3-p1-sl15`(94615)를 사용자가 제출했다. R2x·R4x 의 ck1500·ck2000·final 가중치 평균을 만들어(비트 대조 OK) 평가 큐를 걸었다.
- Next: 세 학습 PARITY, 평가 큐 12 + 평균 4 완료 뒤 R1x 대 R1·R2x, R1x-a 대 R1x, R2m 대 R2x·R2(같은 언어 가중 0·1.5·3), 평균 대 단일 체크포인트 비교.
- By: tskim
