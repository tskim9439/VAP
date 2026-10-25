## [2026-10-25] task | 화자 분리 통합 E0(정답 구간 다시 귀속)·E1(화자 분리 가지) 준비

- Changed: `vapasr/hf/diar.py`(새 — DiarBranch·sort_pit_loss·der_counts·load_encoder), `experiments/diar_train.py`·`experiments/diar_eval.py`(새), `tests/test_diar.py`(새), `slurm/submit-diar-train.sh`(새), `slurm/stage3-train-apex.sbatch`(SCRIPT 환경 변수, 기본은 그대로), `raw/sources/experiments/2026-10-25-ami-sdm-analysis/`(E0 `ami_oracle_diar.py`·README), [[status]]
- Reason:
  - 사용자 "계획대로 진행하는 데 1.7b 모델로 진행해"([[output-diarization-integration-20261025]]).
  - E0: 1.7B 회의 + 4 슬롯 avg3 의 SDM eval δ10 을 정답 화자 구간으로 다시 귀속하면 cp 53.5 → 39.3 이다. 한 스트림 형식의 천장이다.
  - E1: 1.7B 인코더(동결 fp32) 위 화자 분리 가지를 만들었다.
    - 데이터 dry-run 은 19 셋(실제 약 1,050 h, 합성 약 1,900 h)이다.
    - GPU 스모크는 30 step, 0.4–0.7 s/step, 저장 대조 차이 0 이다.
    - AMI 창 DER 평가 경로를 점검했다.
- Next: 사용자가 E1 학습을 제출한다. 그동안 E2(정답 활동 SSA 주입 + 슬롯별 디코드, 1.7B)를 구현한다.
- By: tskim
