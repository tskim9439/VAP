## [2026-10-25] task | 화자 분리 통합 E2(정답 활동 주입 + 화자별 디코드) 구현

- Changed:
  - 데이터: `vapasr/data/cst_dialogue.py`(target_spk·cst_sequence target·collate target_act·한 슬롯 점검 허용)
  - 모델: `vapasr/features/online.py`(pre-encode inject 훅), `vapasr/hf/modeling_vapasr.py`(adapter_inject·add_spk_inject·encode act·forward target_act), `vapasr/hf/configuration_vapasr.py`(spk_inject_hidden), `vapasr/hf/trainer.py`(target_act 전달)
  - 학습·평가: `experiments/stage3_train.py`(--target-spk·--inject-hidden·dry-run 대상 시퀀스 점검·지문 target_spk), `experiments/cst_stage3_eval.py`(--target-oracle, 창 묶음 배치)
  - 시험: `tests/test_target_spk.py`(새), `tests/test_cst_decision.py`(소스 검사 문자열)
  - 제출·점검: `raw/sources/experiments/2026-10-21-v3-attribution/v3_eval_task.sh`(amitgt), `slurm/submit-q17-e2-tgt.sh`(새), `slurm/submit-evals.sh`(q17-e2-tgt), `raw/sources/experiments/2026-10-25-ami-sdm-analysis/e2_check.sh`(새), [[status]]
- Reason:
  - 사용자 "계획대로 진행하는 데 1.7b 모델로 진행해"([[output-diarization-integration-20261025]] E2 — 정답 활동 상한).
  - SSA(arXiv:2506.22646)처럼 대상 화자 활동을 인코더 pre-encode 출력에 주입한다(f(X⊙y) + X, 마지막 Linear 0 초기화 → 시작 = init 모델). 창마다 그 화자만 한 슬롯으로 받아쓴다.
  - 기본 경로는 시퀀스·난수가 그대로다(골든 시험 통과). 1.7B 점검:
    - dry-run 의 대상 시퀀스 불일치는 0 이다.
    - 스모크는 20 step 이고, 저장 대조 차이는 0, 주입 마지막 층 norm 은 0 → 2.04 다.
    - 평가 경로(AMI SDM 6 창)를 확인했다.
- Next:
  - 사용자가 E2 를 제출한다(`submit-q17-e2-tgt.sh`, 학습 4,000 step + AMI 정답 활동 평가).
  - E1(95482)이 끝나면 AMI DER 을 잰다.
  - E3 은 E1 예측 활동으로 E2 를 평가한다.
- By: tskim
