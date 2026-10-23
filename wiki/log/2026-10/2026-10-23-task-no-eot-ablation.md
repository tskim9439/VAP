## [2026-10-23] task | `<EOT>` 없는 대화 학습 시험

- Changed: `vapasr/data/cst_dialogue.py`(no_eot — window_streams(4)·build_cst(4)_sequence·decision_positions anchors·검사), `experiments/stage3_train.py`(`--no-eot`, 지문 키 no_eot), `slurm/submit-stage3-train.sh`(`--no-eot`), `slurm/submit-q06-noeot-chain.sh`(새 파일), `slurm/submit-evals.sh`(noeot-v3, v3set), `tests/test_cst_decision.py`·`tests/test_v3_weights_avg.py`, `raw/sources/experiments/2026-10-21-v3-attribution/`(README 22·24 절, `eot_switch.py`, `v3_check.sh` r1xne), [[status]]
- Reason:
  - 사용자 "<EOT> 를 없애는게 더 평가수치가 좋지 않았었나?" → "턴 예측은 주된 목표가 아니야. 화자귀속이 더 중요한데 … '<EOT> 를 낸 뒤에 교대한다' 를 배우지 않을까 걱정" → "모델이 어차피 <EOT> 를 내지 않는다면 애초에 학습할 필요도 없는거 아닌가?" → "진행하도록 해".
  - 22 절 분석에서 교대는 `<EOT>` 에 묶여 있지 않았다. 학습 문맥과 추론의 `<EOT>` 불일치 효과는 빼고 학습해 가른다.
- Next:
  - 94867 시작 로그를 확인한다(`<EOT>` 없음, n_eot 0).
  - 평가 뒤 R1x 와 미리 정한 기준으로 비교한다.
  - 채택하면 1.7B 체인 5·6 단계를 다시 낸다(사용자 취소·재제출).
- By: tskim
