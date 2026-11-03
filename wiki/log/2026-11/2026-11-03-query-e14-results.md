## [2026-11-03] query | E14 회의 통째 결과·X 칸 귀속 진단, 평가 재제출

- Changed:
  - 코드: `experiments/cst_stage3_eval.py`·`cst_stage4_eval.py`(X 칸 `<EOT>` 버림, 커밋 0c85380), `slurm/submit-evals.sh`·`raw/sources/experiments/2026-10-21-v3-attribution/v3_eval_task.sh`(재제출 — 커밋 9d29cd2), `raw/sources/experiments/2026-11-03-e14/auto_e14.sh`(커밋 62bd7b5), `raw/sources/experiments/2026-11-03-e14/x_attr_diag.py`(새 진단)
  - 위키: [[output-e14-xslot-20261103]] 6–8 절(학습·평가·결과·다음), [[output-ami-results-summary-20261030]] 1.1 표에 E14 두 줄
- Reason:
  - 사용자 "진행현황 확인해줘": E14 평가 99730 의 창 몫 0·1 과 번역 99731 이 X 칸 `<EOT>` 로 멈췄다. 고친 뒤 사용자가 99831(apex, 99829 뒤)로 다시 냈다.
  - 회의 통째 표준 결과: IHM-mix eval cp 24.39 → 26.32, 나머지 분할은 E13 수준(dev −0.3). agn 은 IHM-mix eval 23.09 → 21.74.
  - 진단: 손해의 대부분(+1.56)이 ES2004b IHM-mix 한 회의다. N3D 가 한 화자를 2 분 넘게 놓쳤고, X 칸이 그 말 1,495 단어를 되찾았지만 85 개만 맞는 화자에 붙었다(N3D 확률에 근거가 없다).
- Next:
  - 99831 창 평가·번역 결과(자동 보고 `e14-report.txt`).
  - X 칸 귀속을 모델이 하도록(X 칸에서 화자 슬롯도 내게 학습) 할지 G2b·G3 결과와 함께 정한다.
- By: tskim
