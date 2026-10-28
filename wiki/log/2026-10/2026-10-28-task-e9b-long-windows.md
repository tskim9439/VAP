## [2026-10-28] task | E9b — 학습 창 20–90 s(E9 의 긴 창 붕괴 대책) 구현

- Changed:
  - 코드: `experiments/stage3_train.py`(--shrink-speakers — 격자 창도 화자 상한을 넘으면 다섯 번째 화자 직전까지 줄임, 지문 키 shrink_speakers)
  - 제출: `slurm/submit-q17-f1.sh`(--window lo,hi 넘기기), `slurm/submit-q17-e9b.sh`(새), `slurm/submit-evals.sh` 의 q17-e9b
  - 위키: [[output-diarization-integration-20261025]] 9 절 E9 결과 1·E9b
- Reason:
  - 사용자 "학습 창을 90 s 까지 늘려서 E9b 구현해놓자".
  - E9 AMI 창 δ10:
    - 60 s 미만 창은 모든 분할에서 E6 대비 cp 2.3–3.3 pt 낮았다(IHM-mix eval 36.45 → 33.37, SDM 43.93 → 40.61, agn 도 낮다).
    - 60 s 넘는 창은 `<EOT>` 반복(같은 슬롯 연속, 반복의 99 % 가 60–90 s 구간)으로 무너졌다.
    - 학습 창은 20–40 s(약 3,200 위치)이고, 평가 90 s 창은 약 7,000 위치다.
- Next:
  - 확인: 맥 시험 31 개가 통과했다(test_stage3_train·test_slot_blocks·test_v2_slots). 서버 코드 md5 전수 대조에서 실행 파일은 모두 같다.
  - 서버 dry-run(E9b 인자 그대로)은 19 셋 불일치 0 이다. 화자 상한 줄이기로 dlg_mt 755·dlg_far 3,235 창을 살렸다.
  - 사용자 제출: `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-e9b.sh)` — E6 final 에서 5,000 step, 2 노드, 학습 뒤 평가 q17-e9b.
  - E9 회의 통째(문맥 30 s) 평가가 끝나면 E9·E9b 와 같은 표에 넣는다.
- By: tskim
