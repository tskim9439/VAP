## [2026-10-30] task | LLM 활동 추종 진단 구현 — 추종 지표·cp_follow·활동 게이트 디코드, 서버 스모크

- Changed:
  - 코드: `vapasr/data/cst_eval.py`(act_runs·follow_stats·follow_pool), `vapasr/hf/batch_decode.py`(ActGate·gate_tid·decode_blocks_batch gates), `vapasr/hf/realtime.py`·`long_session.py`(gate)
  - 평가: `experiments/cst_stage3_eval.py`(행 act_runs·follow·cp_follow, `--act-gate`·`--follow-window`·`--follow-n3d`), `ami_report.py`, `v3_eval_task.sh`(GATE=·FOLLOW=1), `slurm/submit-evals.sh`(q17-follow·q17-e11-gate)
  - 시험: `tests/test_cst_eval.py`·`tests/test_slot_blocks.py`
  - 위키: [[output-n3d-branch-20261029]] 6.5–6.6 절
- Reason:
  - 사용자 "LLM 활동 추종 진단 구현해놓자". E11 회의 통째 LLM 혼동(SDM eval 12.5 %)이 전문가 혼동(1.7 %)보다 큰 까닭을 단어 수준에서 본다.
  - 스모크(E11, AMI SDM, 창 16 개·회의 하나): 단어의 91–92 % 가 활동이 켜진 슬롯에 들어간다. 다른 슬롯 단어(7–9 %)는 정답으로 보면 76–91 % 가 활동이 맞다.
    - 어긋난 단어를 활동 슬롯으로 옮기면 cp 42.80 → 37.17(창), 35.19 → 25.77(회의), agn 그대로.
    - 게이트 3,1 은 cp 37.72·26.90, agn +1.5·+2.3.
  - 대조: 게이트 배치 = 직렬 디코드, 다시 계산한 활동 = 디코드 때 활동(모두 같다).
- Next:
  - 사용자 제출: `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-evals.sh q17-follow)`(약 30–40 분), `… q17-e11-gate`(약 4 h)
  - 옮기기가 회의 전체에서도 크면 출력 쪽 재귀속(스트리밍 그대로 — 단어 f + 1 프레임까지의 활동만 쓴다)을 정식 디코드 옵션으로 둘지 정한다.
- By: tskim
