## [2026-10-27] task | E8 헤드 출력 지연 R(받아쓰기 지연 안에서 헤드에 앞을 보이기) 구현

- Changed:
  - 코드
    - `vapasr/hf/diar.py`(shift_frames), `vapasr/hf/configuration_vapasr.py`(act_shift), `vapasr/hf/modeling_vapasr.py`(E6 forward 의 헤드 정답·마스크·흔든 정답 입력 이동), `vapasr/hf/long_session.py`(AOSC 와 함께 막음)
    - `experiments/stage3_train.py`(--act-shift), `experiments/diar_train.py`(--shift), `experiments/diar_eval.py`(unshift), `experiments/diar_extract.py`, `experiments/cst_stage3_eval.py`(헤드 바꿔 끼울 때 shift 대조)
  - 제출: `slurm/submit-q17-e8.sh`(새), `slurm/submit-evals.sh` 의 q17-e8
  - 시험: `tests/test_diar.py`·`tests/test_slot_input.py` 각 1 개
  - 위키: [[output-diarization-integration-20261025]] 8 절
- Reason:
  - 사용자 "그런데 아직까지 cpWER 이 한참 모자르네" → "E8 구현해놓자".
  - AMI SDM eval δ10 창 cp 45.35 가운데 귀속이 17.29 다(정답 활동이면 1.4). 인과 헤드가 말 시작을 늦게 잡는 놓침이 크다.
  - LLM 이 단어를 δ 늦게 내므로, 헤드가 R ≤ δ 프레임 뒤를 보고 판단해도 지연이 늘지 않는다.
- Next:
  - 확인: 서버 시험 test_diar·test_slot_input 25 개가 통과했다. E8 인자 그대로 `stage3_train --dry-run` 이 통과했다(DRY_EXIT 0, 지문 act_shift 4·diar_head E1c·slot_input).
    `submit-q17-e8.sh --dry-run` 도 통과했다.
  - 사용자 제출: `bash slurm/submit-q17-e8.sh`(2 노드, E6 과 같은 레시피 + --act-shift 4, 평가 q17-e8 자동).
  - AOSC(E7a)와 합치는 것은 E8 결과를 본 뒤에 한다.
- By: tskim
