## [2026-10-27] task | E7 회의 통째·실시간 화자 유지(AOSC) 구현

- Changed:
  - 코드
    - `vapasr/hf/diar.py`, `vapasr/hf/long_session.py`, `vapasr/hf/modeling_vapasr.py`, `vapasr/hf/configuration_vapasr.py`, `vapasr/data/cst_dialogue.py`
    - `experiments/diar_train.py`, `experiments/diar_eval.py`, `experiments/diar_extract.py`(새), `experiments/cst_stage3_eval.py`, `experiments/stage3_train.py`
  - 시험: `tests/test_diar.py`(5), `tests/test_slot_input.py`(5)
  - 실행
    - `slurm/submit-q17-e7a.sh`(새), `slurm/submit-evals.sh`(q17-e6-amifull·q17-e7-amifull)
    - `v3_eval_task.sh`(amifull AOSC·DIAR), `raw/sources/experiments/2026-10-26-data-expansion/e7_session_eval.sh`(새)
  - 위키: [[output-diarization-integration-20261025]] 7 절
- Reason:
  - 사용자 "E7 을 먼저 구현해놓자".
  - 창마다 슬롯을 새로 정하는 구조라 회의 통째·실시간 디코드에서 같은 사람이 조각마다 다른 슬롯이 된다.
  - Streaming Sortformer 의 도착 순서 화자 캐시(AOSC)를 헤드에 붙였다. 학습은 스트리밍 흉내 미세조정(E7a), LLM 쪽은 대화 전체 순서 슬롯 학습(E7b, 필요하면)이다.
  - E6 헤드는 diar-q17-e6-head 로 꺼냈다.
- Next:
  - E6 헤드 회의 단위 DER(조각마다 새로 대 AOSC)을 호스트 GPU 4 에서 재는 중이다. E7a 학습 경로 스모크는 GPU 7 에서 돈다.
  - 사용자 제출: `bash slurm/submit-q17-e7a.sh`(1 노드, 성공 뒤 q17-e7-amifull), `bash slurm/submit-evals.sh q17-e6-amifull`(의존 없음).
- By: tskim
