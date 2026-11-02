## [2026-11-02] task | E13 — 활동 게이트 슬롯 블록 + 슬롯 블록 번역 구현, 서버 데이터 점검 통과

- Changed:
  - 코드(커밋 dd50274·f17e9ca): `vapasr/data/cst_dialogue.py`(gate_rounds, 번역 칸, block_problems, 데이터셋 slot_gate·gate_fa·번역 모드), `vapasr/data/cst_eval.py`, `vapasr/hf/batch_decode.py`, `vapasr/hf/realtime.py`, `vapasr/hf/long_session.py`, `experiments/stage3_train.py`, `experiments/cst_stage4_eval.py`, `slurm/submit-q17-f1.sh`(--blocks-trans), `slurm/submit-q17-e13.sh`, `slurm/submit-evals.sh`(q17-e13*), `tests/test_slot_gate_trans.py`
  - 위키: [[output-e13-gate-trans-20261102]] 새 페이지
- Reason:
  - 사용자 "번역을 다시 넣어서 학습할 수 있도록 코드를 작성해줘. … <SPK_A~D> 까지 활성화할 필요는 없을거 같아. 이 고정 슬롯도 해제해보자." — 선택: 활동 있는 슬롯 태그만, 받아쓰기 + 번역 둘 다.
  - 시험: 새 10 개 통과, 바뀐 모듈을 쓰는 시험 315 개 통과(rapidfuzz 없는 2 개 제외). 서버 dry-run 19 셋 불일치 0.
- Next:
  - 사용자 제출 `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-e13.sh --partition low_p_hpc)`.
  - 번역 평가 묶음(cst_stage4_eval 슬롯 블록 경로).
- By: tskim
