## [2026-10-28] task | E9 슬롯 블록 형식(청크마다 모든 슬롯 태그를 입력, <BEGIN>) 구현·서버 확인

- Changed:
  - 코드
    - 데이터 `vapasr/data/cst_dialogue.py`(block_streams·build_block_sequence·block_problems, slot_blocks·declare_range·block_sequence·slot_mask_chunks, collate tag_chunk),
      `vapasr/data/dialogue_dataset.py`(_bad_spans_spk), `vapasr/hf/packing.py`(tag_chunk)
    - 모델 `vapasr/hf/modeling_vapasr.py`(add_slot_blocks — <ONSET> 151721 재초기화·차단 해제, adapter_tagact, build tag_chunk), `vapasr/hf/trainer.py`(tag_chunk 전달)
    - 디코드 `vapasr/hf/batch_decode.py`(BlockState·decode_blocks_batch), `vapasr/hf/realtime.py`(realtime_block_decode·GraphThinker.run_all),
      `vapasr/hf/long_session.py`(LongThinker.run_all·long_block_decode)
    - 평가 `vapasr/data/cst_eval.py`(parse_emitted begin_id·events, activity_from_events, emitted_from_blocks), `experiments/cst_stage3_eval.py`(자동 선택·--oracle --blocks·llm_der·chunk_ms),
      `experiments/cst_attr_errors.py`, `raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py`(LLM DER·청크 ms)
    - 학습 `experiments/stage3_train.py`(--slot-blocks·--begin-weight·--tag-act-lags·--declare-range)
  - 제출: `slurm/submit-q17-e9.sh`(새), `slurm/submit-q17-f1.sh --transcribe-only`, `slurm/submit-evals.sh` 의 q17-e9
  - 시험: `tests/test_slot_blocks.py` 13 개(새), `tests/test_cst_decision.py`(trainer 키)
  - 위키: [[output-diarization-integration-20261025]] 9 절
- Reason:
  - 사용자 "그럼 시퀀스를 개선해볼까? … 1 블록 내에서 <SPK_A~D> 가 한번씩 무조건 입력으로 주어지게끔" → "<BEGIN> 등의 특수 토큰도 예측하도록" → "응 그렇게 E9 구현해놓자".
  - 작업 중 추가 요청 "<SPK_*> 의 수를 추후에 조절할 수 있도록 해줘. 즉 최소 샘플의 화자 수, 최대 4명 화자로 해서 랜덤으로 선택되게끔" → --declare-range(학습 창마다 선언 슬롯 수를 [창 화자 수, 4] 에서 고르게, 추론은 --declare).
  - 7 절 결과 4: 태그가 헤드 활동만 따라도 회의 통째 cp 가 18–25 pt 내린다 — 태그를 LLM 이 고르지 않게 하고 태그에 활동을 붙인다.
- Next:
  - 확인: 맥 시험 13 개·관련 223 개, 서버 시험 97 개 통과.
    - 첫 서버 dry-run 이 δ4 순서 위반 21 창을 찾았다. <BEGIN>·<EOT> 청크와 글자 버킷의 부동소수 경계 차이였고, 고친 뒤 dry-run 은 19 셋 모두 불일치 0 이다.
    - AMI 오라클은 cp 0, LLM DER 바닥값 δ4 0.3–0.5 %·δ10 1.2–1.4 % 다.
  - 사용자 제출(15:51): `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-e9.sh)` — 학습 job 97163(hpc-34·35), 평가 job 97164(97163 성공 뒤).
    E6 final 에서 5,000 step, 2 노드, 평가 q17-e9(AMI 창 δ10 + 회의 통째 AOSC·E7a δ10).
  - 판단: E6·E7b·태그 이끌기와 같은 조건(δ10, 표준 정규화)으로 창 cp·회의 통째 cp·LLM DER·청크 ms 를 외부 기준과 함께 비교한다.
- By: tskim
