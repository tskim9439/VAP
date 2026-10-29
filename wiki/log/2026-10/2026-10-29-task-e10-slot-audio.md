## [2026-10-29] task | E10 — 한 스트림 안 슬롯별 소리 토큰(슬롯 추출기) 구현·서버 스모크

- Changed:
  - 코드
    - `vapasr/hf/slot_audio.py`(새 — SlotAudio)
    - `vapasr/hf/modeling_vapasr.py`(add_slot_audio·slot_audio_of·encode 셋째 스트림·slot_x·mix_mask), `vapasr/hf/batch_decode.py`·`realtime.py`·`long_session.py`(mix_off·슬롯 소리)
    - `experiments/stage3_train.py`(--slot-audio·--mix-drop 등), `experiments/cst_stage3_eval.py`(--mix-off), `raw/sources/experiments/2026-10-21-v3-attribution/v3_eval_task.sh`(MIXOFF)
  - 제출: `slurm/submit-q17-e10.sh`(새), `slurm/submit-evals.sh` 의 q17-e10
  - 시험: `tests/test_slot_audio.py` 7 개(새)
  - 위키: [[output-diarization-integration-20261025]] 10 절 E10 구현
- Reason:
  - 사용자 "E10 구현해놓자"(10 절 제안 — 화자 혼동의 근본 원인은 LLM 이 섞인 소리에서 단어마다 슬롯을 고르는 데 있다).
- Next:
  - 확인:
    - 맥 시험 7·관련 183 개, 서버 시험 39 개가 통과했다. 코드 md5 전수 대조에서 실행 파일은 같다.
    - 서버 GPU 스모크(E9 final + 추출기 16.6M, AMI 창 3 개): forward/backward 가 유한하고 메모리 최대 39.1 GB 다.
    - 같은 창을 배치 디코드와 추측 prefill 디코드로 풀면 섞인 소리 켬·끔 모두 출력이 같다.
  - 사용자 제출(11:50, low_p_hpc): `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-e10.sh --partition low_p_hpc)`
    — 학습 job 97922, 평가 job 97923(q17-e10: 창·회의 통째 × 섞인 소리 켬·끔).
  - B(헤드 키우기)·A′(인코더 위층 슬롯별 재실행)는 결정 대기다.
- By: tskim
