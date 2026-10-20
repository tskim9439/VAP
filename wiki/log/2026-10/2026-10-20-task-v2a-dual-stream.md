## [2026-10-20] task | V2-a 두 스트림 입력 구현 — Nemotron L00 탭 + 0 초기화 adapter_tap, 학습 전 점검 시작

- Changed:
  - `vapasr/features/online.py`(`taps=`), `vapasr/hf/modeling_vapasr.py`(`add_enc_tap`·`adapter_tap`·두 스트림 `encode`/`chunk_embed`), `vapasr/hf/configuration_vapasr.py`(`enc_tap_layer`·`enc_tap_hidden`)
  - `vapasr/hf/offline_decode.py`·`experiments/eval_single_turn_asr.py`(스트림 축), `vapasr/hf/live.py`(두 스트림 모델 거부)
  - `experiments/stage3_train.py`·`slurm/submit-stage3-train.sh`(`--enc-tap-layer`, 지문 키 `enc_tap`), `experiments/spk_probe.py`(LLM 입력 = chunk_embed)
  - `tests/test_enc_tap.py`(새), `tests/test_offline_mode.py`
  - raw [[2026-10-20-v2a-dual-stream]](README·`v2a_check.py`·`v2a_check.sh`·`cmp_v2a.py`), [[output-v2-speaker-memory-plan-20261020]], [[status]]
- Reason: 사용자 "VibeVoice 에서는 Speech Encoder 를 2개써서 완화된거 같다" → V2-a 제안 → "그래 학습 전 점검까지 준비해."
- 결과:
  - 단위 시험: 로컬 tap 시험 4 개 통과, mxc 54 통과·3 건너뜀(varlen packing 포함)
  - 학습 dry-run(st4-m1 설정 + `--enc-tap-layer 0`): 11 셋 352 항목 불일치 0, 지문 `enc_tap {layer 0}`
  - GPU 단계(실제 모델 점검·스모크 20 step·평가 경로)는 호스트 GPU 가 비기를 기다린다(빈 메모리 70 GB 이상).
  - 사용자 "LLM 에 Head 를 하나 추가해서 학습시에만 활용…" → 그 헤드(`--spk-act-weight`)는 V1 에 이미 있고(학습 배치 정확도 0.94–0.95 에서 멈춤), V2-a 와 짝(신호 + 정보 경로)으로 쓴다.
- Next: GPU 점검 → 파일럿 제출 명령(제안: st4-m2a 설정·seed + `--enc-tap-layer 0`, m2a 가 대조군)
- By: tskim
