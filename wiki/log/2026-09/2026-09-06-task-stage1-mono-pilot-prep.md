## [2026-09-06] task | Stage 1 mono 파일럿 준비 완료 — overfit 통과(옛·새 규약), 대화 코퍼스 mxc 업로드, 6,000-step 파일럿 시작

- Changed: `task-uslm-u1-interleaved-asr`(U1a-0 절 신설: 데이터·검증·코드 결함 수정·overfit 결과·업로드), `plans/stage1-mono-pilot.md`(§3.3 텍스트 규약, §4 flush 규약, §8 판정 기록),
  `.env`(MXC_OTOSPEECH_DIR·MXC_AIHUB_*·MXC_TURNBENCH_* 추가), `raw/sources/experiments/2026-09-05-asr-output-style-probe.md`(신설).
  코드: `vapasr/data/{kspon,streams,textnorm}.py`, `vapasr/uslm/{mono_data,mono_model}.py`, `experiments/s1_{verify_data,build_manifest,align,extract_features,train_mono,tok_check}.py`, `vapasr/data/interleave.py`(끝 토큰 폐기 결함 수정).
- Reason: [[decision-mono-input]] 에 따른 단일 화자 mono 경로 신설. 검증기·manifest·정렬·특징 캐시·데이터셋·모델·학습기를 새로 두고 overfit 으로 파이프라인 정상성을 확인했다.
  텍스트 규약은 Qwen3-ASR·Nemotron 출력 실측(EN 숫자 단어, KO 숫자 한글 읽기)에 맞춰 `textnorm.py` 로 통일하고 KsponSpeech 를 `align2/` 에 재정렬했다.
  overfit: 옛 규약 1,500 step·새 규약 900 step 모두 EN/KO 오류 0, tok/chunk = 참조, evidence 위반 0, 지연 p50 ≈ +200 ms(δ=2). tick p99 151–186 ms 는 80 ms 미달 → 실시간성 관문은 단독 측정 전까지 미통과.
  대화 코퍼스(otoSpeech16k 23 GB, AI Hub 71631 wav 54 GB, TurnBench 17 GB)는 rack4 → 맥 T5 → mxc `/soundai/DB/raw/` 로 업로드(rack4→mxc 직접 전송은 정책상 불가, 맥에서는 HF 차단).
- Next: 6,000-step 파일럿(`s1-mono-pilot`, sentinel 1 k 마다) → `--select` 큰 dev 표본으로 ckpt·bias 선택 → `--final` 보고 세트 전량. 대조군(Nemotron RNN-T `[56,0]`·Qwen 오프라인, eval_other 는 같은 2,687 개) 측정.
  tick p99 단독 측정 + CUDA graph/배치 디코드 검토. mxc 잡파일(`._*` 70 개, `_xfer_test` 768 MB)·rack4 azcopy 정리는 사용자 승인 대기.
- By: tskim
