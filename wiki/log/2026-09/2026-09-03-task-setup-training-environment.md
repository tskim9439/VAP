## [2026-09-03] task | 컨테이너 학습 환경 구축 완료

- Changed: `wiki/tasks/task-setup-training-environment.md` → `done`. 신규
  `scripts/setup-container-env.sh`, `scripts/activate-env.sh`, `scripts/smoke-test-models.py`,
  `env/requirements.txt`, `env/requirements-lock.txt`, `env/README.md`. `.env` 에 캐시·conda 항목 추가.
  `sync-rack4.sh exec/shell` 이 activate-env.sh 를 자동 로드. `wiki/sources/source-nemotron-3-5-asr-streaming.md`,
  `source-qwen3-asr.md`, `question-encoder-lookahead-and-causality.md` 에 실측 반영.
  `wiki/todo.md`, `wiki/log.md`, `TODO.md`, `status.md` 갱신.
- Reason: Phase 0 첫 태스크. rack4 `tskim_env` 에 conda env `vapasr`(Python 3.11) 를 만들고
  torch 2.6.0+cu124 / NeMo git main 3.1.0 / transformers 4.57.6 / qwen-asr 0.0.6 / 원 VAP 계열 3개를
  설치했다. 스모크 4종 통과: Nemotron(638M, 79.4 ms/frame, ctx [56,0]·[56,3]), Qwen3-ASR(audio_tower 186M),
  ForcedAligner(한국어 지원), 원 VAP(CPC 5.8M, 50 Hz). 발견 두 가지 — (1) Nemotron 의 언어는
  `transcribe(target_lang=)` 키워드가 dataset 까지 도달하지 않아 **manifest `"lang"` 필드**로 줘야 한다
  (PyPI 3.0.0·git main 공통). (2) `qwen-asr` 가 transformers 를 5.x→4.57.6 으로 다운그레이드하나
  NeMo 와 충돌 없음. 캐시는 `/data3/tskim/cache`, 서드파티 코드는 `/data3/tskim/third_party`.
- Next: [[task-audit-encoder-causality-lookahead]] (Nemotron 은 문서상 우측 context 확인, conv 암묵
  lookahead 절단 실험 남음; Qwen AuT 는 chunk 직접 호출 코드 필요), [[task-verify-aihub-stereo-and-access]],
  [[task-reproduce-vap-turnbench-baseline]] (missing 8 키 확인).
- By: tskim
