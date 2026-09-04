## [2026-09-03] task | 특징 캐시 파이프라인 구축·검증, 추출 시작 (Phase 1)

- Changed: `vapasr/features/{__init__,encoders}.py`(신규), `experiments/{extract_features.py,run_feature_cache.sh,make_16k_copy.py,diag_length_invariance.py,diag_stitching.py,diag_qwen_determinism.py}`(신규),
  `experiments/qwen_aut_mask.py`(causal 모드 좌측 창), `wiki/outputs/output-feature-cache-and-compute-budget.md`(신규),
  `task-compute-budget-and-feature-cache` 진행, `source-qwen3-asr`(13 Hz), `index.md`·`todo.md`·`status.md`.
  서버: `otoSpeech16k` 사본(420 대화), `/data3/tskim/features/` 추출 시작(feat-cache-A: cpc·nemotron-c0, B: qwen ×2·wavlm ×2).
- Reason: Stage 1 encoder probing 을 분 단위로 돌리기 위한 frozen 특징 캐시. 인코더 7종을 공통 인터페이스로 감싸고
  긴 파일 세그먼트 처리를 fp32 무분할과 일치시키는 과정에서 네 가지 함정을 실측으로 잡았다: (1) 층 누적 수용장(Nemotron 107 s →
  겹침 120 s), (2) TF32 노이즈, (3) 출력 프레임 수 ≠ 길이×Hz — **Qwen AuT 는 1 s 당 13 프레임(13 Hz)** 이라는 사실 포함,
  (4) Whisper 프론트엔드의 utterance 정규화. 최종 검증 cpc/nemotron 0 프레임, qwen ≤5 프레임(수치 드리프트). RTF: cpc 0.0018,
  nemotron 0.0052, qwen 0.018, wavlm 0.019 → 214 h 에 ≈17 GPU 시간, 저장 ≈430 GB. 사용자 질문("encoder 도 학습시켜야
  하지 않나")에 단계 구분을 답함 — Stage 1 frozen(H1 검증) → Stage 2 unfreeze → Qwen causal fine-tune.
- Next: 캐시 완료 후 stats 로 표 갱신·task done. [[task-stage1-encoder-probing]] 의 probe head 학습 코드(캐시 로더 + VAP head + TurnBench 평가).
  [[task-event-label-heuristics-validation]] 은 dev gold 로 병행 가능.
- By: tskim
