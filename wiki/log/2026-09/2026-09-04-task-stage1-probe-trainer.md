## [2026-09-04] task | Stage 1 probe 학습기 작성 (encoder probing 착수)

- Changed: `vapasr/probe/{__init__,data,model}.py`(신규), `experiments/{train_probe.py,run_stage1.sh,show_probe_results.py}`(신규),
  `vapasr/features/encoders.py`(fbank floor 인코더, device/`to()`, module 참조), `experiments/extract_features.py`(--seg-s, --ids, --device,
  대화별 `empty_cache`), `experiments/run_feature_cache.sh`(turnbench-dev 는 --include-flagged), `task-stage1-encoder-probing` → doing,
  `TODO.md`·`todo.md`.
- Reason: 사용자 요청. 캐시된 frozen 특징 위에 고정 용량 causal probe head(VAP 256 + VAD)를 학습하고 TurnBench dev 를 공식 sweep/scorer 로
  자동 채점하는 파이프라인. 스모크(CPC, 300 step) 통과, 1 epoch ≈ 3 분. 발견·수정: (1) TurnBench scorer 는 dev 38 대화 전부를 요구 —
  플래그(한 채널 무음) 대화 tb-172 도 캐시해야 함, (2) 캐시 작업 2개가 GPU 1 메모리 34.8 GB 를 점유(캐싱 할당기) → 대화별 empty_cache
  추가, 가벼운 인코더는 `--device cpu` 로 우회, (3) 채점 출력을 grep 으로 가리다 coverage 오류를 놓쳤음 → 파일 저장 후 요약.
  첫 CPC 1-epoch 결과(dev, fp≤0.1 sweep): EOT R 0.899 / FP 0.091 / p50 445 ms, INT 0.943 / 0.099 / 893 — VAP 와 같은 FP(0.045)에서의
  비교는 채점 재실행 후 기록.
- Next: 캐시 완료(feat-cache-A/B) → `run_stage1.sh` 매트릭스(인코더 × 고유율/12.5 Hz) → [[task-stage1-encoder-probing]] 판정.
  DualTurn encoder 확보 검토. AI Hub(한국어) 평가는 VAP 지표(val CE/acc)만 — 한국어 이벤트 gold 는 벤치마크 태스크에서.
- By: tskim
