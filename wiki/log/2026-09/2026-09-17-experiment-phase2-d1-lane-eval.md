## [2026-09-17] experiment | Phase 2 D1 lane 디코드 평가

- Changed: wiki/outputs/output-phase2-d1-lane-eval.md(신규), wiki/tasks/task-phase2-data-prep.md, raw/sources/experiments/2026-09-17-phase2-d1-lane-eval/(reports·windows JSON; audio 는 git-ignore), vapasr/hf/lane_state.py, experiments/p2_eval_lanes.py, vapasr/data/dialogue_corpora.py(CHiME-6·NIKL 로더)
- Reason: D1 실학습 뒤 free-running lane 디코드가 실제로 동작하는지 seen·held-out 창에서 확인하고 오디오와 함께 시각화(사용자 요청)
- Next: D1b(δ_onset>0, lane 3–6 노출, held-out split), 디코더 임계값 sweep, held-out 정렬
- By: tskim

## [2026-09-17] experiment | D1 인코더 저장 사고 수정·재평가·문제 분석

- Changed: vapasr/hf/{configuration,modeling}_vapasr.py(encoder_saved), experiments/p2_train_hf.py(--check-save·encoder_saved 자동), experiments/p2_fix_encoder.py, experiments/p2_tf_probe.py, wiki/outputs/output-phase2-d1-lane-eval.md(전면 수정), raw reports
- Reason: 사용자 지적(첫 블록 ONSET·낮은 WER/CER) 분석 중 재로드 모델의 teacher-forced 손실이 학습 로그와 다름을 발견 → 인코더 미저장 사고 확인·수정
- Next: onset_thr 0.35(c3) 반영, D1b(창 시작 평탄화·δ_onset·태그 가중), held-out split
- By: tskim

## [2026-09-17] query | Phase 2 평가 계획 v2

- Changed: wiki/outputs/output-phase2-eval-plan.md(신규)
- Reason: 사용자 지적 — lane 번호 기반 CER/WER 가 화자 오배정·순서 어긋남을 전사 점수에 전가, 턴 구간 정량 지표 부족, 학습 미사용 객관 평가셋 필요
- Next: lane_metrics 모듈·테스트, NIKL 2020 정렬, TurnBench 로더, 세션 split 파일, p2_eval_sessions
- By: tskim
