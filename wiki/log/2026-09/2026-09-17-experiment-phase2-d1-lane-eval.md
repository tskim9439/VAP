## [2026-09-17] experiment | Phase 2 D1 lane 디코드 평가

- Changed: wiki/outputs/output-phase2-d1-lane-eval.md(신규), wiki/tasks/task-phase2-data-prep.md, raw/sources/experiments/2026-09-17-phase2-d1-lane-eval/(reports·windows JSON; audio 는 git-ignore), vapasr/hf/lane_state.py, experiments/p2_eval_lanes.py, vapasr/data/dialogue_corpora.py(CHiME-6·NIKL 로더)
- Reason: D1 실학습 뒤 free-running lane 디코드가 실제로 동작하는지 seen·held-out 창에서 확인하고 오디오와 함께 시각화(사용자 요청)
- Next: D1b(δ_onset>0, lane 3–6 노출, held-out split), 디코더 임계값 sweep, held-out 정렬
- By: tskim
