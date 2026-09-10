## [2026-09-10] query | Stage 2 run E2(인코더 해동) 최종 평가 보고서

- Changed: `wiki/outputs/output-stage2-e2-final-eval.md` 신규(§4 스윕·§5 test 는 측정 중)
- Reason: 인코더 해동 + LR 2e-5 로 8 코퍼스를 학습한 E2 가 C2 를 모든 셋에서 앞섬(select δ=2 0.076/0.139/0.167/0.292). 기준 모델 교체
- Next: 스윕·test 수치 채움, E3(추가 epoch)·NEXT_AUDIO 가중 실험, 디코더 최적화
- By: tskim
