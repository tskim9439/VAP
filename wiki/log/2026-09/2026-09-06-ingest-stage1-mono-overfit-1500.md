## [2026-09-06] ingest | Stage 1 mono overfit 1,500-step 완료

- Changed: `raw/sources/experiments/2026-09-06-stage1-mono-overfit-1500.md`, `wiki/sources/source-stage1-mono-overfit-1500.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/status.md`
- Reason: @900부터 EN·KO 내용·방출률·설계 지연·evidence-time이 모두 수렴해 @1500까지 유지된 결과를 근거로 overfit 기능 관문을 통과 처리하고, GPU 경합으로 오염된 tick 측정과 여전히 남은 80 ms 실시간성 관문을 분리했다.
- Next: `align2` 완료 후 `overfit-v2` 선택 ID의 KO 변경 타깃·EN 다중발화 경계 coverage를 확인하고, 부족하면 표적 회귀 표본을 추가한다. 900-step 재검증 통과 시 대조군 측정 후 6,000-step 파일럿을 실행하며 최종 tick은 한가한 GPU에서 단독 측정한다.
- By: tskim
