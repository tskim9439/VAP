## [2026-09-18] query | 데이터 선별 CPU worker 확대와 재개 기록

- Changed: [[output-data-selection-execution-plan]], [[status]], `experiments/selection_audio_census_resume.py`, `tests/test_selection_audio_resume.py`.
- Reason: 사용자 요청에 따라 mxc CPU QC를 8→32 worker로 확대했다. 기존 검사·원본·완료 결과를 보존하고 부분 결과를 검증해 재개했다. GPU 동시 허용량 최대 4장도 현황에 반영했다.
- Next: 새 `audio-census-w32` 로그로 전수 검사 완료 확인, 시간축 복구와 사람 표본 검수 후 teacher screening 진행. 완료 marker의 output 경로를 따라 재사용된 결과를 수집할 것.
- By: tskim
