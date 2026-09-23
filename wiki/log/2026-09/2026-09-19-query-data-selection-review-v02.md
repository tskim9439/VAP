## [2026-09-19] query | 데이터 선별 시간축 감사·청취 검수 실행

- Changed: [[output-data-selection-review-v02]], [[output-data-selection-execution-plan]], [[status]]; `selection_review_audit.py`, `selection_review_bundle.py`, `run_selection_review.sh`, 감사 테스트와 새 실험 산출물.
- Reason: 사용자의 다음 단계 진행 요청. 완료된 426만 행 QC를 검증하고 일정한 길이 차이를 연도별로 발견했다. 원본·기존 라벨을 보존하며 886개 청취 표본과 3교사 대조를 실행했다.
- Next: 3교사 대조 및 추가 48 kHz 가설 진단 완료. NIKL 2022 두 세션 16개에서 CER 93.59→14.74%로 개선됐다. 사람 청취 및 포맷/시간축 대응 검증 후 교사·규칙을 동결하고 전량 screening. source hold와 자동 학습 승인 금지 유지.
- By: tskim
