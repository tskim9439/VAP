## [2026-09-15] query | 청소년 134-2 조각 채널 기원·완전성 검증

- Changed: [[output-phase2-data-inventory]] §6 신규; [[output-phase2-lane-plan]] §8 데이터 표에 청소년 행 추가; `raw/sources/experiments/2026-09-15-phase2-youth-crop-origin/`(스크립트·출력).
- Reason: 청소년 조각을 학습셋에서 제외한 이유 질문 → 검증 지시. 겹침 조각 쌍 상호상관으로 채널 기원 통과(성인 대조군과 동일 분포, 인위 혼합과 상이), 대화별 완전성은 실외 523대화 90.1 h 완전·실내 완전본 없음.
- Next: Q0 에서 실내 완전 창 단위 유효 시간 산출, ≥95 % 대화의 결손 마스크 규칙 구현. 다른 세션 WIP 인 [[output-phase2-training-db]] 는 그쪽 커밋 뒤 반영.
- By: tskim
