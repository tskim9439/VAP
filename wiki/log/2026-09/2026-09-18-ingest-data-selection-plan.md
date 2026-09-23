## [2026-09-18] ingest | 데이터 품질 선별 계획 구체화와 DB별 감사

- Changed: [[source-data-selection-plan]], [[output-data-selection-execution-plan]], `wiki/status.md`; 선별 모듈·전수 감사·교사 worker·테스트.
- Reason: 사용자가 DataSelectionPlan 원안 검토와 DB별 선별 실행을 요청했다. 원본 불변, 평가셋 보호, 전사/시각/화자/턴 품질 분리 및 최대 GPU 2장 제한을 반영했다.
- Next: MNSC CSV/파일 매핑 복구, 전수 오디오 QC 및 시간축 수정, 사람 표본 검수와 규칙 보정, 전량 teacher 처리 및 공통 학습 serializer 연결. 전수 metadata 감사 463만 발화와 표본 806개 점검은 완료.
- By: tskim
