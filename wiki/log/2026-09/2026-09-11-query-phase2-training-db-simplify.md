## [2026-09-11] query | Phase 2 학습 DB 보고서 단순화

- Changed: [[output-phase2-training-db]] — 이전 선정안과 A등급 선별 v2를 통합하고 우선 준비·확장·replay/평가·필수 규칙·실행 순서로 재구성.
- Reason: 사용자 요청에 따라 중복 표와 낡은 결론을 제거. 344.2 h의 라벨 합계와 보유 대화 수 차이, 검증 전 Switchboard, 부분 조각의 파일 보유 비율을 실제 학습량과 구분했다. 원본 대화 수와 TurnBench dev 역할도 정정.
- Next: 원본/조각/replay ID 대조와 split을 먼저 고정하고 우선 데이터 manifest를 집계. 정본 §5의 이전 B등급 후보·360 h 추정은 최신 선정 기준으로 후속 동기화 필요.
- By: tskim
