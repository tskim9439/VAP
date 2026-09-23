## [2026-09-17] query | Phase 2 독립 평가 계획

- Changed: [[output-phase2-independent-evaluation-plan]] 신규. 사용자 요청에 따라 기존 보고서·정본은 수정하지 않음.
- Reason: 전사와 화자 귀속의 혼합 페널티, 초기 번호 교환, 발화 경계·턴 구간의 정량 평가, 미학습 DB 기반 객관적 평가 요구. ORC/cp·DER·boundary·semantic EOT를 분리하고 프로젝트 전체 학습 계보의 노출 감사와 dev/locked 구성을 제안했다.
- Next: E0 모델 parity·노출 감사 → E1 전사 scorer와 순열 fixture → E2 화자/경계 → E3 턴 gold → 동일 dev 비교 후 locked 평가. 평가 코드 변경·데이터 수집·학습·외부 제출·커밋·푸시는 하지 않았다.
- By: tskim
