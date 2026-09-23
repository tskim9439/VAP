## [2026-09-15] query | 외부 화자 재매핑을 허용한 Phase 2 대안

- Changed: [[output-phase2-external-speaker-mapping-plan]] 신규; [[output-phase2-dynamic-speaker-memory-plan]]에 조건부 후속 검토 링크.
- Reason: 장기 화자 ID를 외부에서 재매핑할 때 기존 상세안이 여전히 최선인지 묻는 요청. local lane·episode는 유지하고 내부 identity memory는 보류하는 대안을 분석했다. 겹침 crop의 한계, EOT 책임, 외부 ID 가용 지연을 구분했다.
- Next: 외부 모듈의 oracle 구간 상한 시험 → local lane ASR → 예측 구간 연동 → 모델 생성 EOT 유지 여부에 따른 통합. 정본·registry·코드 변경과 학습·커밋·푸시는 하지 않았다.
- By: tskim
