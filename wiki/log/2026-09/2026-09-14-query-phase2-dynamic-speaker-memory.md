## [2026-09-14] query | 전사 채널·동적 화자 메모리 상세 설계

- Changed: [[output-phase2-dynamic-speaker-memory-plan]] 신규; [[output-phase2-speaker-representation-comparison]] 후속 계획 링크.
- Reason: 세 번째 후보를 구체화해 달라는 요청. lane/episode/speaker를 분리하고 SEG_END·지연 EOT pointer·공유 matcher·동적 heads·순차 HF 학습·D0–D5 검증을 설계했다.
- Next: 대안 채택 여부와 별개로 D0의 프로토콜 fixture 및 기존 71631 실물 기반 D1/D2 검증. 정본·registry·코드 변경, 실제 학습·커밋·푸시는 하지 않았다.
- By: tskim
