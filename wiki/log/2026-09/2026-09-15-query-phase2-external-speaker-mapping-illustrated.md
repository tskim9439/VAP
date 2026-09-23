## [2026-09-15] query | 외부 화자 재매핑 구조의 그림 해설

- Changed: [[output-phase2-external-speaker-mapping-illustrated]] 신규; [[output-phase2-external-speaker-mapping-plan]]에 해설 링크.
- Reason: 현재 제안을 그림과 쉬운 설명으로 보고서화해 달라는 요청. 전체 흐름·작업칸 재사용·겹침 crop·지연 EOT를 네 그림으로 정리하고 무음/단일/겹침 출력과 책임을 설명했다. v2의 lazy-free·R=6·EOT lane 결합은 별도 비교안으로 표시했다.
- Next: 외부 모듈의 정답 구간 시험 → local lane 자유실행 → 예측 구간 연동 → 지연 EOT 검증. 정본·registry·코드·원천 자료는 변경하지 않았으며 커밋·푸시·학습은 실행하지 않았다.
- By: tskim
