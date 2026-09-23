## [2026-09-19] query | 데이터 선별: NIKL 복구와 시간축·채널 후속 감사

- Changed: `wiki/outputs/output-data-selection-repair-v03.md`, `output-data-selection-review-v02.md`, `wiki/status.md`, `source-data-selection-human-review-20260919.md`; 복구·VAD·채널 감사 및 검수 축소 코드/테스트; 신규 원형 증거 묶음과 32개 청취 페이지.
- Reason: 사용자의 "차례대로 진행해" 요청에 따라 사람 검수 근거 범위의 NIKL 복구부터 순차 실행했다. NIKL 625개 재추론, VoxPopuli 177,422개 길이 규약, 다채널 3,657개를 검증했다. 기존 원음·전사·학습 입력은 유지했다.
- Next: 32개 추가 청취와 미검수 DB별 calibration. 기준이 확정된 DB부터 GPU 최대 4장으로 전량 screening 후 Phase 1/2 공통 serializer smoke. 학습/EOT 자동 승인은 하지 않음.
- By: tskim
