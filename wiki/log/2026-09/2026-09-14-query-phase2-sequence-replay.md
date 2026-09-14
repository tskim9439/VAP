## [2026-09-14] query | 실제 AMI 시퀀스 재구성 및 표시

- Changed: [[output-phase2-sequence-replay]], [[output-phase2-eot-review-framework]] 실행 예시 링크, `experiments/p2_replay_sequence.py`, `raw/sources/experiments/2026-09-14-phase2-sequence-replay/` 신규 산출물.
- Reason: 실제 데이터 시퀀스를 만들어 보여 달라는 요청. 이전 로컬 원본 경로는 없으므로 보존된 38.4초 PCM·단어/이벤트 기록을 재직렬화해 480블록 ID·화자별 전사 복원을 검증했다. 자동 EOT 5개는 미검수로 유지했다.
- Next: 독립 청취·누락 감사와 라벨 승인. 새 녹음 표본·모델 추론·검수 완료 결과로 해석하지 않는다.
- By: tskim
