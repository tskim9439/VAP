## [2026-09-11] query | Phase 2 정본에 다화자·ONSET/EOT·최신 데이터 계약 통합

- Changed: [[output-phase2-streaming-asr-diarization-plan]] 전반 개정; [[output-phase2-block-and-turn-label-spec]]를 이전 검토안으로 표시.
- Reason: 사용자의 정본 수정 요청. 최대 2화자 제한 해제와 ONSET/EOT 우선·의미 라벨 Stage 3 이월 결정을 통합하고 무음·단독·겹침 블록, K슬롯·미등장 화자, 행동 EOT 예측/확정/서비스 정책, 손실·마스크·평가를 명시했다. 추가 DB 조사도 확보·복원 후보·검증 완료를 구분해 반영했다.
- Next: Q0에서 K·event 규칙·P/C 모드·회의 관문 동결, 원본/조각 노출 감사, serializer와 32창 검증. 모델 코드·실제 라벨 생성·서버 job은 이번 변경에 포함하지 않음.
- By: tskim
