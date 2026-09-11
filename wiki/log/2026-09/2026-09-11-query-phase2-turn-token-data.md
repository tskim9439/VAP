## [2026-09-11] query | turn 토큰·시퀀스 사례·데이터 생성 파이프라인을 별도 제안서로

- Changed: `wiki/outputs/output-phase2-turn-token-proposal.md` 생성. 정본 `output-phase2-streaming-asr-diarization-plan.md` 는 직전 커밋의 §4.4·§5.3 삽입을 되돌려 원상 복구. `output-phase2-final-plan.md` 는 superseded 유지, 링크 수정
- Reason: 사용자가 정본 계획을 기본으로 택하고 교차 직렬화를 선호했으며, turn 토큰 제안은 "정본에 바로 적용하지 말고 별도 보고서로" 요청. 제안서에 규약·사례 7 종·전자동 데이터 파이프라인(VAD → 채널별 정렬 → derive_events 확장 → 슬롯 → 혼합 → 직렬화 → QC)·데이터 종류별 손실 규칙·정본 반영 시 변경 지점을 정리
- Next: 사용자 검토 후 정본 §4.4·§5.3 반영 여부 결정
- By: tskim
