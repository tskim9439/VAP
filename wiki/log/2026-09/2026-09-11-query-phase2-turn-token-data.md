## [2026-09-11] query | turn 토큰·시퀀스 사례·데이터 생성 파이프라인을 별도 제안서로

- Changed: `wiki/outputs/output-phase2-turn-token-proposal.md` 생성. 정본 `output-phase2-streaming-asr-diarization-plan.md` 는 직전 커밋의 §4.4·§5.3 삽입을 되돌려 원상 복구. `output-phase2-final-plan.md` 는 superseded 유지, 링크 수정
- Reason: 사용자가 정본 계획을 기본으로 택하고 교차 직렬화를 선호했으며, turn 토큰 제안은 "정본에 바로 적용하지 말고 별도 보고서로" 요청. 제안서에 규약·사례 7 종·전자동 데이터 파이프라인(VAD → 채널별 정렬 → derive_events 확장 → 슬롯 → 혼합 → 직렬화 → QC)·데이터 종류별 손실 규칙·정본 반영 시 변경 지점을 정리
- Next: 사용자 검토 후 정본 §4.4·§5.3 반영 여부 결정
- By: tskim

## [2026-09-11] query | 제안서 개정 — 토큰 종류를 의미 라벨로, LLM 시드 → 증류 → 전량 → 사람 검증

- Changed: `output-phase2-turn-token-proposal.md` §3b·§3c 신설, 요약·반영 지점·불확실성 갱신
- Reason: 사용자 검토 — 라벨이 음향(타이밍)에만 의존하고 TurnBench 에 과도하게 맞춰졌다는 지적. 위치는 VAD, 종류는 의미(완결/미완결/맞장구)로 분리하고, 텍스트만으로 사후 문맥을 보는 LLM 시드 10 만 → 증류 분류기 → 전량 1 천만 경계 → 사람 κ ≥ 0.7 검증의 저비용 경로를 제안. 완결성 정확도·응답 기회 P/R 을 주 지표로, TurnBench 는 EN 외부 검증
- Next: 사용자 검토 → L1 프롬프트·층화 표본 설계
- By: tskim
