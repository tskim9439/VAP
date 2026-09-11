## [2026-09-11] query | 정본 계획에 turn 토큰·시퀀스 사례(§4.4)와 데이터 생성 파이프라인(§5.3) 추가

- Changed: `output-phase2-streaming-asr-diarization-plan.md` §4.4·§5.3 신설(교차 직렬화 유지), `output-phase2-final-plan.md` superseded 표시
- Reason: 사용자가 정본 계획을 기본으로 택하고 청크 내 화자 교차 직렬화를 선호, turn 토큰 제안은 채택하되 "데이터를 어떻게 만드나" 를 물음. VAD 세그먼트 → 채널별 정렬 → derive_events 확장(방해당한 종료 = EOT 추가) → 슬롯 → 혼합 → 직렬화 → QC 의 전자동 파이프라인과 데이터 종류별 손실 규칙(NIKL 준자연 대화를 KO turn 토큰 원천으로)을 정리
- Next: Q0 에서 derive_events 세그먼트 확장·TurnBench gold 정의 대조·serializer 왕복 테스트 구현
- By: tskim
