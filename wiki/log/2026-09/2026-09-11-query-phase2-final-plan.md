## [2026-09-11] query | Phase 2 최종 계획안(권고)

- Changed: `wiki/outputs/output-phase2-final-plan.md` 생성, 비판·실행 요약 페이지에 링크
- Reason: 사용자가 비판을 반영한 단일 최종 계획안을 요청. 정본 계획의 계약·관문·평가 규율 + 비판의 제안(청크 내 화자별 그룹 직렬화, 텍스트 전용 완결성 사전학습·LLM 약라벨, 슬롯 메모리, 슬롯 조건 오디오 토큰 대안, ASR 강화 트랙 A, 자연형 overlap 합성, E2 레시피, turn 헤드 δ=2, MVP) + 실행 요약의 일정·데이터를 한 문서로
- Next: 사용자 결정 5 건(§13) → Q0 착수(09-15)
- By: tskim

## [2026-09-11] query | 최종 계획안 §4 확장 — 사례별 블록 구성과 Muse 식 turn 토큰

- Changed: `output-phase2-final-plan.md` §4 전면 개정(토큰 6 종, 블록 문법, 시각 규칙, 무음·단일 화자·교대·맞장구·끼어들기·동시 시작·flush 사례, 학습 규칙, 디코드 상태 기계), §3·§5·§7·§9·§11 정합
- Reason: 사용자 지적 — 무음·단일 화자 등 사례별 블록 구성과 학습 방법이 부족하고, 화자별 `<end_of_turn>` 류 turn 라벨(Muse 처럼)이 필요
- Next: serializer 단위 테스트를 §4.4 사례로 작성(Q0)
- By: tskim
