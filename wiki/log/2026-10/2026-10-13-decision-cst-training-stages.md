## [2026-10-13] decision | CST 학습 단계와 화자 슬롯 형식 초안(proposed)

- Changed: [[decision-cst-training-stages-and-speaker-slots]] 신규(decision_status: proposed)
- Reason:
  - 사용자 질문: Stage 3(2 인 대화)에서 화자 귀속·ASR·AST 를 한 번에 할지, 교차 언어 대화 데이터를 어떻게 만들지, 화자 태그를 말한 순서로 할지 언어로 고정할지
  - 사용자가 초안 작성을 요청했다.
- 제안 내용:
  - Stage 1 ASR → 2 AST → 3 2 인 같은 언어 대화의 화자 귀속 + ASR(번역 데이터 일부 유지, 대화 번역 타깃 미리 생성) → 4 + 번역 → 5 교차 언어
  - 화자 슬롯은 언어로 고정(누가 먼저 말하든 선언한 언어면 그 슬롯, 같은 언어면 먼저 말한 사람이 A). 모델 안 화자 메모리를 두지 않는 Phase 2 정본과 맞다.
  - 교차 언어 학습 데이터: 이어 붙인 실제 발화(주력) + 한쪽 언어를 바꾼 실제 대화 + 합성 대화. 실제 교차 언어 대화는 평가 전용
- Next:
  - 2026-10-20 재검토: s4 평가와 Stage 3 원천 조사를 보고 확정한다.
  - 받아들여지면 output-staged-training-plan-20261001 의 단계 번호를 대체한다.
- By: tskim
