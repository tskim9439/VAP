---
title: 결정 — Phase 2 평가 계획 v2 채택(축 분리·held-out 셋·TurnBench 합의 트랙·D1c 부터 split·meeteval 설치)
summary: 2026-09-17 사용자 결정. 평가는 인식(화자 무관)·화자 귀속·턴 구간·지연 4 축으로 분리하고, 객관 held-out 은 NIKL 2020·TurnBench dev·CHiME-6/DiPCo 로 시작한다. TurnBench 턴 채점은 3 트랙 합의 구간만, 7 코퍼스 세션 split 은 D1c 부터 적용, 서버 env 에 meeteval 설치 허용.
type: decision
created: 2026-09-17
related: [output-phase2-eval-plan, output-phase2-d1-lane-eval]
---

# 결정: Phase 2 평가 계획 v2

- **채택**: [[output-phase2-eval-plan]] 의 4 축 지표와 held-out 셋. lane 번호 기준 CER/WER 는 폐기(화자 오배정·초반 순서 어긋남이 전사 점수에 전가되는 결함).
- **TurnBench 어노테이터 트랙**: 턴 채점(ONSET/EOT/턴 교대/backchannel)은 3 트랙(a·b·c) 합의 구간만 쓴다(구간 IoU ≥ 0.5 로 b·c 에 모두 존재하는 a 구간). 전사·화자 축은 트랙 a 전체.
- **7 코퍼스 세션 split**: D1c 학습부터 제외한다. D1·D1b 는 전량 학습이라 seen 으로만 보고.
- **meeteval**: 서버 conda env 에 설치 허용(0.4.3 설치, 2026-09-17). cpWER·ORC-WER·tcpWER 대조 열로 쓴다.
- 근거: 2026-09-17 D1 평가에서 lane 기반 지표의 결함과 seen 편향 확인([[output-phase2-d1-lane-eval]]).
