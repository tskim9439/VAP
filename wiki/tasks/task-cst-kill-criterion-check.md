---
type: task
status: open
owner: tskim
due: TBD
priority: p1
created: 2026-10-02
updated: 2026-10-02
summary: joint 모델 개발 전 관문 — VAP+2×strong SimulST 가 통제 세트 상호작용 지표를 거의 다 푸는지, oracle 격차 측정
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
  - '[[cst-bench]]'
---

# kill criterion 검증 (Phase 3 → 4 관문)

## 배경
- joint 모델(B5)을 만들 가치가 있는지 먼저 판단한다([[cst-bench]] §Kill criterion).

## 완료 조건
- [ ] B3 와 B4(oracle)의 조건별 격차 표(맞장구·끼어들기·빠른 전환·의미 정정)
- [ ] 판정:
  - 계속: 맞장구 오전환·끼어들기 지연·조기 commit 오류·방향 오류가 남고, oracle 로 크게 개선된다
  - 재검토: strong pipeline 이 거의 다 푼다
- [ ] 판정을 `wiki/decisions/` 에 기록한다

## 진행 기록
- 2026-10-02: 생성.
