---
type: task
status: open
owner: tskim
due: TBD
priority: p0
created: 2026-10-02
updated: 2026-10-02
summary: CST-Bench v0 평가 도구 — revision-free commit 프로토콜, 번역 품질·지연·방향·끼어들기·Stop/Switch Latency 지표
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
  - '[[cst-bench]]'
---

# CST-Bench v0 지표·평가 도구 (Phase 2)

## 배경
- 모델 독립 평가 도구다. 무엇을, 언제, 어느 방향으로 냈는지만 입력으로 받는다([[cst-bench]]).

## 완료 조건
- [ ] 시스템 출력 형식: 시각·방향·확정 여부가 붙은 토큰열
- [ ] 번역 품질(COMET·BLEU·chrF)은 확정 텍스트만 채점한다(revision-free)
- [ ] 지연:
  - AL·LAAL, 첫 토큰 지연
  - semantic commit 지연: 기존 창 없는 commit 지연 지표([[output-semcommit-v035-d8-eval]])를 확장
- [ ] 상호작용:
  - 방향 정확도, 잘못된 중단·전환
  - 놓친 끼어들기, 잘못된 방향 번역
  - Stop/Switch Latency
- [ ] 짝 bootstrap 신뢰구간
- [ ] Natural Set 과 Controlled Set 의 분리 집계

## 진행 기록
- 2026-10-02: 생성.
