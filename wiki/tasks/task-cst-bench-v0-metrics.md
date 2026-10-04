---
type: task
status: doing
owner: tskim
due: TBD
priority: p0
created: 2026-10-02
updated: 2026-10-05
summary: CST-Bench v0 평가 도구 — revision-free commit 프로토콜, 번역 품질·지연·방향·끼어들기·Stop/Switch Latency 지표
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
  - '[[cst-bench]]'
---

# CST-Bench v0 지표·평가 도구 (Phase 2)

## 배경
- 모델 독립 평가 도구다. 무엇을, 언제, 어느 방향으로 냈는지만 입력으로 받는다([[cst-bench]]).

## 완료 조건
- [x] 시스템 출력 형식: 확정 조각(session·lang·t·text, 선택 turn_id) — `cstbench eval` (2026-10-05)
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
- 2026-10-05: `cst-bench/src/cstbench/evaluate.py` + `cstbench eval` 구현.
  - 방향별 BLEU·chrF(정규화: 소문자·구두점 제거·숫자 풀어쓰기), StreamLAAL, 턴 종료 후 지연(EndOffset), 빈 턴.
  - 턴 ID 가 없으면 방향별 최소 편집거리 재분할(mwerSegmenter 기준).
  - TAXI 640 턴 점검: 참조 그대로 BLEU 100·EndOffset 0, 원문 복사 BLEU 0.4/2.4. 테스트 10 개.
  - 끼어들기·Stop Latency 는 v1 범위에서 빠졌다([[decision-cst-paper-v1-scope]]).
  - 남은 것: COMET, switch latency, 턴 ID 없는 잘못된 방향 검출, 세션 bootstrap, WER·cpWER·DER, 지표 교란 검증(논문 부록 C).
