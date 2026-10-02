---
type: task
status: open
owner: tskim
due: TBD
priority: p1
created: 2026-10-02
updated: 2026-10-02
summary: Controlled Interaction Set 합성기 — Normal·Early Turn·Interruption·Backchannel·Barge-in 타임라인, semantic chunk 기준 난이도
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
  - '[[cst-bench]]'
---

# 통제 상호작용 타임라인 생성기 (Phase 1)

## 배경
- 실제 대화의 내용·화자·턴 의존성은 유지하고 타이밍만 바꿔 상호작용 조건을 통제한다([[cst-bench]]).
- 끼어들기 시점은 semantic chunk 경계를 기준으로 고른다. Easy·Medium·Hard 의 세 난이도다.
- chunk 경계 후보로 `<SEM_END>` 라벨([[output-semcommit-recipe-v0.3]])을 재사용할 수 있는지 확인한다.

## 완료 조건
- [ ] 다섯 유형의 타임라인 생성:
  - Normal, Early Turn, Interruption
  - Backchannel: KO/EN 맞장구 목록
  - Explicit Barge-in: 문맥 조건 LLM + TTS. 판정 LLM 규칙상 EXAONE-4.0·Qwen3.8-27B 만 쓴다
- [ ] 겹침 구간 혼합 규칙(채널별 / 혼합), 사건 라벨(시작 시각·유형·난이도) 출력
- [ ] 생성 표본 청취 점검

## 진행 기록
- 2026-10-02: 생성.
