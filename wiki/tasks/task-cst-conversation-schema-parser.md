---
type: task
status: open
owner: tskim
due: TBD
priority: p1
created: 2026-10-02
updated: 2026-10-02
summary: 원본 대화 코퍼스 → 공통 conversation schema(세션·턴·화자·언어·시각·전사·번역) 파서와 턴 재구성
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
  - '[[task-cst-data-access]]'
---

# 공통 conversation schema 와 파서 (Phase 0–1)

## 배경
- 코퍼스마다 형식이 다르다. [[cst-bench]] 의 모든 단계(합성·평가·기준선)는 한 schema 를 읽어야 한다.

## 완료 조건
- [ ] schema 정의:
  - session_id, turn 순서, speaker_id, speaker 언어
  - 턴 시작/끝 시각(있으면), 턴 음성 경로(세션 또는 화자 채널 + 구간)
  - 전사, 참조 번역(있으면)
- [ ] 첫 확보 코퍼스 파서 + 턴 재구성 + 화자·언어 매핑 + 번역 정규화
- [ ] 16 kHz 턴 단위 음성 정규화, 단위 테스트

## 진행 기록
- 2026-10-02: 생성.
