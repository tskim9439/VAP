---
type: task
status: doing
owner: tskim
due: TBD
priority: p1
created: 2026-10-02
updated: 2026-10-04
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
- 2026-10-04: 공통 스키마 `vapasr/cst/schema.py`(Session/Turn/Translation, JSONL 왕복)와 TAXI 파서 `vapasr/cst/taxi.py`, 빌드 스크립트 `experiments/cst_build_taxi.py`, 테스트 `tests/test_cst_taxi.py` 를 만들었다.
  - T5 TAXI 결과: 86 세션 · 697 DP/CL 턴 중 usable 640(DP 349 · CL 291) · 52.6 분 · 6,815 단어 · 세션당 usable 턴 중앙값 7 · 턴 길이 중앙값 4.0 s(p90 9.1)
  - 매니페스트: `/Volumes/Samsung_T5/VAPKT-DB/TAXI/cst/taxi-sessions.jsonl`, 16 kHz 리샘플(전화 대역 유지): `.../cst/wav16k/`
  - 특성: 버튼(DTMF)으로 턴을 나눠 겹침이 없고 원본 턴 간 시각도 없다 → `timing="turn_order_only"`. 재배포 금지 → `redistributable=False`
  - 남은 것: 강제 정렬로 단어 시각 만들기, 세션 타임라인(턴 간격 합성) 생성기 연결, XDailyDialog·BConTrasT 파서
