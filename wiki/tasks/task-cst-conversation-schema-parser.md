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
- 2026-10-04: 공통 스키마 `vapasr/cst/schema.py`(→ 이후 `cst-bench/src/cstbench/schema.py`)(Session/Turn/Translation, JSONL 왕복)와 TAXI 파서 `vapasr/cst/taxi.py`, 빌드 스크립트 `experiments/cst_build_taxi.py`, 테스트 `tests/test_cst_taxi.py` 를 만들었다.
  - T5 TAXI 결과: 86 세션 · 697 DP/CL 턴 중 usable 640(DP 349 · CL 291) · 52.6 분 · 6,815 단어 · 세션당 usable 턴 중앙값 7 · 턴 길이 중앙값 4.0 s(p90 9.1)
  - 매니페스트: `/Volumes/Samsung_T5/VAPKT-DB/TAXI/cst/taxi-sessions.jsonl`, 16 kHz 리샘플(전화 대역 유지): `.../cst/wav16k/`
  - 특성: 버튼(DTMF)으로 턴을 나눠 겹침이 없고 원본 턴 간 시각도 없다 → `timing="turn_order_only"`. 재배포 금지 → `redistributable=False`
- 2026-10-04: 세션 연속 음성 생성 `vapasr/cst/timeline.py` + `experiments/cst_make_sessions.py`.
  - 출력: 세션마다 mono 혼합·화자별 2 채널·timeline.json(턴 시작·끝, 전사, 참조)
  - 턴 파일 앞뒤 무음(중앙값 앞 0.88 s · 뒤 0.44 s)을 에너지 기준으로 자른 뒤, 간격을 합성해 겹침 없이 이어 붙인다
  - 간격 판: natural(약 0.2 s) 0.74 h, mediated(1–3 s, 번역을 읽고 답하는 대기) 1.0 h. 86 세션, 세션 길이 중앙값 약 25–40 s
  - 무음 자르기는 에너지 기준이라 작은 소리가 잘릴 수 있다 → 강제 정렬 단어 시각으로 검증·대체 예정
- 남은 것: 강제 정렬 단어 시각, 겹침 사건 생성기(L1·L2), XDailyDialog·BConTrasT 파서
- 2026-10-04: **공개용 독립 패키지 `cst-bench/` 로 옮겼다.**
  - 구성:
    - `src/cstbench/{schema,timeline,audio,cli}.py`, `corpora/taxi.py`
    - `configs/taxi_{natural,mediated}.json`
    - `checksums/` · `tests/` · `docs/FORMAT.md` · README · Apache-2.0
  - 사내 코드(`vapasr`)에 의존하지 않는다. 앞의 `vapasr/cst`·`experiments/cst_*`·`tests/test_cst_taxi.py` 는 지우고 패키지로 대체했다.
  - CLI: `cstbench build-taxi`(taxi-manifest → prepare-audio → render 두 판), `cstbench verify`(타임라인 다이제스트는 정확히 일치해야 함, 음성 해시는 라이브러리 버전에 따라 다를 수 있어 따로 보고)
  - 재현 확인: 패키지로 다시 만든 86 세션 타임라인이 앞 빌드와 86/86 일치. 기준 체크섬을 `checksums/taxi-{natural,mediated}.json` 에 저장했다.
  - TAXI 데이터·파생 음성은 저장소에 넣지 않는다(재배포 금지). 출력: `/Volumes/Samsung_T5/VAPKT-DB/TAXI/cstbench-v0.1/`
