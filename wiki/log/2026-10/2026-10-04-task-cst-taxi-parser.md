## [2026-10-04] task | CST 공통 스키마와 TAXI 파서

- Changed:
  - 코드: `vapasr/cst/{__init__,schema,taxi}.py`, `experiments/cst_build_taxi.py`, `tests/test_cst_taxi.py`
  - 위키: `wiki/tasks/task-cst-conversation-schema-parser.md`
- Reason: 사용자가 BAS TAXI 를 T5 에 받고 파서 진행을 요청했다.
- 결과: 86 세션 · usable 640 턴(DP 349 · CL 291) · 52.6 분, 16 kHz 리샘플본을 T5 에 생성. 겹침·턴 간 시각 없음, 재배포 금지.
- Next:
  - 단어 시각 강제 정렬
  - 타임라인 생성기
  - XDailyDialog·BConTrasT 파서
  - mxc 업로드(사용자)
- By: tskim
