## [2026-10-04] task | CST-Bench 빌드 도구 공개용 패키지(cst-bench/)

- Changed:
  - `cst-bench/` 신규: 패키지·CLI·설정·체크섬·테스트·문서·Apache-2.0
  - 지움(패키지로 대체): `vapasr/cst/*`, `experiments/cst_build_taxi.py`, `experiments/cst_make_sessions.py`, `tests/test_cst_taxi.py`
  - 갱신: `wiki/tasks/task-cst-conversation-schema-parser.md`
- Reason: 사용자가 TAXI 세션 생성 과정을 나중에 오픈소스로 재현할 수 있게 따로 정리해 달라고 요청했다.
- 결과: 테스트 5 개 통과. 실제 TAXI 로 다시 만든 타임라인이 앞 빌드와 86/86 일치. verify 통과.
- Next:
  - 코드 라이선스(Apache-2.0 가정)와 공개 저장소 위치를 소유자가 결정
  - XDailyDialog·BConTrasT 리더, 겹침 사건 생성기, 평가 지표를 패키지에 추가
- By: tskim
