## [2026-09-11] query | Phase 2 사용 데이터 목록 (mxc 실측)

- Changed: `wiki/outputs/output-phase2-data-inventory.md` 생성, 실측 로그 `raw/sources/experiments/2026-09-11-phase2-data-inventory/mxc-inventory.txt`; 제 Phase 2 페이지 4 종(실행 요약·비판·turn 토큰 제안서·최종안)은 superseded 표시
- Reason: 사용자 결정 — 정본 계획만 따르고 이번 단계 데이터를 먼저 리스트업. 정본 §5.1 이 "실측 전" 이라 서버에서 규모·형식·시각 정보를 쟀다: 71631 TS_01.실내_5 757 대화 196.6 h(E2 노출)·VS_02 186 대화 51.7 h(dev), otoSpeech 420 대화 104.9 h(미노출, actor id), TurnBench dev 38, Switchboard 파일명에 시각 있음(복원 가능), CallHome 시각 없음, CANDOR 없음
- Next: Q0 확인 항목 5 건, 사용자 결정(71631 추가 반입, CANDOR)
- By: tskim

## [2026-09-11] query | 영어 회의 코퍼스 후보 추가 (AMI·ICSI·CHiME-6·NOTSOFAR-1·DiPCo)

- Changed: `output-phase2-data-inventory.md` §2b
- Reason: 사용자 요청. 모두 3 명 이상 회의라 2 명 활동 창 선택(A) 또는 헤드셋 2 채널 합산(B) 방식으로만 쓰고 turn 손실은 마스크. AMI 는 서버에 발화 단위 사본 있음
- Next: Q0 에서 라이선스 원문·시각 정보 확인, AMI 원본 회의 파일 확보 검토
- By: tskim
