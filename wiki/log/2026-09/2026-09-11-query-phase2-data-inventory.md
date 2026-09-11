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

## [2026-09-11] decision | "최대 2 화자" 제약 제거

- Changed: `wiki/decisions/decision-multi-speaker-scope.md` 생성, `output-phase2-data-inventory.md` §2b 를 다화자 전량 사용으로 수정
- Reason: 사용자 결정. 회의 코퍼스를 자연 다화자 데이터로 쓰고 슬롯·헤드·평가를 K 화자로 일반화. 정본 계획 §1·§3·§4·§6·§8 개정 필요 항목을 결정 페이지에 표로 정리
- Next: 사용자 확인 후 정본 계획 개정, K(슬롯 상한) 결정
- By: tskim

## [2026-09-11] query | 확보 DB 추가 조사 — NIA24 134-1/134-2 발화 조각이 화자 채널 기원임을 검증

- Changed: `output-phase2-data-inventory.md` §5(추가 조사·정렬 등급·우선순위), 조사 로그·스크립트를 `raw/sources/experiments/2026-09-11-phase2-data-inventory/` 에 보관
- Reason: 사용자 요청(5 경로 탐색)과 "정렬은 화자별 독립 채널이 필요" 지적. 상호상관 검증으로 NIA24 조각이 한 채널과 r=1.000·다른 채널과 ≈0(겹침 포함)임을 확인 → 조각 단위 정렬과 2 채널 복원 가능. 실외 1,492 대화 완전(≈360 h), 실내는 발화 ~45 % 무작위 결손, 청소년은 미검증. 콜센터·CallHome·사내 8 kHz 는 시간축 없음
- Next: Q0 에서 134-2 완전성 검증, 002_Meeting JSON 구조, Switchboard 복원 검증
- By: tskim
