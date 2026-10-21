## [2026-10-22] task | V3 R2x·R4x 결과 — 같은 언어 회귀 해소, TAXI 이득 일부 반납, arrival 효과 유의하지 않음

- Changed: `raw/sources/experiments/2026-10-21-v3-attribution/`(README 14 절, 집계 파일 `v3-p1*/`·`v1-m2a-ck2000/`·`onset/`·`bias/`, `v3_summary.py`), [[output-v3-attribution-plan-20261021]], [[status]]
- Reason: R2x(94525)·R4x(94526) 학습·평가 큐 8 개와 R1·R2 OOM 재실행 완료. 같은 언어 가드(태그 수 포함), 사전 규칙 체크포인트 선택, TAXI 세션 짝 bootstrap 을 정리했다.
- Next: R2x ck2000·R4x ck1500·final 온셋 분류(진행 중), 다음 방향(중간 강도의 같은 언어 교대 가중, 번역 뒤 슬롯 선택(RESUME) 대응, 전화 합성 셋 C2 결합) 사용자 결정.
- By: tskim
