## [2026-10-21] task | V3 R1·R2 평가 종합 — 온셋 분류, 같은 언어 회귀(교대 태그 과잉)

- Changed: `raw/sources/experiments/2026-10-21-v3-attribution/`(README 11–12 절, `v3_eval_rerun.sh`, `v3_check.sh` r2x·r4x), [[output-v3-attribution-plan-20261021]], [[status]]
- Reason: 평가 큐 8 개 완료, 온셋 분류(R2 ck1500·V1 ck2000) 완료. 가드 점검에서 같은 언어 대화 회귀를 찾았다. 현재 코드로 V1 을 다시 재 400/400 창이 같음을 확인해 디코드 코드 원인을 배제했다.
- Next: OOM 으로 깨진 평가 5 단계 재실행 완료 뒤 표 갱신, R4 스모크·평가 경로 점검 완료, 사용자 결정(A: R2x·R4x / B: 원안 R4)에 따라 제출 명령, 제출 뒤 평가 큐.
- By: tskim
