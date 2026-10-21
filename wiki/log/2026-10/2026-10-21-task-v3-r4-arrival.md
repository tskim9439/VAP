## [2026-10-21] task | V3 R4 — 교차 언어 슬롯 먼저 말한 순서 구현, R2 bias check

- Changed: `vapasr/data/cst_dialogue.py`(cst_head_text_xl·xl_order), `experiments/stage3_train.py`·`slurm/submit-stage3-train.sh`(--xl-slot-order), `experiments/cst_stage3_eval.py`·`cst_stage4_eval.py`(score4 xl_perm)·`cst_stage4_taxi.py`(map_slots)·`cst_attr_turns.py`(창별 슬롯 언어), `raw/sources/experiments/2026-10-21-v3-attribution/`(taxi_v3_probe.py·onset_probe.sh·v3_check.sh r4·README 9–10 절), `tests/test_xl_arrival.py`, [[output-v3-attribution-plan-20261021]], [[status]]
- Reason: 사용자 "온셋 분류하고 R4 구현도 진행해줘." — R4 = R2 + 교차 언어 창 슬롯을 먼저 말한 순서로(prefix 는 두 언어만). R2 디코드 bias 는 사전 규칙대로 check 반쪽에서 확인했다(`wi=0.5` 가 태그를 하나도 바꾸지 않아 채택 안 함).
- Next: mxc 반영·시험, `v3_check.sh r4`(dry-run·스모크·평가 경로), R4 제출 명령(사용자 제출)과 평가 큐, 온셋 분류(R2 ck1500·V1 ck2000) 결과 정리, V3 평가 큐 완료 뒤 dev 로 체크포인트 선택·회귀 가드.
- By: tskim
