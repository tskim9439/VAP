## [2026-10-20] task | 회의 번역 타깃 생성 제출(apex 93807 → 수리 93808), V2 평가·점검 스크립트

- Changed:
  - raw `2026-10-20-v2-multispeaker/`: `meet_targets_watch.sh`(수리가 끝나면 타깃 묶기), `v2_eval_extra.sh`(TAXI 통째·긴 창·회의·3–4 인 합성·회의 번역), `v2_check.sh`(V2 dry-run·스모크·평가 경로)
  - `experiments/cst_stage3_eval.py`·`cst_stage4_eval.py`: split 없이 창 전부(합성 평가 셋), 모델 슬롯이 창 화자 상한보다 적으면(V1 기준선) 모델 슬롯 수로 선언·디코드·채점
  - `experiments/cst_attr_errors.py`: `--split` 선택
- Reason: 사용자 제출(회의 원문 단위 `stage4-dlg-units-meet-v1` → 생성 93807, 수리 93808 afterany, 작업 164 = 82 shard × 영→독·영→한)
- 결과: 진행 중. 3–4 인 합성 평가 셋(held-out 재료, 언어마다 300 세션) 영·한 끝(실패 0, 겹침 6 %)
- Next: 수리 끝 → 타깃 묶기(자동) → V2 dry-run·스모크(`v2_check.sh MEET_TR=1`) → 제출 명령
- By: tskim
