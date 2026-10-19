## [2026-10-19] task | Stage 4 본 학습 st4-m1 준비 — 번역 행 사이 띄어쓰기 버그 수정(--trans-lead-space), 2,500 step·영↔독 합성 가중치 1.0, 점검 통과

- Changed:
  - `vapasr/data/cst_dialogue.py`: `lead_space_row`, `CstDialogueDataset(trans_lead_space=)`
  - `experiments/stage3_train.py`·`cst_stage4_eval.py`·`cst_stage4_dryrun.py`, `slurm/submit-stage3-train.sh`: `--trans-lead-space`
  - raw [[2026-10-19-stage4-dialogue-translation]]: "본 학습 st4-m1 준비" 절, `lead_space_dry.sh`·`stage4_m1_check.sh`·`stage4-dry-ls-*.json`, 평가 큐 `TRANS_LEAD_SPACE=1`
  - [[output-stage4-prep-plan-20261019]] S6
- Reason: 사용자 "본 학습을 진행하고 싶은데" → 설정 제안 승인("그래 진행하자.")
- 결과:
  - st4-p1 평가 참조 단어의 8 %(대화)·10 %(합성)가 행 경계에서 붙어 있었다("hurry,If"). 행 첫 단어를 앞 공백 토큰으로 바꾸는 옵션으로 고쳤다(기본 끔).
  - 교차 언어 영어 반복 생성: δ10 강제 진행 창 19 → 31(ck2000 → final), 그 창 영어 오류 174 %. 대응은 짧은 학습·합성 가중치 하향(가설 검증은 st4-m1 평가로).
  - 점검: 타깃 dry-run 9 셋 각 500 창 문제 0, 학습 dry-run 불일치 0, 스모크 20 step PARITY OK
- Next: st4-m1 제출(사용자), 평가 큐 ck1500·ck2000·final(`TRANS_LEAD_SPACE=1`)
- By: tskim
