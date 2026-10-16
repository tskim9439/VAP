## [2026-10-16] task | Stage 3 파일럿 p1b–p5 비교 — 언어 비율 KO 중심(p4)·soft EOT·replay 0.5 유지, 맞장구 규칙은 EOT 정의에 따라 결정 필요

- Changed:
  - [[output-stage3-prep-plan-20261014]]: "P5 파일럿 p1b–p5 비교" 절
  - [[2026-10-16-stage3-pilots]] 신규
  - `experiments/cst_stage3_eval.py`: `--backchannel`(맞장구 규칙 참조로 채점)
- Reason: 사용자가 파일럿 다섯 개를 apex 에 제출(job 88406–88410)했고, 모두 끝나 평가 큐가 대화·단일 화자 평가를 마쳤다.
- 결과(δ10 평균):
  - **cp:** s4 57.9 → p1b 33.1 · p2 31.9 · p3 31.4 · p4 31.4 · p5 34.2
  - **agn:** 17.1–17.6
  - **EOT hard:** 보류 오방출 0.17 → 0.28
  - **맞장구 규칙:** 같은 정의로 재면 정밀도 0.39 → 0.57, 재현율 0.75 → 0.66
  - **replay 0.25:** 퇴행 가장 큼
  - **단일 화자:** 모든 run 에서 약한 퇴행이 남는다.
- Next:
  - 사용자 결정: 맞장구 규칙(EOT 정의)과 본 학습 레시피(언어 비율 p4·soft·replay 0.5·step 수)
  - 단일 화자 퇴행을 줄일 방법(replay 비율↑·lr↓·step 수)
- By: tskim
