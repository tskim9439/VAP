## [2026-10-18] task | Stage 3 합성 2 인 대화(sim) 구현 — Stage 1·2 단일 화자 발화 + Phase 2 화자 재짝짓기, 실제 시간 구조·겹침, EOT 가림; v1 언어별 3,000 세션, v2 1 만 세션 생성 중

- Changed:
  - `vapasr/data/sim_dialogue.py`(새): Unit·화자 키 규칙·사람 id·말소리 구간·구 나누기·시간 구조·세션 합성·점검
  - `experiments/sim_build_dialogues.py`(새): 풀·시간 구조·병렬 합성·mono(+잡음·잔향) 캐시
  - `experiments/sim_preview.py`(새): 로컬 미리보기 페이지
  - `vapasr/data/cst_dialogue.py`: `meta.mask_eot` 면 `<EOT>` 결정 label −100
  - `experiments/stage3_train.py`: `--sim-data`, 셋 `sim_ko·sim_en·sim_de`
  - `slurm/submit-stage3-train.sh`: `--sim-data`
  - `tests/test_sim_dialogue.py`(9 개)
  - raw [[2026-10-18-stage3-sim-dialogue]]
  - [[output-stage3-prep-plan-20261014]]: "합성 2 인 대화" 절
- Reason:
  - 사용자 제안 "Stage 1 혹은 Stage 2 에서 활용한 DB 를 활용해서 … 서로 대화한 것처럼 만들어서 학습 … 일정 부분도 겹치도록"
  - "그래 합성 대화 구현을 시작해"
- 결과:
  - 실제 대비 분당 교대·겹침: 영어 7.4 · 4.1 %(실제 9.1 · 3.4 %), 한국어 11.6 · 6.2 %(10.7 · 5.3 %), 독일어 12.5 · 4.7 %(14.6 · 4.4 %)
  - 첫 판의 교대 부족(재개 비율을 실제 값으로 씀)을 고쳤다.
  - NSC 가 짝지어지지 않던 문제(역할 이름 화자)를 화자 단위 사람 id 로 고쳤다.
  - v1 언어별 3,000 세션(약 75 h), 실패 0·자기 겹침 0
  - 학습 스모크: 합성 셋 포함 점검 불일치 0, parity OK
- Next:
  - v2(언어별 1 만 세션) 생성이 끝나면 파일럿 p10 을 준비한다. p9 레시피 + 합성 v2, 사용자 제출.
  - 사용자 청취 확인을 받는다(미리보기 T5).
- By: tskim

## [2026-10-18] task | 합성 v2(언어별 1 만 세션) 생성·dry-run 통과, 파일럿 st3-p10-sim 준비

- Changed:
  - raw [[2026-10-18-stage3-sim-dialogue]]: v2 통계·dry-run·파일럿 절
  - [[output-stage3-prep-plan-20261014]]: 합성 절의 생성·파일럿
- Reason: v1(3,000 세션)은 창이 언어당 6–8 천이라 4,000 step 동안 약 20 번 반복된다. 반복을 줄이려고 늘렸다.
- 결과:
  - v2 실패 0, 자기 겹침 0, 82 GB
  - 창: 한국어 29,261 · 영어 22,128 · 독일어 28,286
  - dry-run 점검 불일치 0, 제출 명령 dry-run 확인
  - 평가 큐 3 개(GPU 2·5·0)가 체크포인트를 기다린다.
- Next: 사용자가 st3-p10-sim 을 제출한다. p9 와 같은 step 끼리 비교한다.
- By: tskim

## [2026-10-18] task | 파일럿 st3-p10-sim 제출(job 90667) — 학습 전 점검 통과, 비중 계획대로

- Changed: raw [[2026-10-18-stage3-sim-dialogue]] 의 파일럿 절·[[output-stage3-prep-plan-20261014]]
- Reason: 사용자 제출 job 90667(10:48 hpc-39·40)
- 결과:
  - 합성 셋 창 29,261 · 22,128 · 28,286, 점검 불일치 0
  - 비중: 실제 0.373 · 합성 0.231 · replay 0.397
  - 11:02 step 10
- Next: 평가 큐(GPU 2·5·0)가 ck2000·3000·final 을 잰다. p9 와 같은 step 끼리 비교한다.
- By: tskim
