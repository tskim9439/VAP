---
author_id: tskim
created: 2026-10-08
source_type: source-drop
---

# 영·한·독 Stage 1 학습 준비 원자료 (mxc, 2026-10-08)

- `de-tn2-summary-train.json`, `de-tn2-summary-dev.json`: 독일어 정규화 `asr-tn-de-v1.1.0` 결과(`experiments/yodas_de_textnorm.py --tag tn2`). 출력 mxc `VAPKT-data/data/yodas-de-v1/tn2/`, 목록 `lists/de-words-{train,dev}-tn2.list`
- `de-tn1-summary-train.json`: v1.0.0(`tn1`) 결과 — 비라틴 문자를 지우기만 하던 판, 학습에 쓰지 않음
- `eval-yodas-de-dev-v1-coverage.json`: 독일어 평가셋 `VAPKT-data/data/evaluation/single-turn-yodas-de-dev-v1` 준비 기록(tn1 dev 목록으로 만들었고, tn2 dev 와 행·텍스트·토큰이 같음을 확인)
- `de-train-dryrun-20261008.json`, `de-train-dryrun-d4to10-20261008.json`: 학습 스크립트 CPU dry-run(영·한·독 일부 목록) — 셋 통계·제외·시퀀스 검사. 앞은 δ 1,2,3,4,6,8, 뒤는 δ 4–10
- `job-82738-startup.log`: 0.6B 학습 job 82738 로그 발췌(지연 확장 토큰, 인코더 context, 셋별 통계, step/epoch, 첫 손실)
- 1.7B job 82730 은 모델 로드 직후(17:16:59) 사용자 취소. 로그 mxc `VAPKT/logs/train/SA_SFT_FullDuplex-82730.out`
