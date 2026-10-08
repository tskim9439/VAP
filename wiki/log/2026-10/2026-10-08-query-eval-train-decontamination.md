## [2026-10-08] query | 평가-학습 데이터 겹침 방지 결정

- Changed: `wiki/decisions/decision-eval-train-decontamination.md`(신규), `wiki/outputs/output-eval-set-plan-asr-simulst-20261007.md`(확보 결과·겹침 링크)
- Reason: 사용자가 받은 Europarl-ST(영↔독)를 확인하다가 영→독 test 연설의 일부가 백본 학습에 쓴 VoxPopuli 영어와 같은 연설임을 확인했다(같은 날짜 8-gram 일치 28.1 %, 날짜를 바꾼 대조군 0 %). 사용자가 결정 페이지로 남기라고 요청했다.
- 결정: VoxPopuli 는 학습 허용하되 평가셋과 겹치는 문장 제거, Europarl-ST 는 학습 제외, 영→독 평가는 겹침 없는 부분집합, 새 학습 코퍼스는 학습 전 8-gram 대조.
- Next: VoxPopuli 겹침 제외 목록과 Europarl-ST 영→독 부분집합 생성(요청 시), YODAS-Granary 독일어 시간 집계·대조
- By: tskim
