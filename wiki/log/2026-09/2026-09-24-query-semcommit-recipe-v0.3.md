## [2026-09-24] query | semcommit 라벨링 레시피 v0.3 (v0.3.3 까지)

- Changed: `wiki/outputs/output-semcommit-recipe-v0.3.md`(신규), `wiki/outputs/output-semcommit-gold-v1.md`(레시피 안내), `raw/sources/experiments/2026-09-24-semcommit-recipe-v0.3/`(신규),
  `raw/sources/experiments/2026-09-24-semcommit-gold-v1/README.md`(점수·관문 사용법), 코드 `vapasr/data/semcommit_llm.py`·`experiments/semcommit_{teacher,gold}.py`·`experiments/semcommit_teacher_gate.sh`·tests(커밋 78cf789 · 41ce71c · 711f6bc).
  rack4: `/data4/tskim/semcommit/labels/v0.3{,.1,.2,.3}`, `runs/{v0.3-r3,v0.3.3-r5}`, `eval/{v0.3-r3,v0.3.3-r5}`, `gold/v1/{README.md,scores/}`, 로컬 T5 `semcommit-gold-v1/{README.md,scores/}`.
- Reason: 사용자 요청(본 학습 전에 라벨링 레시피 수정 — 후보·N 축소·보정 등급). 후보 추가(마지막 단어·구간 끝·구두점 문장 끝), 판정자 확률 기반 등급을 골드 dev 절반에서 가지마다 정밀도 0.90 으로 조정,
  LLM N 폐기. 조정 중 전체 floor 의 함정(그 외 가지가 영어 학습 A 의 25 %, 정밀도 ~0.3)을 가지별 floor 로, 한국어 머리 대답어를 대답어 가지로 고쳤다.
  v0.3 라벨 모델(r3)이 한국어에서 과확정(발화 끝 A 는 많고 음성은 없음)해 골드로 검증한 규칙 음성(머리 대답어·한국어 연결어미, 정밀도 0.99+)을 더했다(v0.3.3, r5).
  결과: 교사 A 재현율 0.14–0.31 → 0.70–0.82(정밀도 0.89–0.99), 모델 최고 F1 영어 낭독 0.35 → 0.80·대화체 0.11 → 0.71·한국어 0.75 → 0.84 / 0.71 → 0.80, ASR 기준 이하. 구두점 숨긴 재주석으로 순환성 없음 확인.
  스트림 길이 섞기는 미룸(근거였던 길이 지름길은 <TURN_END> 의 것).
- Next: mxc 교사 관문(`semcommit_teacher_gate.sh`, 기본 v0.3)으로 큰 교사를 골라 임계값 재조정 → 본 라벨링(전 언어·대화체). 본 학습 코퍼스별 전사 구두점 유무 확인(없으면 큰 교사의 그 외 가지나 구두점 복원),
  영어 대화체 오판(후보 아닌 자리)용 규칙 탐색, 한국어 축약 -ㄴ데·다시 말하기 규칙 후보를 골드로 측정, 언어별 bias 운영점.
- By: tskim
