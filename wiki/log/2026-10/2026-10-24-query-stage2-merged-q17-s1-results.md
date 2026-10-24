## [2026-10-24] query | 합친 Stage 2 시험·1.7B Stage 1 평가 결과와 진행 점검

- Changed: `raw/sources/experiments/2026-10-23-q06-st2-merged/`(README 결과 절, `cmp_runs.py` — `cmp_s4_s3.py` 를 run 이름 인자로), `raw/sources/experiments/2026-10-22-q17-chain/`(README 진행·Stage 1 평가 절, `cmp_s1.py`), [[status]]
- Reason:
  - 사용자 "현재 진행 상황을 알려줘".
  - 끝나 있던 평가 둘을 미리 정한 방법으로 비교했다: 합친 Stage 2(job 94851), 1.7B Stage 1(job 94840).
  - 합친 Stage 2: 받아쓰기는 나빠진 곳이 없다. 번역은 FLEURS 영→한 δ4 −1.44(유의) 하나로 기준상 불합격이다.
  - 1.7B Stage 1: 한·독은 모든 δ 에서 0.4–0.7 pt 낫고, 영어는 짧은 δ 에서 조금 나쁘다.
  - 1.7B AST 3 차는 노드 OOM 으로 두 번 죽고 이어 가 곧 끝난다.
- Next:
  - 사용자가 합친 Stage 2 채택 여부를 정한다.
  - 1.7B 대화 단계(94813·94814)와 평가(94841)를 지켜본다.
  - 회의 + 4 슬롯 시험과 회의 통째 평가는 사용자 제출을 기다린다.
- By: tskim
