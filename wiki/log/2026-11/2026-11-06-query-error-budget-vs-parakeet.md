## [2026-11-06] query | 오류 예산 — 유형별로 얼마나 내려야 Parakeet + N3D 를 넘나

- Changed:
  - `raw/sources/experiments/2026-11-04-agn/gap_diag.py`(`--norm whisper` — 표준 채점과 같은 Whisper 정규화, 문장 단위), `raw/sources/experiments/2026-11-04-agn/err_budget.py`(신규 — 유형별 cp 몫·누적 시나리오·손익분기)
  - [[output-agn-reduction-20261104]] 10 절(신규), 7 절 혼자 말 주장 정정, summary
- Reason: 사용자 "우리 모델의 오류 유형을 분석하고 … 각 유형의 오류율을 어느 정도까지 내릴 수 있다면 Multitalker parakeet 보다 앞설 수 있는지 분석해봐"
- 발견: Whisper 정규화로 유형 합이 표준 cpWER 와 맞는다(23.51 / 29.78, Parakeet 15.90 / 20.52). 겹친 말을 Parakeet 수준으로 받아써도 2.6 / 4.8 pt 뒤진다. 혼자 말도 2.0 / 3.2 %p 뒤진다(7 절의 "같다"는 정규화 탓).
- Next: E15b 결과가 나오면 같은 셈으로 유형별 위치를 잰다. 혼자 말 대치(원거리) 몫을 줄일 길을 따로 찾는다.
- By: tskim
