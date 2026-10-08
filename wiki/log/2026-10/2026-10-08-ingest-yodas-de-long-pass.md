## [2026-10-08] ingest | YODAS-Granary 독일어 30–40 s 처리 결과

- Changed:
  - 원자료: `raw/sources/experiments/2026-10-08-yodas-de-align-run/`(`align-summary-dur_long.json`, `long-pass.md`)
  - 위키: `wiki/decisions/decision-yodas-de-asr-selection.md`, `wiki/outputs/output-yodas-granary-de-transcript-check-20261008.md`
- Reason: 사용자가 제출한 30–40 s 처리 job 82624 가 끝났다.
- 결과:
  - 23 분 동안 shard 2,899 개를 처리했고 오류는 0 이었다.
  - 231.1 h 를 채택해, 1–30 s 와 합쳐 2,276.5 h 가 됐다.
  - 표본 검증 위반은 0 이었다.
- Next: 학습 스크립트(`semcommit_train.py`)에 독일어 셋 연결, 40 s 항목을 고려해 묶음 토큰 예산 확인
- By: tskim
