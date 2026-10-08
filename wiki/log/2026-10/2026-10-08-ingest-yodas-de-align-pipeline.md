## [2026-10-08] ingest | YODAS-Granary 독일어 전처리 파이프라인과 시험 실행

- Changed:
  - 코드: `experiments/yodas_de_prepare.py`, `experiments/yodas_de_align.py`, `slurm/yodas-de-align.sbatch`, `slurm/submit-yodas-de-align.sh`
  - 원자료: `raw/sources/experiments/2026-10-08-yodas-de-align-smoke/`
  - 위키: `wiki/decisions/decision-yodas-de-asr-selection.md`(신규), `wiki/outputs/output-yodas-granary-de-transcript-check-20261008.md`
- Reason: 사용자가 전사 점검의 권장안(원 전사↔Qwen WER ≤ 3 % 선별, Qwen 원출력 타깃, Whisper 재전사 제외)에 동의했다. 그래서 전량 전처리 파이프라인을 만들고 시험했다.
- 결과:
  - 사전 점검 후보는 train 6,315.7 h, dev 23.0 h 이다.
  - 시험 shard 6 개에서 정렬 실패가 8 건이었는데, 80 ms 경계 보정 뒤 1 건만 남았다.
  - 출력 검증(tar, 시각, tokenizer 재분할, 로더)은 위반 0 건이었다.
- Next: 사용자 SLURM 제출(5 노드 hpc) → `align-summary.json` 확인 → 학습 스크립트에 독일어 셋 연결
- By: tskim
