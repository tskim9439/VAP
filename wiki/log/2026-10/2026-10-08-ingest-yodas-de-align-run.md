## [2026-10-08] ingest | YODAS-Granary 독일어 본 처리 결과와 30–40 s 처리 추가

- Changed:
  - 코드: `experiments/yodas_de_align.py`(`--select-flags`·`--max-dur`, 코드 해시를 시작 시 한 번 계산), `experiments/semcommit_build_speechlm_candidates.py`(`build_candidate` 의 길이 상한을 `max_duration_s` 인자로, 기본 30 s 그대로), `slurm/yodas-de-align.sbatch`, `slurm/submit-yodas-de-align.sh`(`--long`)
  - 원자료: `raw/sources/experiments/2026-10-08-yodas-de-align-run/`
  - 위키: `wiki/decisions/decision-yodas-de-asr-selection.md`, `wiki/outputs/output-yodas-granary-de-transcript-check-20261008.md`
- Reason:
  - 본 처리 job 82542 가 끝났다(1–30 s, 2,045.4 h 채택, shard 오류 0).
  - 사용자가 30 s 넘는 발화도 넣고 싶다고 했다. 다른 사유가 없는 30–40 s 행 1,394.5 h 를 같은 기준으로 처리하는 long pass 를 추가했다.
- 결과:
  - long pass 시험 77 발화 중 14 채택, 정렬 실패 0, 검증 위반 0
  - 본 처리 무작위 12 shard 검증 위반 0
- Next: 사용자 제출 `bash slurm/submit-yodas-de-align.sh --long --nodes 4 --partition hpc` → 채택 시간 확인 → 학습 스크립트에 독일어 셋 연결
- By: tskim
