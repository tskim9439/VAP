## [2026-10-24] query | `<EOT>` 없는 R1x 결과와 회의 시험 되돌림

- Changed: `raw/sources/experiments/2026-10-21-v3-attribution/`(README 27 절, `noeot_summary.py`, `noeot_diag.py`), `slurm/submit-q06-mt4.sh`(기본 `<EOT>` 있음, `--no-eot` 선택, low_p_hpc·MAX_RESTART 20), `slurm/submit-evals.sh`(mt4-v3·q17-mt4-v3, --partition), `slurm/submit-q17-chain.sh`(--dlg mt4), [[status]]
- Reason:
  - `<EOT>` 없는 시험의 평가(job 94869)가 끝났다.
  - 미리 정한 기준에서 dev 귀속은 통과했지만, 같은 언어 받아쓰기 가드를 크게 넘었다(한국어 대화 붕괴). 채택하지 않는다.
  - 사용자 "제출명령어를 low_p_hpc 로 돌아가도록 수정해줘" — 회의 시험을 low_p_hpc 로 내도록 고쳤다. 선점에 대비해 평가 노드 `--requeue`, 완료 표시 건너뛰기, 평균 임시 폴더도 고쳤다(커밋 408c242).
- Next:
  - 회의 + 4 슬롯(`<EOT>` 있음) 0.6B 시험을 제출한다.
  - 결과로 1.7B 대화 단계를 r1x 로 둘지 mt4 로 바꿀지 정한다.
- By: tskim
