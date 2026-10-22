## [2026-10-22] task | V3 SLURM 평가 노드 job 94789 — R1x·R1x-a 평균 전체 평가, 남은 V3 단계

- Changed: `slurm/v3-eval-node.sbatch`, `raw/sources/experiments/2026-10-21-v3-attribution/v3_eval_task.sh`, 같은 폴더 README 17 절
- Reason: 사용자 "남은 평가를 SLURM 으로 수행해도 돼" — 1 노드 8 GPU job 이 작업 목록(`VAPKT-data/tmp/v3-eval-tasks-20261022.txt`, 27 줄)을 GPU 마다 차례로 돈다(사용자가 제출, job 94789). R1x·R1x-a 의 ck1500·ck2000·final 평균은 호스트에서 만들었다. R2m 은 같은 언어 가드 실패(같은 언어 교대 가중 1.5 도 핑퐁).
- Next: job 끝나면 R1x·R1x-a·R2m·평균 체크포인트를 한 표로 묶고 dev 선택·가드·TAXI 세션 짝 bootstrap 을 보고한다.
- By: tskim
