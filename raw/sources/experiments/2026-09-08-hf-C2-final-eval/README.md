# run C2 최종 평가 원자료 (2026-09-08/09)

- `run.json`: 학습 설정(job 66260 재개 기준). `eval/offline-*.json`: 학습 중 오프라인 sentinel(10/10/100, seed 1, δ=2) — checkpoint 별. `offline-0.json` = final.
- `eval/select-*.json`: select 표본(50/50/300, seed 7). `select-<step>` 은 δ=2, `select-d3/d4-21960` 은 final 의 δ=3/4.
- `eval/sweep-d<δ>-b<bias>-21960.json`: final, sentinel 표본 seed 7, δ·next_bias 스윕.
- `eval/fulldev-*.json`: 전체 dev(있으면).
각 json: {set: {bias0: {err, wer|cer_official, lat_p50/p90/p99, viol, viol_80ms, tok_per_chunk, ref_per_chunk, forced_frac, backlog_p99, flush_rounds_mean, tick_ms_p50/p99, matched, n, sec}, best, best_bias}}.
