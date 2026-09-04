## [2026-09-04] task | 특징 캐시 완료(447 GB), Stage 1 매트릭스 시작

- Changed: `task-compute-budget-and-feature-cache` → done, `output-feature-cache-and-compute-budget` 최종 표(stable), `experiments/show_cache_stats.py`(신규),
  `todo.md`·`TODO.md`·`status.md`. 서버: `/data3/tskim/features/` 7 인코더 × 816 대화 완비(fbank 추가, tb-172 보충), `stage1-matrix` bg 작업 시작.
- Reason: 캐시 작업 3건(A/B/C) 완료. 실측 RTF·peak·용량으로 표를 갱신했다. 사용자 질문("캐시를 돌리는 이유")에 답함 — Stage 1 은 인코더를
  freeze 한 공정 비교(H1)이며 인코더 출력은 고정값이라 한 번(≈17–26 GPU h)만 계산하면 probe 매트릭스(조건당 ≈3 분)를 수십 번 돌릴 수 있다.
  `run_stage1.sh fbank cpc nemotron-c0 qwen-aut-causal qwen-aut-cc1s wavlm-base wavlm-large` (각 고유율 + 공통 12.5 Hz, 4 epoch) 시작.
- Next: 매트릭스 결과 → 고정 FP 격자(0.045/0.06/0.08/0.10) recall 표 → H1 판정 초안. 학습 데이터 EN-only 조건 추가 검토.
- By: tskim
