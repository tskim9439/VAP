---
author_id: tskim
created: 2026-10-08
source_type: source-drop
---

# YODAS-Granary 독일어 전사 품질 점검 원자료 (mxc, 2026-10-08)

- 표본: `experiments/yodas_de_qwen_check.py`(seed 0) — 하위 묶음 4 개 × `asr_only`/`ast` 마다 parquet 10 개 × 25 발화, 1,983 발화(3.45 h)
- 전사: Qwen3-ASR-1.7B(언어 German, `results.jsonl`), Whisper-large-v2·v3(`experiments/yodas_de_whisper_check.py`, greedy, German, 30 s 초과는 long-form)
- 결과 원본: mxc `VAPKT-data/results/yodas-de-check-20261008/`(results.jsonl, whisper-large-v2/v3.jsonl, review.tsv — 전사 텍스트 포함, CC-BY-3.0)
- 이 폴더: `qwen-summary.json`(Granary↔Qwen WER 요약), `rules-analysis.txt`(교사 간 일치·선별 규칙별 유지율), `analyze_rules.py`
- 처리 속도(H200 1 장): Qwen3-ASR-1.7B 3.45 h 를 132 s(배치 32), Whisper-large 각 155–159 s(배치 16)
