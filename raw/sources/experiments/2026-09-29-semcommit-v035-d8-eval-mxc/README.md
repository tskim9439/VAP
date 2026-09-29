---
author_id: tskim
created: 2026-09-29
source_type: source-drop
---

# semcommit v035-snap0929-d8 평가 원자료 (mxc, 2026-09-29)

모델 `/soundai/Model/VAPASR/semcommit-v035-snap0929-d8/final`(job 76601, init `/soundai/Model/VAPASR/hf-E2/final`)의 평가 결과 중 작은 파일만 옮겼다.
큰 파일(스트림 jsonl, 발화별 predictions)은 mxc 에 그대로 있다. 모든 실행은 mxc 로그인 호스트 GPU 0–2(직접 실행) 또는 CPU.

## 1. ASR 전체셋 (single-turn v1 규약, `experiments/single-turn-asr-evaluation.md`)

- 결과 폴더: `/soundai/users/tskim/VAPKT-data/results/single-turn-semcommit-v035-d8-v1`(δ 2,4,6,8, GPU 0·1), `/soundai/users/tskim/VAPKT-data/results/single-turn-E2-final-v1`(δ 2,4,6, GPU 2)
- 실행: 각 폴더의 `run_single_turn_conda.sh` = `experiments/run_single_turn_asr_mxc.sh` 에서 `PY=` 한 줄만 conda env python 으로 바꾼 사본
  (`/tmp/sa_tskim-vapasr-env-local` 가 깨짐 — numpy `PyCapsule_Import could not import module "datetime"`). 인자: `<ckpt> <out> <gpus> 128 0 <deltas>`.
- `single-turn-v035-d8-summary.json`, `single-turn-E2-final-summary.json`: 각 폴더의 `summary.json`(complete=true, 44,984 / 33,738 예측).
- `st_compare.py`: 세트 × δ 표. `st_bootstrap.py` → `single-turn-bootstrap-E2-vs-v035-d8.txt`: 같은 발화 짝 bootstrap(2,000 회, LibriSpeech 화자 군집, Kspon 발화 단위).

## 2. commit 평가 (골드 v1 스트림 599 개)

- 디코드: `run_gold_eval.sh`(88220f7 판 `experiments/semcommit_eval.py`, bias −2..2, batch 32) → `/soundai/users/tskim/VAPKT-data/runs/semcommit/eval/v035-snap0929-d8/{set}-d{δ}.{json,streams.jsonl}`
  - words: `/soundai/users/tskim/VAPKT-data/data/semcommit/gold/v1/words-{set}.jsonl`
  - labels: `/soundai/users/tskim/VAPKT-data/data/semcommit-work/gate/speechlm-all19-g1-punctcoord-v035-20260926/labels-{set}.jsonl`
    = **v0.3.5 교사(g1 punctcoord)가 골드 스트림에 단 라벨**이지 골드 주석이 아니다.
- 골드 기준 채점: `gold_score.py`(= `semcommit_gold.py score-eval` 과 같은 `commit_metrics.gold_commit_counts`·`finalize_gold_commits`, + sha1 dev/test 절반) → `gold-v1-score.json`
  (키 `{set}-d{δ}` → `bias=b|all|dev|test`). 골드: `/soundai/users/tskim/VAPKT-data/data/semcommit/gold/v1/gold-{set}.jsonl`.
- 교사 라벨 기준(참고): `eval_summary.py` → `teacher-label-summary-rows.json`(88220f7 채점);
  `rescore_latency.sh`(5103556 판 `--score-only`, 창 없는 commit·display·after_text 지연) → `.../latency/{set}-d{δ}.json`, `latency_summary.py` → `teacher-label-latency-summary-rows.json`.
- `teacher-v035-vs-gold.txt`: 게이트 폴더의 `gate-{set}.json` 카운트(v0.3.5 교사 A/B/N vs 골드, 전체 스트림 — dev 절반은 교사 임계값 튜닝에 쓰였다).
- `train-parity.json`: 학습 종료 재로드 parity(`/soundai/Model/VAPASR/semcommit-v035-snap0929-d8/parity.json`).
