## [2026-09-24] query | Semantic commit v0.2 rack4 소규모 실행

- Changed: `wiki/outputs/output-semcommit-v0.2-rack4-small.md`(신규), `raw/sources/experiments/2026-09-24-semcommit-v0.2-rack4/`(등급 통계·평가 보고서·parity·블라인드 점검·실행 스크립트),
  `vapasr/hf/commit_metrics.py`(`pc_punct_after`·`pc_commit_counts` — LibriSpeech-PC 문장 끝 독립 참조), `experiments/semcommit_pc_eval.py`(신규), `tests/test_commit_metrics.py`.
  rack4: `/data4/tskim/semcommit/{labels/v0.2, runs/v0.2-r1, runs/v0.2-r2, eval/v0.2-r1, eval/v0.2-r2}`.
- Reason: 사용자가 semantic commit 계획(`raw/inbox/streaming_asr_semantic_commit_plan.md`)을 작은 데이터·작은 오픈 모델로 rack4 GPU 1 장에서 구현·시험하라고 요청.
  라벨(Qwen3-8B·EXAONE-3.5-7.8B, gpt-oss-20b 는 EN 판정 97 % WAIT 라 KO tie-break 진단으로만) → E2 에서 r1·r2 학습(parity OK) → 평가.
  ASR 가드레일 통과, SEM F1 KO 0.56 / EN 0.17–0.24. 병목은 교사 일치도(κ 0.12–0.21)·N 오판·Stage A 발화 끝 누락·TURN 길이 지름길.
- Next: mxc 본 실행 전에 큰 교사로 ls-test·ks-eval 후보의 κ·A 비율·블라인드 정밀도를 먼저 재고, Stage A 에 스트림 마지막 단어 후보 추가, N 정책 축소, 스트림 길이 다양화·KO 다문장 스트림.
- By: tskim
