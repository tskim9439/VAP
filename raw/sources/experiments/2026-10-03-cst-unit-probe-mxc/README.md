---
author_id: tskim
created: 2026-10-03
source_type: source-drop
---

# CST 번역 단위 실험 원자료 (mxc, 2026-10-02~03)

- 스크립트: `experiments/cst_translation_unit_probe.py`(sample·run·analyze·simul·score)
- 결과 원본: mxc `VAPKT-data/results/cst-unit-probe-v1/`
- 실험 구성:
  - streams-v2: 언어별 300 스트림
  - 문맥 정렬: Qwen3.8-27B · EXAONE-4.0, 새 표본 seed 1
  - simul3: 정책 utt_end·SEM_END·MU, 언어별 150 개, 참조 = EXAONE 오프라인 번역
- 파일:
  - `unit-analyze-qwen-exaone-seed1.txt`: <SEM_END> 대 MU 겹침·단위 길이·대기
  - `simul3-score-and-examples.txt`: 정책별 BLEU·chrF + 원문·번역·MU 확정 과정 예시
  - `simul_fail.py`: 붕괴 분류(다른 문자·원문 복사·반복)와 붕괴 제외 재채점
  - `unit_latency.py`: 정책별 오라클 첫 토큰·LAAL
