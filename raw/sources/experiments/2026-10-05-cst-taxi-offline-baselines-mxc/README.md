---
author_id: tskim
created: 2026-10-05
source_type: source-drop
---

# CST TAXI 오프라인 정답 분할 베이스라인 원자료 (mxc job 80130, 2026-10-05)

- 스크립트: `cst-bench/baselines/offline_oracle.py`, 채점 `cstbench eval`(`cst-bench/src/cstbench/evaluate.py`), 제출 `slurm/cst-taxi-offline-baselines.sbatch`(low_p_hpc, GPU 2 장)
- 입력: TAXI natural 세션(`cstbench-v0.1/taxi-natural/sessions.jsonl`, 86 세션 · 640 턴)
- 결과 원본: mxc `VAPKT-data/results/cst-taxi-baselines/offline-oracle-80130/`
  - hyp·segments·asr.jsonl 은 TAXI 전사·번역 텍스트를 담고 있어 이 저장소로 가져오지 않았다(재배포 금지).
- job 경과: 00:27 시작 → 결과 파일을 모두 쓴 뒤 00:37 선점 → 자동 requeue → 00:50 재시작, 끝난 단계는 건너뛰고 채점만 다시 함 → 00:51 DONE.
- 파일:
  - `report-<system>-t.json`: 이상적 시계(조각 방출 시각 = 정답 턴 끝)
  - `report-<system>-elapsed.json`: 측정한 계산 시간을 더한 시계(배치 16 의 평균을 턴마다 나눠 붙인 근사)
  - `run_info-*.json`: 모델 경로·설정. ASR WER 은 `logs-excerpt.txt` 에 있다. 재시작에서 run_info-asr.json 이 덮어써져 WER 항목이 빠졌고, 이후 코드에서 고쳤다.
  - `logs-excerpt.txt`: 단계 로그와 sbatch 출력, 턴 평균 WER
- 시스템: gold-mt(정답 전사 → Qwen3.8-27B), cascade(Qwen3-ASR-1.7B, 언어 지정 → Qwen3.8-27B), whisper(Whisper-large-v3 task=translate, 독→영만). 턴마다 한 번씩 번역, 대화 이력 없음.
