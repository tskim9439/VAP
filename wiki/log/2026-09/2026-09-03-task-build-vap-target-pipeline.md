## [2026-09-03] task | VAP target 파이프라인 완료 — Phase 0 종료

- Changed: `vapasr/` 패키지 신설 (`data/conversation.py`, `vad.py`, `corpora.py`, `targets.py`, `dataset.py`), `experiments/build_targets.py`,
  `recompute_events.py`, `test_dataset.py`, `raw/sources/experiments/2026-09-03-target-pipeline/`(stats 3 + QC 7),
  `wiki/outputs/output-vap-target-pipeline.md`(신규), `task-build-vap-target-pipeline` → done, `task-event-label-heuristics-validation`
  갱신, `todo.md`·`TODO.md`·`status.md`·`index.md`·`log.md`. `.env` `AIHUB_ADULT_ROOT`, 서버 ASCII 심볼릭 링크(`adult-*`).
  서버: `/data3/tskim/manifests/{aihub-vs02,otoSpeech,turnbench-dev}/` (npz + manifest + qc).
- Reason: Phase 0 마지막 p0. 코퍼스 3종을 공통 `Conversation` 으로 읽어 채널 VAD@50 Hz 를 저장하고, VAP 256-class(원 VAP
  코드 재사용)·hazard τ(censoring)·이벤트(SHIFT/HOLD/INT/BC)를 로드 시 파생한다. 12.5 Hz 는 bins 2/5/8/10. 합성 VAD 테스트와
  QC 이미지 검수로 이벤트 규칙 결함 2건(가짜 SHIFT, terminal overlap→INT)과 판정창(1→3 s)을 고쳤고, AI Hub 화자↔채널
  매핑을 파일별 자동 판정(3/186 뒤바뀜)했다. 결과 156 h, 20 s 창 55,139개, `WindowDataset` 50/12.5 Hz 동작 확인.
  운영 교훈: bg 래퍼에 한글·괄호 경로를 넘기면 깨진다 → ASCII 링크 사용; matplotlib 누락으로 QC 1회 실패.
  추가: TS_01.실내_5 수신·해제 후 빌드 — 757 파일 **196.6 h**(zip 크기 추정 95 h 의 2×, wav 압축률 때문). 서버 총 보유 ≈ 360 h.
- Next: Phase 1. [[task-compute-budget-and-feature-cache]](otoSpeech 리샘플 300 ms/item 해소), [[task-event-label-heuristics-validation]]
  (TurnBench dev gold 로 즉시 가능), [[task-stage1-encoder-probing]]. AI Hub TS_01.실내_5 수신 대기.
- By: tskim
