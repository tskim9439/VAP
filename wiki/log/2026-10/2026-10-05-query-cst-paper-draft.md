## [2026-10-05] query | CST 논문 초안(ICML 양식)과 TAXI 오프라인 베이스라인

- Changed:
  - 원고: `paper/cst-s2tt-icml/`(본문 7 절·부록 A–H·표 19·그림 15·검증된 참고문헌 140, tectonic 컴파일 49 쪽)
  - 위키: `decisions/decision-cst-paper-v1-scope.md`, `outputs/output-paper-draft-cst-s2tt-icml-20261005.md`, `outputs/output-cst-taxi-offline-baselines-20261005.md`, `tasks/task-cst-bench-v0-metrics.md`, `tasks/task-cst-strong-baselines.md`
  - 원자료: `raw/sources/experiments/2026-10-05-cst-taxi-offline-baselines-mxc/`
- Reason: 사용자가 ICML 양식 논문 초안(빈 수치 칸은 공백, 실험 설계 포함, 모델은 ASR·2인 diarization 도 수행)과 그림 생성을 요청했다.
- 결과: 컴파일 성공, 미정의 참조·인용 없음. 본문이 약 12.9 쪽으로 8 쪽 제한을 약 4.9 쪽 넘는다. 리뷰 이슈 111 건(치명 3 · 주요 37) 중 다수 반영, 일부 보류.
- Next:
  - 본문 8 쪽으로 줄이기(긴 표·문단을 부록으로)
  - 보류된 리뷰 이슈 확인(워크플로 결과의 fixer 보고)
  - E0(수치 기록)·E2(채점기 확장)·E4(L1 렌더)
- By: tskim
