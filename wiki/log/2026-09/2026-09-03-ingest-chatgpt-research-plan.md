## [2026-09-03] ingest | ChatGPT Streaming VAP 연구 계획 초안

- Changed: `raw/inbox/ChatGPT_Research_Plan.md` → `raw/sources/` 로 분류 이동.
  신규 페이지 33개 — `wiki/sources/` 7, `wiki/concepts/` 7, `wiki/questions/` 6,
  `wiki/decisions/` 2, `wiki/outputs/` 1, `wiki/tasks/` 17.
  `wiki/overview.md`, `wiki/status.md` 갱신. 파생 파일 3종 재생성.
- Reason: 사용자가 ChatGPT 로 작성한 streaming ASR + VAP 통합 연구 초안을 제공하고
  개선·계획 수립·태스크 도출을 요청했다. 초안이 인용한 8개 자료를 웹으로 검증한 결과
  **전부 실재**했으나, 계획을 수정해야 하는 사실 4건을 발견했다:
  (1) DualTurn(arXiv 2603.08216)이 VAP 를 weighted F1 0.633 vs 0.389 로 앞섰는데
  초안에 누락 — H1 의 경쟁 가설이자 필수 baseline.
  (2) Muse Voice Transcribe 는 **closed weights, API 전용** — backbone 후보에서 제외.
  (3) AI Hub 는 내국인 한정 + 재배포 제약 — "Korean TurnBench" 공개 배포 불가.
  (4) Qwen3 AuT 의 causality/lookahead 가 미문서화 — 검증 없이는 latency 비교가 무효.
  추가로 encoder lookahead 회계, Stage 1 교란 변수 통제, τ 의 생존분석 정식화,
  손실 균형과 WER 가드레일, 누락 baseline 3종을 계획에 반영했다.
- Next: Phase 0 의 p0 태스크 4건이 나머지를 막고 있다 —
  AI Hub 실물 검증, encoder causality 감사, VAP baseline 재현, target 파이프라인.
  볼트 운영으로는 `.llm-wiki-local/user.yaml` 의 member_id 사용자 확인,
  원격 저장소 연결, 초기 커밋이 남았다.
- By: tskim
