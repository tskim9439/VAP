## [2026-09-04] query | 이중 프레임율+RNN-T 안 기각, IS-SLM 단일 주력 확정

- Changed: `decision-target-architecture`(확정: A 기각, B′ 단일 주력, 대조군은 외부), `output-model-architecture-proposal` → superseded,
  `output-unified-slm-architecture-plan`·`output-streaming-vap-research-plan` 갱신 표식, `task-stage2-…`·`task-stage3-…` → 폐기(done, U3 병합),
  신설 `task-uslm-u1-interleaved-asr`(p0)·`task-uslm-u2-self-conditioned`(p1)·`task-uslm-u3-multitask`(p0), `TODO.md` Phase 2 절 교체,
  `README.md` §2·§3·§5 갱신, `todo.md`·`index.md`·`status.md`.
- Reason: 사용자 결정 — "이중 프레임율 + RNN-T 는 완전 기각, IS-SLM 을 주력으로". Paper 1 = Stage 1 표현 비교 + U0–U3. fallback 이 없으므로
  U0·U1 을 가장 먼저 짧게 돌려 WER 관문(≤10 %)을 조기 판정한다. 50 Hz 이점(Stage 1 실측)은 U3 하이브리드 ablation 으로 흡수.
  "통합의 가치" 는 같은 encoder 의 encoder-only probe 대 IS-SLM 상태 위 헤드로 판정.
- Next: Stage 1 매트릭스 마무리(재채점) → 결과 페이지·H1 판정 → U0 착수.
- By: tskim
