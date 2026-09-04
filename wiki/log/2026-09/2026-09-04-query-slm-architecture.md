## [2026-09-04] query | 통합 SLM 구조 — 합산(v0) 대 Interleaved(IS-SLM) 검토, IS-SLM 채택

- Changed: `output-unified-slm-architecture-plan`(신규, 합산 v0 + 12항목 평가), `output-interleaved-streaming-slm-architecture`(외부 보고서, 말미에 '검토와 통합' 절 추가),
  `decision-target-architecture`(신규 → B′ 채택으로 갱신), `task-uslm-feasibility-u0`(신규, M 예산·interleaved 생성기로 조정), `index.md`·`todo.md`·`TODO.md`·`status.md`.
- Reason: 사용자가 RNN-T 독립 전사 대신 통합 SLM 을 목표로 제시(합산 융합 아이디어) → v0 계획·평가 작성. 이어 사용자가 IS-SLM 보고서를
  제시하며 더 적합하다고 판단 → 대조 검토. 동의: 합산은 KV 가 주는 정보와 중복이라 gated residual ablation 으로 격하, `<NEXT_AUDIO>` 가변
  방출이 토큰율 상한을 해소, dense 는 병렬 헤드. 보고서 보완: (1) 12.5 Hz audio-clock 위 turn 헤드는 Stage 1 실측(50 Hz 우위, INT 1.3 s)과
  충돌 → 50 Hz 사이드 브랜치 하이브리드, (2) 겹침 발화 텍스트 직렬화 규약, (3) tick 당 (2+M) forward 비용 → joint chunk token 기본,
  (4) 초기화 경로(Nemotron adapter / causal AuT + Qwen3-ASR thinker), (5) U1 WER 관문 ≤10 %.
- Next: [[task-uslm-feasibility-u0]] 착수(토큰율 → M, ForcedAligner 정렬, interleaved target 생성기). Stage 1 매트릭스 완료 후 H1 판정과 함께 하이브리드 필요성 확정.
- By: tskim
