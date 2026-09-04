<!-- generated: do not edit -->
# TODO

마지막 생성: 2026-09-04

`wiki/tasks/` 의 `open` / `doing` / `blocked` 태스크를 owner 별로 모은 대시보드.
직접 편집하지 않는다. 단계별 실행 체크리스트는 저장소 루트 `TODO.md` 를 본다.

## tskim

| 상태 | 우선순위 | 마감 | 태스크 |
|------|----------|------|--------|
| **doing** | **p0** | 2026-10-16 | [[task-uslm-u1-interleaved-asr]] — USLM U1 — Nemotron frozen + Qwen3-0.6B LoRA interleaved ASR(텍스트 스트림만), WER 상대 열화 ≤10% 관문 |
| **doing** | **p0** | 2026-10-22 | [[task-stage1-encoder-probing]] — frozen encoder 비교 실험 — CPC/WavLM/FastConformer/AuT/DualTurn + random floor, 교란 통제 |
| open | **p0** | 2026-11-27 | [[task-uslm-u3-multitask]] — USLM U3 — audio-clock 헤드(VAP/τ/VAD)+이벤트 토큰 멀티태스크, 50Hz 사이드 브랜치 하이브리드 ablation, encoder-only probe와 비교(H2) |
| open | p1 | 2026-09-17 | [[task-secure-english-corpora]] — otoSpeech·CANDOR 라이선스 동의 및 확보, SpokenWOZ 채널 구조 확인 |
| **doing** | p1 | 2026-09-18 | [[task-uslm-feasibility-u0]] — USLM U0(streamability+타당성) — 토큰율→M 예산, ForcedAligner 정렬·QC, interleaved target 생성기 |
| open | p1 | 2026-09-19 | [[task-checkpoint-retention-policy]] — /data4 여유 575G 상황에서 체크포인트 보존·정리 규칙과 실험 폴더 규약 수립 |
| open | p1 | 2026-09-24 | [[task-paper1-scoping]] — Paper 1 범위 확정과 아웃라인 — Stage 1+2 + 한국어 어노테이션 레이어 |
| open | p1 | 2026-10-29 | [[task-add-missing-baselines]] — VAD+threshold, cascade, DualTurn, JAL-Turn 등 초안에 빠진 baseline 4종 구축 |
| open | p1 | 2026-10-29 | [[task-latency-quality-curve]] — chunk 80/160/320/640ms의 latency-quality 곡선과 lookahead 회계 포함 보고 |
| open | p1 | 2026-10-30 | [[task-uslm-u2-self-conditioned]] — USLM U2 — self-generated history 혼합·corruption 학습, gold/self WER·delay 격차 보고 |
| open | p1 | 2026-12-10 | [[task-time-to-next-turn-survival-head]] — τ head를 discrete-time hazard로 구현하고 censoring 처리 및 EOT 확률 곡선 산출 |
| open | p1 | 2026-12-24 | [[task-korean-benchmark-design]] — 한국어 turn-taking 어노테이션 프로토콜 설계 — 샘플링, IAA, 배포 형식 |
| open | p2 | 2026-12-10 | [[task-event-label-heuristics-validation]] — 유도 이벤트 라벨 규칙을 사람이 라벨한 otoSpeech에 대해 검증하고 수용 기준 판정 |
| open | p2 | 2027-01-14 | [[task-korean-turn-cue-literature-review]] — 한국어 어미·prosody와 turn 완결성 관계의 문헌 근거 확보 |
| open | p2 | 2027-02-11 | [[task-bilingual-and-qwen-port]] — 최종 Nemotron–adapter–Qwen thinker backbone의 KO/EN temperature sampling과 언어별·잡음 조건 강건성 평가 |
