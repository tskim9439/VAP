<!-- generated: do not edit -->
# TODO

마지막 생성: 2026-09-03

`wiki/tasks/` 의 `open` / `doing` / `blocked` 태스크를 owner 별로 모은 대시보드.
직접 편집하지 않는다. 단계별 실행 체크리스트는 저장소 루트 `TODO.md` 를 본다.

## tskim

| 상태 | 우선순위 | 마감 | 태스크 |
|------|----------|------|--------|
| open | p1 | 2026-09-17 | [[task-secure-english-corpora]] — otoSpeech·CANDOR 확보, SpokenWOZ 채널 확인 |
| open | p1 | 2026-09-19 | [[task-checkpoint-retention-policy]] — /data4 575G 상황의 체크포인트 보존 규칙 |
| open | p1 | 2026-09-24 | [[task-paper1-scoping]] — Paper 1 범위 확정과 아웃라인 |
| **doing** | p1 | 2026-09-18 | [[task-uslm-feasibility-u0]] — USLM U0: 토큰율 + 정렬 파이프라인 (GPU 거의 불필요) |
| **doing** | **p0** | 2026-09-25 | [[task-uslm-u05-adapter-bridge]] — U0.5 Nemotron→thinker adapter bridge test (WER 관문) |
| open | **p0** | 2026-10-16 | [[task-uslm-u1-interleaved-asr]] — U1 interleaved ASR, **WER 관문 ≤10 %** |
| open | p1 | 2026-10-30 | [[task-uslm-u2-self-conditioned]] — U2 gold/self 격차 |
| open | **p0** | 2026-11-27 | [[task-uslm-u3-multitask]] — U3 헤드 + 하이브리드 ablation + H2 |
| **doing** | **p0** | 2026-10-22 | [[task-stage1-encoder-probing]] — **매트릭스 실행 중** (7 인코더 × 2 조건, 2026-09-04 11:49 시작) |
| open | p1 | 2026-10-29 | [[task-add-missing-baselines]] — VAD/cascade/DualTurn/JAL-Turn |
| open | p1 | 2026-10-29 | [[task-latency-quality-curve]] — chunk별 latency-quality 곡선 |
| open | p1 | 2026-12-10 | [[task-time-to-next-turn-survival-head]] — τ hazard head |
| open | p2 | 2026-12-10 | [[task-event-label-heuristics-validation]] — 유도 라벨 검증 |
| open | p1 | 2026-12-24 | [[task-korean-benchmark-design]] — 한국어 어노테이션 프로토콜 |
| open | p2 | 2027-01-14 | [[task-korean-turn-cue-literature-review]] — 한국어 turn 단서 문헌 |
| open | p1 | 2027-01-28 | [[task-qwen-aut-causal-adaptation]] — Qwen AuT 적응 연구 (Qwen 포팅의 전제) |
| open | p2 | 2027-02-11 | [[task-bilingual-and-qwen-port]] — bilingual 학습 + Qwen 포팅 |

**합계**: open 15 / doing 3 / blocked 0 / done 8 (폐기 2 포함)

## 관문

**Phase 0 완료.** 다음 관문은 [[task-stage1-encoder-probing]] (p0, 10-22) — 선행: 특징 캐시, 코퍼스 확보.
causality 감사는 완료되어 [[decision-asr-backbone]] 이 수정되었다 (Paper 1 = Nemotron 단일).
