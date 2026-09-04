---
type: task
status: done
owner: tskim
due: 2026-11-26
priority: p1
created: 2026-09-03
updated: 2026-09-03
summary: (폐기) 모델 A 전용 — H2 검증은 IS-SLM U3 의 encoder-only probe 대 LLM 상태 비교로
sources:
  - [[output-streaming-vap-research-plan]]
---

# Stage 3 — Linguistic state fusion

## 배경

→ [[acoustic-linguistic-fusion]]. 가설 H2 의 직접 검증.

[[source-turn-taking-related-work-2026]] 의 JAL-Turn 이 유사한 fusion 을 이미 했으므로,
**차별점은 hold/shift 이진이 아니라 future projection 으로 확장** 하는 데 있다.

## 완료 조건

- [ ] `z_t = F(h_audio, h_ling)` 구현 — RNNT predictor state 사용
- [ ] fusion 방식 비교: concat / cross-attention / gating
- [ ] acoustic-only 대비 이득 측정 — **특히 mid-turn pause vs true EOT 구분**에서
- [ ] **exposure bias ablation**: gold transcript 조건 vs 모델 자체 가설 조건.
      격차가 크면 H2 의 실용 가치가 제한됨을 논문에 명시
- [ ] 한국어에서의 이득이 영어보다 큰지 확인 → [[korean-turn-taking-cues]] 가설 검증
- [ ] cascade baseline 대비 latency·정확도 우위 정량화

## 진행 기록

- 2026-09-04: **폐기.** RNNT predictor state 융합은 기각된 A 의 요소. H2 는 [[task-uslm-u3-multitask]] 의 'encoder-only 헤드 vs IS-SLM 상태' 비교로 검증.

- 2026-09-03: 생성.
