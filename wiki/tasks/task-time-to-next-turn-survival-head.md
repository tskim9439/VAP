---
type: task
status: open
owner: tskim
due: 2026-12-10
priority: p1
created: 2026-09-03
updated: 2026-09-03
summary: τ head를 discrete-time hazard로 구현하고 censoring 처리 및 EOT 확률 곡선 산출
sources:
  - [[output-streaming-vap-research-plan]]
---

# Time-to-next-turn hazard head

## 배경

→ [[turn-taking-objectives]] 2절.

초안의 bucket 분류를 **discrete-time hazard** 로 바꾼다. censoring 을 원리적으로
처리하고 `P(EOT before Δ)` 를 임의 시점에 계산할 수 있다.

## 완료 조건

- [ ] τ 대상 정의를 문서로 고정 — 화자 지정, overlap 처리, 동일 화자 재개 제외
- [ ] hazard head 구현: `h_k(t) = P(구간 k 에서 onset | k 이전 없음)`
- [ ] **right-censored 구간을 likelihood 에서 올바르게 제외**
- [ ] bucket 분류 방식과 성능 비교 (초안 방식이 더 나으면 그대로 간다)
- [ ] hazard 로부터 **−600 / −300 / 0 ms 시점의 P(EOT) 곡선** 산출
      → [[turn-taking-evaluation-protocol]]
- [ ] Next-Turn 의 "320 ms 이내 endpoint accuracy" 지표로 비교

## 진행 기록

- 2026-09-03: 생성.
