---
type: task
status: open
owner: tskim
due: 2026-10-30
priority: p1
created: 2026-09-04
updated: 2026-09-04
summary: USLM U2 — self-generated history 혼합·corruption 학습, gold/self WER·delay 격차 보고
sources:
  - [[output-interleaved-streaming-slm-architecture]]
  - [[decision-target-architecture]]
---

# USLM U2 — self-conditioned streaming

## 배경

U1 은 teacher forcing. 추론에서는 자기 오류 토큰이 다음 audio step 을 오염시킨다 — 격차를 재고 회복 능력을 만든다.

## 완료 조건

- [ ] gold history vs self-generated history 의 WER·delay 격차 보고
- [ ] scheduled sampling / corruption 주입 학습, 격차 축소 폭
- [ ] 20–60 s 창 학습 + 이전 텍스트 요약·KV carry
- [ ] 긴 대화(15 min) 추론 안정성: backlog·KV 메모리

## 진행 기록

- 2026-09-04: 생성 (IS-SLM 단일 주력 결정에 따라).
