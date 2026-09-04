---
type: task
status: open
owner: tskim
due: 2026-10-29
priority: p1
created: 2026-09-03
updated: 2026-09-03
summary: chunk 80/160/320/640ms의 latency-quality 곡선과 lookahead 회계 포함 보고
sources:
  - [[output-streaming-vap-research-plan]]
---

# Latency-quality 곡선

## 배경

→ [[streaming-causality-and-latency-budget]]

초안도 "80/160/320 ms latency-quality curve 는 꼭 보여줘야 한다" 고 했다. 옳다.
다만 **lookahead 를 합산한 실제 latency** 로 그려야 의미가 있다.

## 완료 조건

- [ ] Nemotron chunk 5단계(80/160/320/560/1120 ms)별 VAP 성능 측정
- [ ] 각 지점의 latency 를 `frame_period/2 + lookahead + compute` 로 계산해 표기
- [ ] 임계값을 쓸며 **latency-recall 궤적**을 그린다 (단일 operating point 금지)
- [ ] 같은 그림에 VAP(368 ms) 와 사람(−151 ms) 기준선 표기
- [ ] chunk 별 **한국어 CER 도 함께** 보고 — 작은 chunk 에서 ASR 이 얼마나 나빠지는가
- [ ] RTF·GPU memory 동반 보고

## 진행 기록

- 2026-09-03: 생성.
