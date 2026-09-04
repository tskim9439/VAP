---
type: task
status: open
owner: tskim
due: 2026-11-27
priority: p0
created: 2026-09-04
updated: 2026-09-04
summary: USLM U3 — audio-clock 헤드(VAP/τ/VAD)+이벤트 토큰 멀티태스크, 50Hz 사이드 브랜치 하이브리드 ablation, encoder-only probe와 비교(H2)
sources:
  - [[output-interleaved-streaming-slm-architecture]]
  - [[decision-target-architecture]]
---

# USLM U3 — conversational multi-task

## 배경

IS-SLM 의 존재 이유가 판정되는 단계. Stage 1 실측(50 Hz > 12.5 Hz)에 따라 하이브리드가 첫 ablation 이다.

## 완료 조건

- [ ] audio-position hidden 위 VAP256 / τ hazard / VAD 헤드 + `<SPEECH_ONSET|ENDPOINT>` 토큰 학습 (encoder·LLM freeze → adapter·상위 층 점진 개방)
- [ ] 손실 균형(uncertainty weighting / GradNorm), **WER/CER 회귀 가드레일 ≤ 5 % 상대**
- [ ] **하이브리드 ablation**: 50 Hz CPC 사이드 브랜치 → 헤드 vs 순수 12.5 Hz vs LLM 25 Hz
- [ ] **H2 판정**: 같은 encoder 의 encoder-only probe(Stage 1) vs IS-SLM 상태 위 헤드 — 언어 상태가 turn 에 실제 이득인가; 한국어에서 더 큰가
- [ ] TurnBench dev 공식 규약(lookahead 접기 포함) + AI Hub 실내/실외 CE
- [ ] 총 RTF·메모리로 "한 모델" 의 시스템 이점 정량화

## 진행 기록

- 2026-09-04: 생성 (IS-SLM 단일 주력 결정에 따라).
