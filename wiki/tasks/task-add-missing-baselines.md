---
type: task
status: open
owner: tskim
due: 2026-10-29
priority: p1
created: 2026-09-03
updated: 2026-09-03
summary: VAD+threshold, cascade, DualTurn, JAL-Turn 등 초안에 빠진 baseline 4종 구축
sources:
  - [[output-streaming-vap-research-plan]]
---

# 누락 baseline 구축

## 배경

초안의 비교 대상이 부족하다. 리뷰어가 반드시 요구할 baseline 이다.
특히 초안은 cascade 를 **측정 없이** "latency 가 커진다" 며 기각했다.

## 완료 조건

- [ ] **VAD + silence threshold** — `turnbench/baselines/rms_vad` 재사용. 산업 표준. 이걸 못 이기면 연구 의미가 없다
- [ ] **cascade (ASR → 텍스트 turn 모델)** — 실제 구현해 **latency 와 정확도를 실측**.
      joint 방식이 번 것을 정량화한다 → [[acoustic-linguistic-fusion]]
- [ ] **DualTurn** — `turnbench/baselines/dualturn` 구현 재사용 ([[source-turnbench]]); 논문 F1 0.633 은 별도 지표
- [ ] **JAL-Turn** 재현 또는 인용 — hold/shift 이진 분류 대비 우위 논증
- [ ] 모든 baseline 을 [[turn-taking-evaluation-protocol]] 동일 규약으로 평가
- [ ] 사람 −151 ms 기준선을 그림에 표기

## 진행 기록

- 2026-09-03: 생성. DualTurn 은 [[source-turn-taking-related-work-2026]] 참조.
