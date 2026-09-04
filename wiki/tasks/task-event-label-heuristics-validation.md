---
type: task
status: open
owner: tskim
due: 2026-12-10
priority: p2
created: 2026-09-03
updated: 2026-09-03
summary: 유도 이벤트 라벨 규칙을 사람이 라벨한 otoSpeech에 대해 검증하고 수용 기준 판정
sources:
  - [[output-streaming-vap-research-plan]]
---

# 이벤트 라벨 휴리스틱 검증

## 배경

→ [[question-event-label-derivation-validity]]

AI Hub·CANDOR 에는 이벤트 라벨이 없어 휴리스틱으로 유도해야 한다.
**otoSpeech 는 사람이 라벨했으므로 정답지가 있다.**

## 완료 조건

- [ ] 휴리스틱 규칙을 otoSpeech 에 적용 — **TurnBench dev gold(EOT 1,904 / INT 347)로 즉시 시작 가능**: 현재 SHIFT 992 / INT 318 ([[output-vap-target-pipeline]])
- [ ] 이벤트별 Precision / Recall / F1 측정
- [ ] **혼동행렬 — 특히 BACKCHANNEL ↔ INTERRUPTION**
      (둘 다 겹쳐 시작하고, 차이는 "이후 floor 를 가져가는가" 라는 사후 정보)
- [ ] 타이밍 오차 분포
- [ ] **수용 기준 F1 ≥ 0.8** — 미달 이벤트는 auxiliary supervision 에서 제외하거나
      라벨 노이즈를 논문에 명시
- [ ] 한국어 어노테이션 서브셋으로 재검증 (backchannel 형태가 언어마다 다르다)

## 진행 기록

- 2026-09-03: 생성.
