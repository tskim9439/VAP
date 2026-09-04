---
type: task
status: done
owner: tskim
due: 2026-11-12
priority: p1
created: 2026-09-03
updated: 2026-09-03
summary: (폐기) 모델 A 전용 — IS-SLM U3 로 병합
sources:
  - [[output-streaming-vap-research-plan]]
---

# Stage 2 — Multitask VAP + WER 가드레일

## 배경

RNNT 를 그대로 두고 VAP head 만 추가한다. encoder freeze 로 시작해 상위 4~8층만
점진 unfreeze. **turn 성능이 올라도 ASR 이 망가지면 통합 모델의 존재 이유가 없다.**

## 완료 조건

- [ ] VAP head 추가, encoder·RNNT freeze 상태로 학습
- [ ] 상위 4층 / 8층 unfreeze 조건 비교
- [ ] **매 조건마다 WER(EN)·CER(KO) 를 함께 보고** — 회귀 허용 한계를 사전에 정한다
      (예: 상대 5% 이내)
- [ ] 손실 균형: 고정 λ vs **uncertainty weighting** vs GradNorm 비교
      → [[turn-taking-objectives]]
- [ ] class imbalance 대응 — balanced sampling 또는 focal loss 효과 확인
- [ ] 이벤트 단위 평가로 보고 (256-class CE 값만으로 판단 금지)

## 진행 기록

- 2026-09-04: **폐기.** 이중 프레임율+RNN-T 안 기각([[decision-target-architecture]]). WER 가드레일·손실 균형 항목은 [[task-uslm-u3-multitask]] 로.

- 2026-09-03: 생성.
