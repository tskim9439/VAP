---
type: task
status: done
owner: tskim
due: 2026-09-17
priority: p0
created: 2026-09-03
updated: 2026-09-03
summary: Nemotron·Qwen3 AuT의 실효 lookahead를 절단 실험으로 측정 — 모든 모델링의 선행 조건
sources:
  - [[output-streaming-vap-research-plan]]
---

# Encoder causality 및 lookahead 감사

## 배경

→ [[streaming-causality-and-latency-budget]], [[question-encoder-lookahead-and-causality]]

**이 태스크의 결과가 [[decision-asr-backbone]] 을 뒤집을 수 있다.**
lookahead 를 모르면 latency 비교가 무효이고, offline 모드로 특징을 뽑으면
미래가 새어 들어가 모든 VAP 결과가 무의미해진다.

## 완료 조건

- [x] 절단 실험 스크립트 작성 — `experiments/causality_audit.py` (특징 단위 절단, fp32, rel tol 1e-3)
- [x] Nemotron chunk 설정별 실효 lookahead 표 — ≤80/160/320/480/880 ms
- [x] Qwen3 AuT — sdpa 경로 비인과(마스크 미호출 버그), 블록 모드 1 s: 0–800 ms 평균 420. **causal 아님**
- [x] 원 VAP CPC 대조군 — 0 ms (encoder, full model 모두)
- [x] 회계표 — [[output-encoder-causality-audit]]
- [x] **판정**: Qwen 초과 → [[decision-asr-backbone]] 수정, [[task-qwen-aut-causal-adaptation]] 생성

## 진행 기록

- 2026-09-03: **완료.** 결과 `raw/sources/experiments/2026-09-03-causality-audit.json`. 첫 실행에서 시간축 슬라이싱 버그(mel축을 자름)로 Nemotron 이 0 으로 나왔던 것을 수정.
- 2026-09-03: 생성. Qwen 기술 보고서에 causality 명시 없음을 확인.
- 2026-09-03: Nemotron 모델 카드에서 `att_context_size=[56,N]` 확인 — 80 ms chunk 는 우측 0
  (attention strictly causal). 절단 실험은 conv 암묵 lookahead 확인용으로 축소. Qwen 은
  transformers 백엔드가 offline 기본이라 AuT 직접 chunk 호출 코드가 필요.
