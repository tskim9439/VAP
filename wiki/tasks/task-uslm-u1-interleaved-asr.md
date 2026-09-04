---
type: task
status: open
owner: tskim
due: 2026-10-16
priority: p0
created: 2026-09-04
updated: 2026-09-04
summary: USLM U1 — Nemotron frozen + Qwen3-0.6B LoRA interleaved ASR(텍스트 스트림만), WER 상대 열화 ≤10% 관문
sources:
  - [[output-interleaved-streaming-slm-architecture]]
  - [[decision-target-architecture]]
  - [[decision-asr-backbone]]
---

# USLM U1 — aligned interleaved ASR

## 배경

[[output-interleaved-streaming-slm-architecture]] U1. 전사가 무너진 통합 모델은 의미가 없으므로 이 관문을 **가장 먼저** 통과해야 한다.
A 안이 기각되어 fallback 이 없다 — 조기 판정이 중요하다.

## 완료 조건

- [ ] **backbone = Nemotron [56,0] → adapter(U0.5 통과본) → Qwen3-ASR thinker LoRA** — [[decision-asr-backbone]]. 두 화자 joint chunk token 용 merge 학습. U0 의 interleaved target 사용
- [ ] `<NEXT_AUDIO>` / `<EMPTY_AUDIO>` / `<SPK_A|B>` 어휘 추가, chunk 당 최대 M 방출
- [ ] 여러 지연 예산(δ)으로 emission schedule 무작위화, aux CTC(학습 전용)
- [ ] 평가: WER/CER(EN otoSpeech·KO AI Hub VS_02), TTFT, time-to-final, evidence-time 위반률 0, **p99 tick < 80 ms**
- [ ] **관문**: Nemotron RNN-T 대비 WER/CER 상대 열화 ≤ 10 % — 실패 시 [[decision-target-architecture]] 재논의
- [ ] 융합 ablation 준비: interleaving-only(기본) vs gated contextual residual

## 진행 기록

- 2026-09-04: 생성 (IS-SLM 단일 주력 결정에 따라).
