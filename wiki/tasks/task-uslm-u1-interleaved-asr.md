---
type: task
status: doing
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
- [ ] **관문**: Nemotron RNN-T **`[56,0]`(25.4 / 23.4 / 24.5 %)** 대비 WER/CER 상대 열화 ≤ 10 % — 즉 스트리밍 방출 후에도 oto ≤ 27.9 / 실내 ≤ 25.7 / 실외 ≤ 27.0 %. 실질 목표는 U0.5 오프라인 수준(18–20 %) 유지. 실패 시 [[decision-target-architecture]] 재논의
- [ ] 융합 ablation 준비: interleaving-only(기본) vs gated contextual residual

## 진행 기록

- 2026-09-04: 생성 (IS-SLM 단일 주력 결정에 따라).
- 2026-09-04: U0.5 통과(관문 재정의) → **착수**. 학습 준비 구현:
  - `vapasr/uslm/interleave_data.py` — 정렬 jsonl + Nemotron 12.5 Hz 캐시(1 프레임 = 1 chunk)에서 30 s 창을 자른다. 창 시작은 **양 화자 침묵 시각**(부분 발화 없음). δ ∈ {2,3,4,6} 프레임을 창마다 무작위 → `<DELAY_d>` 프리픽스 토큰으로 조건화. M=4(U0 확정). 시퀀스 = prefix(`…assistant\nlanguage {Lang}<asr_text><DELAY_d>`) + `[AUDIO_k] (<SPK_x>) tok… <NEXT_AUDIO>` 반복. 손실은 audio 위치·prefix 제외 전부.
  - `vapasr/uslm/model.py::InterleavedASR` — chunk 임베딩 = merge([adapter(f_A); adapter(f_B)]), merge 는 [½I ½I](+미세 비대칭) 로 초기화해 U0.5 adapter 를 그대로 잇는다. 특수 토큰 12 개(`<NEXT_AUDIO> <EMPTY_AUDIO> <SPK_A> <SPK_B> <DELAY_1..8>`)는 thinker 임베딩 행렬의 여유 행(151705–151716)을 쓰고 grad mask 로 그 행만 학습(lm_head tied). `stream_decode` = KV cache 로 chunk 마다 greedy: `<NEXT_AUDIO>` 또는 M 도달 시 다음 chunk(강제 횟수 기록).
  - `experiments/u1_train_interleaved.py` — U0.5 ckpt(adapter+LoRA) 초기화, cosine 스케줄, 진단용 손실 분리(NEXT_AUDIO vs 텍스트). 평가 = 고정 val 창(코퍼스당 20)을 스트리밍 디코드 → 화자별 WER/CER, 토큰 지연((k+1)·80 ms − 정렬 종료 시각; difflib 단조 매칭) p50/p90/p99, evidence 위반률(지연 < 0, < −80 ms), M 강제 비율, chunk 당 토큰 수.
  - v0 에서 뺀 것: aux CTC, 지연 커리큘럼(무작위화로 대체), 인코더 unfreeze(ablation 으로 예정), gated residual fusion(U3 ablation).
