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
- 2026-09-04: **스모크 2 회 통과** (`u1-smoke`, `u1-smoke2`). 학습 창 1,651(대화당 3 개 표본 시) / 전체 침묵 격자 사용 시 훨씬 많음. 시퀀스 길이 ≈ 850–930 (30 s 창: audio 375 + NEXT_AUDIO 375 + 텍스트 ≈ 150).
  150 step(bs 4) 손실: 전체 11.1 → 0.9, `<NEXT_AUDIO>` 위치 16.0 → 0.2, 텍스트 위치 6.2 → 4.2 — 모델이 먼저 "무방출(84 % chunk)" 을 학습하고 텍스트는 뒤따른다(초기 평가 tok/chunk 0 → 정상적인 초기 상태, v0 step 3000 평가로 확인).
  스트리밍 평가 속도: LoRA 미병합 시 forward 당 68 ms(창 128 s) → `merge_adapter()` 후 56 ms(창 21 s). HF 단일 토큰 forward 의 파이썬 오버헤드가 지배 → **개선 후보: 창 배치 디코드(ragged KV) 또는 CUDA graph**. 현재는 코퍼스당 8 창 × 3 = 8–10 min/평가.
  vs02 val 창 0 — vs02 정렬이 id 정렬 앞부분(107/186)까지만 끝나 val 분할(뒤 8 %)이 비어 있음 → 정렬 완료 후 채워짐.
- 2026-09-04 23:08 KST: 첫 v0 시도 bs 8 OOM — GPU 1 은 **40 GB 카드**, fp32 logits(B·L·152k) 가 4.4 GB 씩 → `--accum` 추가, bs 4 × accum 2 로 재시작(25.8 GB, 0.7 s/step, 12k ≈ 2.5 h + 평가).
- 2026-09-04 23:10 KST: **v0 run 시작** (`u1-interleaved-v0`, 12k step, bs 8, lr 1e-4/1e-4, δ∈{2,3,4,6}, M=4, U0.5 distill-12k 초기화, 평가 3000 step 마다 8 창/코퍼스). 예상 ≈ 4–5 h.
- 2026-09-04 23:31 KST: **v0 재시작**. 첫 v0(step 1750) 은 QC 필터 이전 정렬을 써서 중단했다 — otoSpeech 창의 일부가 최대 14.5 s backlog 를 담고 있어 지연 목표가 오염된다.
  재시작본(`u1-v0b`)은 `bad_utterance` 필터 적용, 나머지 설정 동일(12k step, bs 4 × accum 2, δ∈{2,3,4,6}, M=4). 손실 궤적은 이전과 동일(step 150 에서 next 0.14 / text 4.0).
- 2026-09-05 00:10 KST: **v0(next_weight 1.0) step 3000 평가 — 방출 0, WER 1.0.** 원인 진단(`experiments/u1_diag_emission.py`, teacher forcing 4 창):
  | 위치 | P(정답) | P(`<NEXT_AUDIO>`) | 정답 top-1 | 정답 top-5 |
  |---|---|---|---|---|
  | 텍스트를 내야 하는 위치(n=221) | 0.376 | **0.463** | 39.8 % | 92.8 % |
  | 무방출 위치(n=750) | 0.902 | 0.902 | 97.3 % | 100 % |
  → 모델은 **무엇을 쓸지는 안다**(top-5 92.8 %). 문제는 **언제 쓸지의 결정**이다. 정확한 chunk 하나만 정답으로 두는 CE 는 ±1 chunk 모호성을 허용하지 않아
  모델이 "지금은 아님"(라벨의 83 %)으로 헤지하고, greedy argmax 가 매 chunk `<NEXT_AUDIO>` 를 고른다. RNN-T 의 blank 지배와 같은 구조.
  디코드 페널티 sweep(`next_bias`, blank penalty 유사): 0 → 방출 0; **2 → tok/chunk 0.135**(목표 0.20), WER 0.85; 4 → 폭주(tok/chunk 3.28, M 강제 68 %).
  = 작동 구간이 좁고 불안정 → 디코드 보정만으로는 부족, **학습 목표를 고쳐야 한다**.
- 2026-09-05 00:22 KST: **v1 시작 — `--next-weight 0.3`** (`<NEXT_AUDIO>` 위치 CE 가중치 0.3, 불균형 83:17 ≈ 5:1 을 부분 보정). 평가에 **bias sweep(0,1,2)** 내장.
  v0 는 `u1-interleaved-nw1.0/` 로 보존(next_weight 1.0 대조군, step 3000 ckpt + 진단 수치).
  다음 후보(효과 없을 시): (a) 목표 시각 허용 창(±1 chunk 라벨 스무딩 또는 lattice), (b) U2 self-conditioning 조기 도입(교사 강제 100 % 가 방출 오류 회복을 못 배우게 함), (c) 방출 결정을 별도 헤드로 분리.
- 2026-09-05 01:0x KST: **v1(next_weight 0.3) step 3000 — 방출 붕괴 해소, 정확도는 미달.** otoSpeech 6 창:

  | bias | tok/chunk (목표 0.20) | WER | 지연 p50 | 지연 p99 | evidence 위반 | M 강제 |
  |---|---|---|---|---|---|---|
  | 0 | 0.088 | 0.765 | 232 ms | 664 ms | 10.9 % | 0 |
  | 1 | 0.281 | 0.774 | 216 ms | 1028 ms | 8.4 % | 1.3 % |
  | 2 | 2.20 | 8.61 | 44 ms | 4593 ms | 48.5 % | 43.3 % |

  `next_weight` 0.3 만으로 bias 0 에서도 방출이 살아났다(v0 는 0.000). 지연 p50 232 ms 는 δ=2(160 ms) + 정렬 오차로 타당하다.
  **남은 문제는 내용 정확도**(WER 0.77, 참조 토큰의 31–58 % 만 일치). 원인 후보 셋:
  1. **노출 편향** — 학습은 교사 강제 100 %, 평가는 자기 이력. 방출 하나가 틀리면 이후 이력이 오염된다(U2 의 주제이나 U1 에서 이미 치명적).
  2. **두 화자 joint chunk token** — merge([adapter(A); adapter(B)]) 가 겹친 발화를 1024-d 하나로 평균한다. 화자당 오디오 토큰 2 개(창당 375 → 750 위치) 안을 ablation 으로 둔다.
  3. 학습량 — step 3000/12000, 텍스트 손실 1.05 로 아직 내려가는 중.
  → v1 을 12k 까지 돌려 3000/6000/9000/12000 추세로 (3) 을 먼저 배제한다.
