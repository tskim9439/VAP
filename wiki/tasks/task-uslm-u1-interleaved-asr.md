---
type: task
status: doing
owner: tskim
due: 2026-10-16
priority: p0
created: 2026-09-04
updated: 2026-09-06
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
- 2026-09-05 01:30 KST: **원인 분해 진단 2 종** (`u1_diag_emission.py`, v1 step-3000 ckpt).
  1. **화자 배정은 원인이 아니다.** 화자별 오류와 화자 무시(pooled) 오류가 사실상 같다: otoSpeech 0.607 vs 0.601(bias 1), vs02 0.622 vs 0.619. 방출된 토큰의 `<SPK_x>` 배정은 맞다.
  2. **겹침 구간이 더 나쁘지만 소수다.** teacher forcing top-1: otoSpeech 단독 54.5 % / 겹침 37.5 %(n=24), vs02 단독 67.3 % / 겹침 50 %(n=6). 겹침은 텍스트 위치의 5–20 %.
  → **단독 발화에서도 top-1 이 54–67 %** 인 것이 본질. next_weight 0.3 으로 방출 보정 자체는 개선됨(텍스트 위치 P(gold) 0.376→0.502, P(next) 0.463→0.278, top-1 39.8→57.5 %).
  남은 가설: (a) 학습량(텍스트 손실 1.05→0.95 계속 하강), (b) **merge 로 인한 분포 이동** — U0.5 adapter 는 단일 채널 입력으로 학습됐는데 U1 은 두 채널 평균을 넣는다(단독 발화에서도 무음 채널과 평균 → 스케일·SNR 변화), (c) 노출 편향(교사 강제 100 %).
- 2026-09-05 01:35 KST: **화자별 오디오 토큰 옵션 구현**(`--audio-per-chunk 2`). chunk 당 오디오 토큰을 A, B 두 개로 두어 merge 를 없앤다(U0.5 와 같은 단일 채널 분포 유지). 오디오 위치 375 → 750, 시퀀스 ≈ 1250. v1 추세 확인 후 v2 로 실행.
- 2026-09-05 10:50 KST: **v1(next_weight 0.3, merge 토큰) 12k step 완주.** 코퍼스별 최적 bias 기준:

  | step | otoSpeech WER | 실내 CER | 실외 CER |
  |---|---|---|---|
  | 3000 | 0.765 | 0.485 | 0.478 |
  | 6000 | 0.524 | 0.679 | 0.462 |
  | 9000 | 0.518 | **0.365** | 0.341 |
  | 12000 | 0.583 | **0.339** | **0.271** |

  지연(최적 bias): p50 140–230 ms, p90 310–390 ms — δ=2(160 ms) 설계와 일치. p99 는 이월 때문에 최대 13 s 꼬리가 남는다.
  학습이 진행되며 최적 bias 가 1 → 0 으로 이동(모델이 스스로 보정됨).
  **판정: 가설 (a) 학습량은 한국어에서만 유효.** 실외 0.478 → 0.271 로 계속 개선, 실내 0.485 → 0.339. 반면 **영어는 0.52 근방에서 정체**(12k 에서 오히려 0.583).
  U0.5 오프라인(EN 18.2 / 실내 17.5 / 실외 14.5 %) 대비 EN 3 배, KO 2 배 열화 — U1 관문(RNN-T `[56,0]` 대비 ≤ +10 %, 즉 EN ≤ 27.9 %)에 크게 미달.
- 2026-09-05 10:57 KST: **v2 시작 — `--audio-per-chunk 2`**(화자별 오디오 토큰, merge 제거). 시퀀스 L 900 → 1240, 26.9 GB, ≈0.83 s/step(12k ≈ 2.8 h).
  가설 (b) 검증: U0.5 adapter 는 단일 채널로 학습됐는데 v1 은 두 채널 평균을 넣어 분포가 이동했다. 같은 step 수에서 v1 과 직접 비교한다.

### U1a-0 — 단일 화자 mono 파일럿 (Stage 1, `plans/stage1-mono-pilot.md`)

- 2026-09-05: **입력 mono 단일 채널 결정**([[decision-mono-input]]) 에 따라 두 채널 경로(v1 merge / v2 화자별 오디오 토큰)는 진단 기록으로만 보존하고, mono 경로를 새로 둔다: `vapasr/uslm/mono_data.py`·`mono_model.py`, `experiments/s1_*.py`. 데이터는 LibriSpeech train-clean-100(100.6 h, 챕터 내 연결 20–30 s 스트림 13,182) + KsponSpeech_01 0001~0062(96.3 h, 파일 = 스트림 62,000). 검증기(PCM 16 kHz·16-bit 가정 통과, `.trn` 파서, LibriSpeech 집계) 통과. 텍스트 규약은 `vapasr/data/textnorm.py` 로 통일(EN 숫자 단어, KO 숫자 한글 읽기 — Qwen3-ASR·Nemotron 출력 실측과 동일, `raw/sources/experiments/2026-09-05-asr-output-style-probe.md`).
- 2026-09-05: 코드 결함 3 건 수정 — (1) `build_interleaved` 가 δ 로 스트림 끝을 넘긴 토큰을 `buckets[n_chunks]` 에서 버리던 것(두 채널 경로도 동일) → `<EMPTY_AUDIO>` 입력 flush 라운드 규약으로 학습·디코드 양쪽 처리, (2) 임베딩 행렬이 AdamW weight decay 에 전 행 감쇠 → wd 0 그룹, (3) overfit 모드 신설(언어별 16 고정, δ=2, 타깃 보존 assert, 표적 사례 강제 포함). 평가 3 종 분리(sentinel/select/final), matched=0 처리, tick p99, 언어별 bs 2/8.
- 2026-09-05 **overfit(1,500 step, 옛 규약)**: 900 step 부터 EN WER 0.000 / KO CER 0.000, tok/chunk = 참조, viol80 = 0, 지연 p50 +201 / +189 ms, p99 +240 / +312 ms — **기능 관문 통과**. tick p99 151–186 ms(경합 전) > 80 ms → **실시간성 관문 미통과**(단독 측정 전).
- 2026-09-06 **overfit-v2(900 step, 새 규약 `align2/` + textnorm)**: 표본 EN 16(발화 경계 8) + KO 16(숫자 이중표기→발음형 8). @900 EN WER **0.000** (tok/chunk 0.237 = 참조, p50 +201, p90 +231, p99 +240 ms, viol80 0) · KO CER **0.000** (0.292 = 참조, p50 +202, p99 +278, viol80 0). 새 규약에서도 동일하게 통과. `losti` 결합 해소 확인.
- 2026-09-06: mxc 에 대화 코퍼스 업로드 완료(rack4 → 맥 T5 → mxc; rack4→mxc 직접은 정책상 불가): otoSpeech16k 2,100 파일 23 GB, AI Hub 71631 wav 943 파일 54 GB(라벨은 기존 zip), TurnBench dev(refs/main c29aa4e)+test 41 파일 17 GB → `.env` `MXC_OTOSPEECH_DIR` 등. **6,000-step 파일럿(`s1-mono-pilot`) 시작**(GPU 6).
- 2026-09-06 **6,000-step 파일럿 결과**(random init, LoRA r16, EN bs 2 / KO bs 8, 1 GPU): sentinel(bias 0) dev-clean WER 0.556 → 0.221(5k) → 0.228(6k), dev-other 0.582 → 0.289 → 0.311, kspon-dev CER 0.851 → 0.623(tok/chunk 0.17, 참조 0.27, 최적 bias 1.0). EN 방출률·타이밍 정상(viol80 0.4 %, p50 ≈ 200 ms). 대조군(같은 세트·채점): Nemotron RNN-T `[56,0]` dev-clean 4.4 / dev-other 8.2 / kspon-dev 20.2 %, Qwen 오프라인 2.4 / 4.3 / 12.2 % (`raw/sources/experiments/2026-09-06-s1-baselines/`). 데이터 노출 < 0.5 epoch, 교사 강제 top-1 0.745 → 과소학습.
  **WER 진단**(`experiments/s1_diag_wer.py`, ckpt-6000, `raw/sources/experiments/2026-09-06-s1-pilot-diag.json`): dev-clean 교사강제 top-1 0.728(top-5 0.939) vs 자유실행 오류 0.233 = S 0.153 + D 0.020 + I 0.061 → **격차 없음 = 노출 편향 아님**, 음향적 유사어 치환("zeal of cyril" → "truth is that caroline")이 주 오류 → 용량·학습량. dev-other 0.595 / 0.326. **kspon-dev 는 D 0.444** — 교사강제 top-1 0.648 인데 자유실행에서 문장 중간에 방출을 멈춤(예 "소개팅 앱을 …" → "아 쇼케") → KO 는 **방출 결정 붕괴**가 본질.
  → 개선 run A/B(SLURM, 8 GPU): 공통 = mono adapter 증류 init(`s1_distill_adapter.py`, 타깃 qwen-aut-block8s) + KO `next_weight` 0.15 + 유효 배치 EN 96 / KO 384 + 15 epoch. A = LoRA r16(lr 4e-4), B = thinker 0.6B full FT(lr 4e-5). 학습기는 torchrun DDP·자동 재개·선점 저장(`slurm/s1_mono.sbatch`).
