---
type: output
status: draft
created: 2026-09-08
updated: 2026-09-08
summary: 1,930 h(LibriSpeech-960 + KsponSpeech) full FT run C2 최종 모델의 WER/CER 과 방출 타이밍 평가 — select 표본으로 checkpoint 선정, 전체 dev 로 최종 수치, δ·next_bias 스윕으로 지연–정확도 트레이드오프
sources:
  - [[source-hf-trainer-migration]]
  - [[output-vapasr-model-and-sequence]]
  - [[output-stage1-mono-pilot]]
  - [[source-stage1-mono-run-ab]]
---

# Stage 2 run C2 최종 평가 (2026-09-08)

## 1. 대상과 학습 설정

- 모델: [[output-vapasr-model-and-sequence]] (Nemotron [56,0] 동결 → adapter → Qwen3-ASR thinker full FT, 600.2 M 학습 파라미터).
- 데이터: librispeech-960 (1,033 h) + kspon-full (1,189 h). 시퀀스·손실 규약은 §4 참조(δ ∈ {2,3,4,6}, next_weight EN 0.3 / KO 0.15).
- 학습(job 66103 → 66201 → 66227 → 66260, HF Trainer `experiments/s3_train_hf.py`): C step-500 가중치에서 시작, 16 GPU(step ≤ 1,500) → 32 GPU 로 재개, 유효 배치 EN 384 / KO 1,536, lr 6e-5(adapter 1e-3), warmup 500 + cosine, 30 epoch = 21,960 step, 총 5.6 h(32 GPU), 최종 학습 손실 0.010(top-1 0.999). Liger 커널·gradient checkpointing·bf16.
- 산출물: `/soundai/Model/VAPASR/hf-C2/{final, eval/, eval-ckpts/, tb/}`.

## 2. 학습 중 추세 (sentinel: dev-clean 10 스트림 / dev-other 10 / kspon-dev 100 발화, δ=2, bias 0)

| step | dev-clean WER · viol80 · lat p50/p90 | dev-other WER · viol80 · lat p50/p90 | kspon-dev CER · viol80 · lat p50/p90 |
|---|---|---|---|
| 1500 | 0.229 · 0.016 · 139/221 ms | 0.334 · 0.025 · 172/252 | 0.413 · 0.072 · 139/281 |
| 4000 | 0.175 · 0.011 · 168/218 | 0.333 · 0.021 · 162/252 | 0.392 · 0.072 · 132/260 |
| 6000 | 0.120 · 0.004 · 188/258 | 0.301 · 0.021 · 180/269 | 0.371 · 0.060 · 133/253 |
| 8000 | 0.134 · 0.001 · 183/241 | 0.277 · 0.015 · 186/269 | 0.378 · 0.060 · 138/254 |
| 10000 | 0.125 · 0.004 · 199/264 | 0.237 · 0.002 · 208/288 | 0.333 · 0.060 · 151/244 |
| 12000 | 0.107 · 0.001 · 194/263 | 0.218 · 0.006 · 213/293 | 0.313 · 0.053 · 175/275 |
| 14000 | 0.109 · 0.004 · 209/273 | 0.229 · 0.005 · 213/293 | 0.301 · 0.038 · 180/286 |
| 16000 | 0.107 · 0.004 · 209/273 | 0.237 · 0.008 · 213/286 | 0.278 · 0.032 · 190/286 |
| 18000 | 0.110 · 0.002 · 209/276 | 0.222 · 0.010 · 224/293 | 0.265 · 0.029 · 193/294 |
| 20000 | 0.111 · 0.003 · 209/272 | 0.229 · 0.006 · 224/293 | 0.256 · 0.033 · 195/295 |
| final (21960) | 0.109 · 0.003 · 209/276 | 0.225 · 0.006 · 224/293 | 0.256 · 0.027 · 195/291 |

- EN 은 step 12,000(epoch 16) 이후 정체, KO 는 끝까지 하락. 학습 손실은 0.01 까지 내려갔으므로 EN 은 이 데이터 규모의 포화로 본다(과적합으로 인한 dev 반등은 없음).
- 방출 지연 p50 은 학습이 진행될수록 130 → 210 ms 로 늘고 viol80(참조보다 80 ms 이상 이른 방출)은 0.07 → 0.003–0.03 으로 줄었다: 모델이 δ=2 규약(이론 160–240 ms)에 맞춰 "기다렸다 내는" 쪽으로 수렴.

## 3. checkpoint 선정 (select: 셋당 50 스트림 + 300 발화, seed 7)

TODO(select 결과)

## 4. 최종 수치 (전체 dev)

TODO(full dev)

## 5. 타이밍: δ 와 next_bias 스윕 (final, sentinel 표본 seed 7: dev-clean 10 / dev-other 10 스트림, kspon-dev 100 발화)

δ 는 prefix 의 `<DELAY_δ>` 조건 토큰(학습 시 {2,3,4,6} 혼합). 지연 = (방출 청크+1)·80 ms − 참조 토큰 종료 시각. viol80 = 80 ms 이상 이른 방출 비율.

| δ (이론 지연) | dev-clean WER · 지연 p50/p90/p99 | dev-other WER · 지연 p50/p90/p99 | kspon-dev CER · 지연 p50/p90/p99 | viol80 |
|---|---|---|---|---|
| 2 (160–240 ms) | 0.095 · 209/256/305 ms | 0.155 · 222/290/386 | 0.236 · 195/288/608 | 0.001 / 0.003 / 0.029 |
| 3 (240–320 ms) | **0.061** · 288/327/382 | **0.118** · 290/351/420 | **0.184** · 278/361/656 | 0.000 / 0.003 / 0.014 |
| 4 (320–400 ms) | **0.049** · 368/406/462 | **0.111** · 378/430/509 | **0.171** · 358/441/693 | 0.000 / 0.001 / 0.007 |

- 조건 토큰만 바꿔도 정확도–지연이 크게 움직인다: δ 2→4 에서 dev-clean WER 0.095→0.049(−48 %), dev-other −28 %, kspon −28 %, 지연 p50 +160 ms. δ=4 의 dev-clean 은 RNN-T 대조군(0.044) 수준에 근접(단, sentinel 표본; select 표본 확인 §3).
- 강제 종료(청크당 상한) 비율은 모든 설정에서 0, tok/chunk 는 δ 와 무관(0.21–0.27) — 방출 총량은 같고 시점만 늦춰진다.
- tick p99 는 110–120 ms 로 80 ms 청크보다 길어 **실시간 관문(tick < 80 ms) 은 미달**(H200 1 GPU, KV cache greedy). 디코더 커널·캐시 최적화 과제.
- next_bias(δ=2, `<NEXT_AUDIO>` logit 을 b 만큼 낮춰 방출을 앞당김):

| bias | dev-clean WER · 지연 p50 | dev-other WER · 지연 p50 | kspon-dev CER · 지연 p50 |
|---|---|---|---|
| 0 | 0.095 · 209 ms | 0.155 · 222 ms | 0.236 · 195 ms |
| 1 | 0.102 · 207 ms | 0.180 · 203 ms | 0.290 · 177 ms |
| 2 | 0.137 · 191 ms | 0.242 · 188 ms | 0.334 · 156 ms |

  bias 는 지연을 10–30 ms 밖에 못 줄이면서 오류를 크게 늘린다 → 지연 제어는 δ 조건 토큰으로, bias 는 쓰지 않는다(파일럿 결론과 동일).

## 6. 판정

TODO

## 7. 재현

- 오프라인 평가: `experiments/s3_train_hf.py --eval-only --init <ckpt> --out-dir <run> --sentinel-stream N --sentinel-utt M --eval-seed S --eval-tag T --eval-delay δ --eval-bias b`
- select 루프 `experiments/s3_select_loop.sh`, 스윕 `experiments/s3_timing_sweep.sh`, 학습 중 자동 평가 `experiments/s3_eval_loop.sh`. 원자료: `raw/sources/experiments/2026-09-08-hf-C2-final-eval/`.
