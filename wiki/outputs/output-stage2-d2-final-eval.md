---
type: output
status: active
created: 2026-09-10
updated: 2026-09-10
summary: 확장 DB(EN 2.2 k h + KO 3.4 k h, 8 코퍼스) run D2 최종 모델의 WER/CER·타이밍 평가 — C2 와 같은 select 표본으로 비교, 71631-dev(자유대화) 추가. 71631 오디오 조립 버그(src_offset_s 누락)로 D2 는 오염된 run 이며 D3 로 재실행
sources:
  - [[output-stage2-c2-final-eval]]
  - [[output-dataset-schema-v1]]
  - [[source-mxc-model-load-latency]]
  - [[output-vapasr-model-and-sequence]]
---

# Stage 2 run D2 최종 평가 (2026-09-10)

## 0. 요약

- D2 = C2 final 에서 시작, 8 코퍼스(EN librispeech-960·swbd·voxpopuli·yodas, KO kspon-full·nikl-1000·aihub71631·031/033) 5.6 k h, 10 epoch = 16,240 step, 4 노드 32 GPU.
- **결함**: 학습·평가 로더의 행 인덱스가 segment 의 `src_offset_s` 를 버려 **71631 발화(KO 배치의 ~9 %)가 대화 파일 첫머리 오디오(대부분 무음)로 조립**됐다. 텍스트는 정상, 오디오만 엉뚱한 "음성 근거 없는 라벨" 이었고, step 2,000·6,000 의 조기 방출·불안정과 맞물린다. 2026-09-10 수정(커밋 a7c059a, 항목 캐시 v2) → 같은 설정으로 D3 재실행.
- 같은 select 표본(dev-clean 50 / dev-other 50 / kspon-dev 300 / ah71-dev 300, seed 7)에서 D2 final 은 C2 final 보다 **세 기준 세트 모두 나쁘고**(δ=2: 0.126/0.212/0.306 vs 0.103/0.180/0.244), 자유대화(ah71-dev)도 C2 보다 나쁘다(0.524 vs 0.484 — 오염된 71631 학습의 직접 증거). δ=4 에서는 EN 이 C2 와 비슷(0.062/0.125 vs 0.056/0.122).
- 판정: **D2 는 채택하지 않는다.** "확장 DB 의 효과" 는 D3 로 판단한다.

## 1. 대상과 학습 설정

- 모델: [[output-vapasr-model-and-sequence]] (Nemotron [56,0] 동결 → adapter → Qwen3-ASR thinker full FT, 600.2 M).
- 데이터(발화 시간): EN librispeech-960 1,033 · swbd 290 · voxpopuli 521 · yodas-en129 334 (≈2,180 h); KO kspon-full 1,189 · nikl-1000 1,451 · aihub71631 159 · 031/033 578 (≈3,380 h). rank 당 배치 EN 24 / KO 96 → 유효 EN 768 / KO 3,072, steps/epoch 1,624.
- 학습: job 67126(코드 오류 `lang_of` 로 실패) → 67293(hpc 4 노드, 선점 3 회 후 restart 3 에서 완주). lr 4e-5(adapter 1e-3), warmup 200 + cosine, 10 epoch = 16,240 step, 1.60 s/step, 학습 구간 약 7.2 h(16:36 → 00:33). 노드 로컬 env 스테이징·인코더 캐시로 재시작당 로드 8–154 s.
- 산출물: `/soundai/Model/VAPASR/hf-D2/{final, eval/, eval-ckpts/, tb/}`. 산출물은 보존하되 후속 run 의 초기값으로 쓰지 않는다.

## 2. 학습 중 추세 (sentinel: dev-clean 10 / dev-other 10 스트림 / kspon-dev 100 발화, δ=2)

| step | dev-clean WER · viol80 · lat p50/p90 | dev-other WER · viol80 · lat p50/p90 | kspon-dev CER · viol80 · lat p50/p90 |
|---|---|---|---|
| 2000 | 0.338 · 0.027 · 129/209 ms | 0.480 · 0.056 · 124/213 | 0.410 · 0.073 · 126/260 |
| 4000 | 0.125 · 0.004 · 209/272 | 0.283 · 0.012 · 213/293 | 0.375 · 0.064 · 152/255 |
| 6000 | 0.271 · 0.023 · 135/209 | 0.368 · 0.011 · 156/244 | 0.390 · 0.087 · 142/253 |
| 8000 | 0.143 · 0.003 · 177/239 | 0.272 · 0.010 · 186/270 | 0.359 · 0.059 · 145/244 |
| 10000 | 0.110 · 0.001 · 209/272 | 0.245 · 0.003 · 208/293 | 0.345 · 0.052 · 164/279 |
| 12000 | 0.114 · 0.004 · 192/258 | 0.252 · 0.008 · 206/286 | 0.322 · 0.051 · 175/266 |
| 14000 | 0.123 · 0.005 · 207/272 | 0.226 · 0.008 · 206/286 | 0.311 · 0.047 · 175/283 |
| 16000 | 0.123 · 0.003 · 193/258 | 0.258 · 0.013 · 206/293 | 0.312 · 0.048 · 175/281 |

- C2 final 의 같은 sentinel 값은 0.109 / 0.225 / 0.256 이었다. D2 는 EN 이 C2 수준에 닿았다 멀어지기를 반복했고(2k·6k 급락은 방출 지연 p50 이 130 ms 대로 짧아지는 "조기 방출" 과 동반), KO 는 끝까지 C2 에 못 미쳤다.
- 이 진동은 처음엔 LR 정점의 최적화 불안정으로 해석했으나, 사후에 71631 오디오 결함이 확인됐다. 무음 오디오에 텍스트 라벨이 붙은 배치가 "음성 없이 내뱉는" 방향의 그라디언트를 주므로 조기 방출·viol80 상승과 일치한다.

## 3. select 평가 (dev-clean 50 / dev-other 50 스트림, kspon-dev 300 / ah71-dev 300 발화, seed 7, bias 0)

ah71-dev 는 aihub71631-dev(VS_02 실외 자유대화, 화자 채널·`src_offset_s` 로 잘라낸 발화) 300 개. 수정된 로더로 잰 값만 싣는다(checkpoint-14000·16000 의 ah71-dev 는 결함 로더 값이라 제외).

| 모델 · δ | dev-clean WER · lat p50 | dev-other WER · lat p50 | kspon-dev CER · lat p50 | ah71-dev CER · lat p50 |
|---|---|---|---|---|
| C2 final · δ=2 | **0.103** · 209 ms | **0.180** · 216 | **0.244** · 200 | **0.484** · 204 |
| D2 14000 · δ=2 | 0.122 · 193 | 0.225 · 213 | 0.306 · – | – |
| D2 16000 · δ=2 | 0.129 · 194 | 0.215 · 216 | 0.303 · – | – |
| D2 final · δ=2 | 0.126 · 195 | 0.212 · 216 | 0.306 · 162 | 0.524 · 154 |
| C2 final · δ=4 | **0.056** · 357 | **0.122** · 371 | **0.156** · 361 | 0.406 · 364 |
| D2 final · δ=4 | 0.062 · 353 | 0.125 · 371 | 0.169 · 332 | 0.378 · 311 |

- D2 의 KO 방출 지연 p50 은 C2 보다 40 ms 짧고 viol80 은 kspon-dev 0.051 · ah71-dev 0.085 로 높다(C2 kspon-dev 0.024). 결함 데이터의 흔적.
- δ=4 에서는 EN 이 C2 와 비슷하고(0.062/0.125 vs 0.056/0.122), KO 는 kspon-dev 가 여전히 뒤지지만(0.169 vs 0.156) ah71-dev 는 D2 가 조금 낫다(0.378 vs 0.406). 즉 결함 데이터로도 자유대화 도메인 자체는 약간 배웠고, 지연을 길게 주면 그 이득이 드러난다.

## 4. 타이밍 스윕 (D2 final, sentinel 10/10/100, seed 7)

(측정 중 — δ ∈ {2,3,4}, δ=2 에서 next_bias ∈ {1,2})

## 5. 판정

1. D2 final 은 C2 final 에 못 미친다(δ=2 기준 dev-clean +0.023, dev-other +0.032, kspon-dev +0.062). 71631 결함 때문에 "새 데이터 + 같은 레시피" 의 효과는 이 run 으로 판단할 수 없다.
2. 자유대화(ah71-dev)는 두 모델 모두 CER 0.48–0.52 로 매우 높다. 낮은 녹음 레벨·거친 라벨의 코퍼스라 D3 에서 제대로 학습해도 절대값은 높을 것이며, 개선 폭이 새 데이터의 가치를 말해 줄 것이다.
3. 후속: D3(같은 설정, 수정 로더, 항목 캐시 v2) → E1(D3 final 에서 인코더 해동, BS 12/24, LR 2e-5 / 인코더 1e-5). D2 의 사전 프로브(experiments/probe_*)는 manifest 를 직접 조립해 결함을 놓쳤으므로, 앞으로 코퍼스 점검은 **학습 데이터셋 객체(MonoStreamDataset) 경로**로 한다(tests/test_mono_rows_offset.py).

## 6. 재현

- 학습: `sbatch --partition=hpc --nodes=4 --requeue --exclude=slurmmxch200v5-hpc-52 --export=ALL,RUN=D2,INIT=/soundai/Model/VAPASR/hf-C2/final,TRAIN=librispeech-960:swbd-train:voxpopuli-train:yodas-en129:kspon-full:nikl-1000:aihub71631-train:aihub-bc-train,EPOCHS=10,BS_EN=24,BS_KO=96,EVAL_EVERY=2000,SAVE_EVERY=500 slurm/s3_train_hf.sbatch`
- 평가: `experiments/s3_d2_eval.sh`(GPUS=0,1,3,4, 컨테이너 sa_tskim_fd) → `hf-D2/eval/select-*.json`, `select-d4-0.json`, `sweep-*`, `hf-C2/eval/selectx-d{2,4}-0.json`.
- 코퍼스 정합 프로브: `experiments/probe_corpus_sanity.py`, `probe_ko_mismatch.py`, `probe_bc_fillers.py`, `probe_align_coverage.py`.
