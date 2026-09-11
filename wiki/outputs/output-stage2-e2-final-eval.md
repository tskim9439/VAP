---
type: output
status: active
created: 2026-09-10
updated: 2026-09-10
summary: 인코더 해동 run E2(C2 final → 8 코퍼스 5.6 k h, Nemotron 인코더 LR 1e-5 · thinker 2e-5, 3 epoch) 최종 평가 — 같은 select 표본에서 C2 대비 EN WER 25 %·KO CER 30–40 % 상대 감소, 자유대화 0.484 → 0.292. dev 는 선택용, test 는 최종 보고용으로 분리
sources:
  - [[output-stage2-c2-final-eval]]
  - [[output-stage2-d2-final-eval]]
  - [[output-vapasr-model-and-sequence]]
  - [[source-soulx-duplug]]
---

# Stage 2 run E2 (인코더 해동) 최종 평가 (2026-09-10)

## 0. 요약

- E2 = C2 final 에서 시작, **Nemotron 3.5 streaming 인코더를 해동**(LR 1e-5, bf16 autocast, NeMo Triton 서브샘플링 커널 끔)하고 thinker LR 2e-5 · warmup 500 으로 8 코퍼스(EN 2.2 k h + KO 3.4 k h)를 3 epoch(14,721 step) 학습. 학습 파라미터 1,209 M.
- 같은 select 표본(dev-clean 50 / dev-other 50 / kspon-dev 300 / ah71-dev 300, seed 7)에서 **C2 대비 모든 셋이 개선**: δ=2 WER/CER 0.103 / 0.180 / 0.244 / 0.484 → **0.076 / 0.139 / 0.167 / 0.292**, δ=4 0.056 / 0.122 / 0.156 / 0.406 → **0.049 / 0.103 / 0.128 / 0.234**. 방출 지연은 C2 와 같은 수준(δ=2 p50 194–201 ms EN · 167 ms KO).
- 초반 불안정(D2·D4 의 step 2,000 붕괴)이 없다: 낮은 LR 과 인코더 적응이 함께 작용. 4 노드 32 GPU, 1.02 s/step, 학습 약 4.3 h(선점 4 회 requeue 포함 6.9 h).
- 판정: **E2 를 현재 최적 모델로 채택**. 로컬 실시간 데모(`experiments/live/`)도 E2 로 전환.

## 1. 설정

- 데이터: [[output-stage2-d2-final-eval]] §1 과 동일한 8 코퍼스(수정된 로더 — `src_offset_s` 보존, 항목 캐시 v2). rank 당 배치 EN 12 / KO 24(인코더 학습 시 활성화 메모리 실측 49 / 61 GB, 32 배치는 DDP 포함 OOM) → 유효 EN 384 / KO 768, steps/epoch 4,907.
- 학습(job 67757, hpc 4 노드): lr 2e-5(adapter 1e-3, encoder 1e-5), warmup 500 + cosine, 3 epoch = 14,721 step, 선점 4 회(checkpoint-2500·10000·14000 에서 재개), 노드 로컬 env 스테이징·인코더 캐시로 재시작당 로드 9–124 s.
- 이전 시도: E1(67542, D2 final 시작, BS 24/32) GPU OOM 반복 → 종료; E1(67645, BS 12/24) 71631 경계 발화(라벨이 파일 끝을 넘음)로 rank 크래시 → 로더에 이웃 항목 대체 추가 후 사용자가 D4 우선으로 취소.
- 산출물: `/soundai/Model/VAPASR/hf-E2/{final, eval/, eval-ckpts/checkpoint-14000, tb/}`. final 은 인코더 가중치를 safetensors 에 함께 저장(4.8 GB, `encoder_trainable: true`).

## 2. 학습 중 추세 (sentinel: dev-clean 10 / dev-other 10 스트림 / kspon-dev 100 발화, δ=2)

| step | dev-clean WER · viol80 · lat p50/p90 | dev-other WER · viol80 · lat p50/p90 | kspon-dev CER · viol80 · lat p50/p90 |
|---|---|---|---|
| 2000 | 0.107 · 0.006 · 169/215 ms | 0.189 · 0.009 · 172/225 | 0.203 · 0.052 · 131/207 |
| 4000 | 0.090 · 0.004 · 192/243 | 0.175 · 0.003 · 190/260 | 0.173 · 0.044 · 161/230 |
| 6000 | 0.101 · 0.003 · 193/257 | 0.197 · 0.005 · 189/252 | 0.181 · 0.044 · 157/232 |
| 8000 | 0.083 · 0.002 · 193/257 | 0.202 · 0.005 · 188/252 | 0.168 · 0.039 · 175/232 |
| 10000 | 0.097 · 0.002 · 192/258 | 0.189 · 0.005 · 186/244 | 0.170 · 0.041 · 168/232 |
| 12000 | 0.097 · 0.001 · 193/263 | 0.179 · 0.003 · 199/269 | 0.156 · 0.036 · 175/233 |
| 14000 | 0.091 · 0.001 · 199/264 | 0.172 · 0.003 · 208/267 | 0.157 · 0.038 · 175/232 |

- step 2,000 에 이미 C2 final(0.109 / 0.225 / 0.256)을 넘었고, 이후 KO 가 완만히 더 내려갔다(ah71-dev sentinel 0.296 → 0.275). 지연 p50 은 초반 130–170 ms 에서 190–210 ms 로 늘어 C2 규약과 같아졌다.
- 같은 데이터·같은 시작점에서 인코더를 동결하고 LR 4e-5 를 쓴 D4 는 step 2,000 에 0.469 / 0.566 / 0.460 으로 붕괴했다(D2 도 0.338 / 0.480 / 0.410). 인코더 해동 + LR 절반이 초반 붕괴를 없앤 것으로, 두 변수를 분리한 대조군(D4 완주)은 사용자 판단으로 생략했다.

## 3. select 평가 (dev-clean 50 / dev-other 50 스트림, kspon-dev 300 / ah71-dev 300 발화, seed 7, bias 0)

| 모델 · δ | dev-clean WER · lat p50 | dev-other WER · lat p50 | kspon-dev CER · lat p50 | ah71-dev CER · lat p50 |
|---|---|---|---|---|
| C2 final · δ=2 | 0.103 · 209 ms | 0.180 · 216 | 0.244 · 200 | 0.484 · 204 |
| D2 final · δ=2 | 0.126 · 195 | 0.212 · 216 | 0.306 · 162 | 0.524 · 154 |
| E2 14000 · δ=2 | 0.077 · 194 | 0.140 · 201 | 0.167 · 166 | 0.278 · 160 |
| **E2 final · δ=2** | **0.076** · 194 | **0.139** · 201 | **0.167** · 167 | **0.292** · 163 |
| C2 final · δ=4 | 0.056 · 357 | 0.122 · 371 | 0.156 · 361 | 0.406 · 364 |
| **E2 final · δ=4** | **0.049** · 353 | **0.103** · 359 | **0.128** · 326 | **0.234** · 317 |

- 참고: Nemotron 3.5 RNN-T 오프라인 기준(C2 보고서 §6) dev-clean 0.044 / dev-other 0.082 / kspon-dev 0.202. E2 δ=4 는 dev-clean 에서 RNN-T 에 근접(0.049)하고 KO 는 크게 앞선다(0.128).
- E2 의 KO viol80(δ=2 0.029 · ah71 0.049)은 C2(0.024)와 비슷하다. D2 의 조기 방출 습관은 없다.

## 4. 타이밍 스윕 (E2 final, sentinel 10/10/100 + ah71-dev 100, seed 7)

| δ · bias | dev-clean WER · viol80 · 지연 p50/p90/p99 | dev-other WER · viol80 · 지연 | kspon-dev CER · viol80 · 지연 | ah71-dev CER · viol80 · 지연 |
|---|---|---|---|---|
| 2 · 0 | 0.088 · 0.000 · 203/244/318 ms | 0.134 · 0.003 · 191/262/308 | 0.171 · 0.039 · 175/229/396 | 0.299 · 0.049 · 170/232/548 |
| 3 · 0 | 0.059 · 0.000 · 287/324/377 | 0.113 · 0.001 · 273/338/398 | 0.142 · 0.012 · 246/309/454 | 0.243 · 0.018 · 250/313/718 |
| 4 · 0 | **0.051** · 0.000 · 363/404/464 | **0.100** · 0.000 · 350/417/466 | **0.131** · 0.007 · 329/387/561 | **0.236** · 0.020 · 328/393/640 |
| 2 · 1 | 0.106 · 0.000 · 168/238/256 | 0.137 · 0.003 · 180/238/306 | 0.229 · 0.063 · 147/226/378 | 0.393 · 0.081 · 144/220/423 |
| 2 · 2 | 0.126 · 0.000 · 158/216/255 | 0.178 · 0.008 · 173/237/306 | 0.334 · 0.089 · 129/202/326 | 0.663 · 0.124 · 125/210/322 |

- δ 손잡이는 C2·D2 와 같은 기울기로 작동한다: δ 2 → 4 에 지연 p50 +160 ms, 오류 EN 약 40 %·KO 약 20 % 상대 감소. δ=3 이 지연 +80 ms 로 그 이득의 대부분을 얻는다(dev-clean 0.059, kspon-dev 0.142).
- next_bias 는 여전히 해롭다(지연 −35 ms 에 KO 오류 급증). E2 에서도 조기 방출 유도는 쓰지 않는다.
- tick p99 112–127 ms 로 C2(94–106 ms)보다 조금 길다. 인코더 가중치가 달라졌을 뿐 구조는 같으므로 로그인 노드 GPU 공유(다른 작업 100 % 점유) 영향으로 본다. 실시간 여유는 디코더 최적화 과제로 남긴다.

## 5. test 최종 수치 (표본 프로토콜 s300/u1000, seed 7)

dev 는 학습 중 sentinel·checkpoint 선정에 썼으므로 최종 수치는 test 로 낸다. 전체 test 는 모델·δ 당 오디오 ≈17 h(스트리밍 디코드는 실시간 속도로 순차)라 공유 로그인 노드에서 6–8 h/run 이 걸려, **seed 7 고정 표본**(LibriSpeech test-clean/test-other 각 300 스트림, KsponSpeech eval_clean/eval_other 각 1,000 발화, 71631-dev 1,000 발화)으로 보고한다. 전체 셋은 `slurm/s3_eval_suite.sbatch` 로 재현 가능. 같은 표본으로 dev 도 함께 재서 dev/test 차이를 본다.

| 모델 · δ | test-clean WER | test-other WER | kspon eval_clean CER | kspon eval_other CER | ah71-dev CER | (dev-clean / dev-other / kspon-dev, 같은 표본) |
|---|---|---|---|---|---|---|
| C2 final · δ=2 | 0.104 · 208 ms | 0.191 · 220 | 0.274 · 202 | 0.293 · 199 | 0.470 · 204 | 0.107 / 0.195 / 0.258 |
| **E2 final · δ=2** | **0.082** · 199 | **0.139** · 202 | **0.158** · 167 | **0.184** · 163 | **0.299** · 163 | 0.076 / 0.141 / 0.171 |
| C2 final · δ=4 | 0.058 · 364 | 0.123 · 374 | 0.187 · 361 | 0.196 · 361 | 0.385 · 361 | 0.056 / 0.124 / 0.171 |
| **E2 final · δ=4** | **0.054** · 357 | **0.107** · 361 | **0.125** · 326 | **0.136** · 322 | **0.246** · 320 | 0.052 / 0.105 / 0.127 |

(각 칸: 오류율 · 방출 지연 p50 ms. viol80: EN 0.2–0.3 %, KO 3–5 %.)

- test 는 dev 와 같은 그림이다: E2 가 C2 대비 test-clean −21 %, test-other −27 %, kspon eval_clean −42 %, eval_other −37 %(상대), 자유대화 −36 %. dev 와 test 의 차이는 EN ±0.01, KO 0.01–0.02 로 dev 기반 선택의 낙관 편향은 크지 않다.
- δ=4 에서도 E2 가 C2 를 모든 셋에서 앞선다: test-clean -7 %, test-other -13 %, kspon eval_clean -33 %, eval_other -31 %, 자유대화 -36 %(상대). δ=2 의 상대 개선(−21/−27/−42/−37/−36 %)과 같은 그림.
- δ=4(320 ms): test-clean 0.054 / test-other 0.107 / kspon eval 0.125·0.136 / 자유대화 0.246 — δ=2 대비 EN 오류 −34 %·KO −20~25 %(상대), 지연 p50 +150 ms. dev 표본과의 차이도 ±0.01.
- KsponSpeech eval_other(0.184)가 eval_clean(0.158)보다 어렵고, 방송·대화 도메인 추가에도 자유대화(0.299)는 여전히 가장 어렵다.

## 6. 판정과 다음

1. E2 채택. 로컬 데모·향후 실험의 기준 모델.
2. 남은 격차: δ=2 에서 EN dev-other 0.139 는 RNN-T 오프라인(0.082)과 아직 멀다. δ=3–4 가 실사용 기본값 후보(지연 +100–150 ms).
3. 후속 후보: (a) E3 = E2 에서 1–2 epoch 추가(LR 1e-5), (b) `<NEXT_AUDIO>` 가중 상향으로 δ=2 정확도 개선, (c) 디코더 스텝 최적화(static cache/graph)로 실시간 여유 확보.

## 7. 재현

- 학습: `sbatch --partition=hpc --nodes=4 --requeue --exclude=slurmmxch200v5-hpc-52 --export=ALL,RUN=E2,INIT=/soundai/Model/VAPASR/hf-C2/final,TRAIN_ENCODER=1,LR_ENCODER=1e-5,LR=2e-5,WARMUP=500,EPOCHS=3,TRAIN=librispeech-960:swbd-train:voxpopuli-train:yodas-en129:kspon-full:nikl-1000:aihub71631-train:aihub-bc-train,BS_EN=12,BS_KO=24,EVAL_EVERY=2000,SAVE_EVERY=500,DEV_EXTRA=ah71-dev=aihub71631-dev:dev:utt slurm/s3_train_hf.sbatch`
- 평가: `experiments/s3_d2_eval.sh`(R=hf-E2) + `experiments/s3_test_eval.sh`, 또는 한 번에 `sbatch --partition=hpc --export=ALL,RUN=E2 slurm/s3_eval_suite.sbatch`.
