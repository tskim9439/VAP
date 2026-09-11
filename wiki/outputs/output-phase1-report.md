---
type: output
status: active
created: 2026-09-10
updated: 2026-09-11
summary: Phase 1(2026-09-03 → 09-10) 종합 보고 — 80 ms 인터리브 스트리밍 ASR(Nemotron 인코더 → adapter → Qwen3-ASR thinker)의 개념·모델 구조·시퀀스 규약·학습 데이터(EN 2.2 k h + KO 3.4 k h)·구현(HF Trainer, 스키마, 인프라)·실험 계보(파일럿 → C2 → D/D2/D4 → E2)와 결과(E2 select δ=2 0.076/0.139/0.167/0.292), 실시간 데모(MLX)까지
sources:
  - [[output-interleaved-streaming-slm-architecture]]
  - [[decision-asr-backbone]]
  - [[decision-mono-input]]
  - [[output-vapasr-model-and-sequence]]
  - [[output-stage1-mono-pilot]]
  - [[output-stage2-c2-final-eval]]
  - [[output-stage2-d2-final-eval]]
  - [[output-stage2-e2-final-eval]]
  - [[output-dataset-schema-v1]]
  - [[source-hf-trainer-migration]]
  - [[source-mxc-model-load-latency]]
  - [[source-soulx-duplug]]
  - [[source-conversation-corpora]]
---

# Phase 1 종합 보고: 인터리브 스트리밍 ASR (2026-09-03 → 09-10)

후속 실행 제안: [[output-phase2-streaming-asr-diarization-plan]] — 최대 2화자 mono 입력의 화자별 전사·겹침 처리·의미 기반 turn-taking. 본문의 Phase 1 가능성 판정과 기존 PLAN의 엄격 관문 충족 여부를 구분하여 이월한다.

## 0. 한 줄 요약

RNN-T 없이 **80 ms 오디오 청크와 텍스트 토큰을 하나의 causal LLM 시퀀스에 교차**시켜 실시간 전사를 내는 모델(IS-SLM 의 ASR 부분)을 설계·구현·학습했고, 한국어/영어 5.6 k h 로 인코더까지 해동한 **E2** 가 같은 인코더의 RNN-T 에 δ=4(320 ms) 조건에서 영어는 근접·한국어는 앞서는 수준에 도달했다. 로컬 Mac 에서 실시간(RTF 0.8)으로 도는 데모까지 갖췄다. Phase 2 는 이 표현 위에 turn-taking(VAP) 을 얹는 단계다.

| 모델 (select δ=2 / δ=4) | dev-clean WER | dev-other WER | kspon-dev CER | 71631-dev CER(자유대화) |
|---|---|---|---|---|
| Nemotron RNN-T 오프라인 (대조군) | 0.044 | 0.082 | 0.202 | – |
| 파일럿(230 h, 2026-09-06) δ=2 | 0.169 | 0.237 | 0.438 | – |
| C2(1,930 h) δ=2 / δ=4 | 0.103 / 0.056 | 0.180 / 0.122 | 0.244 / 0.156 | 0.484 / 0.406 |
| **E2(5.6 k h, 인코더 해동)** δ=2 / δ=4 | **0.076 / 0.049** | **0.139 / 0.103** | **0.167 / 0.128** | **0.292 / 0.234** |
| E2 · **test**(표본 s300/u1000) δ=2 | test-clean **0.082** | test-other **0.139** | kspon eval_clean/other **0.158 / 0.184** | 0.299 |
| C2 · test(표본) δ=2 | 0.104 | 0.191 | 0.274 / 0.293 | 0.470 |
| E2 · test(표본) δ=4 | **0.054** | **0.107** | **0.125 / 0.136** | 0.246 |

## 1. 문제와 개념

### 1.1 왜 인터리브 스트리밍 SLM 인가
프로젝트의 목표는 "한 모델이 전사·화자·turn 을 낸다" 이다([[streaming-conversational-projection-asr]]). 전통적 파이프라인(VAD → RNN-T → turn 분류기)은 지연이 누적되고 의미 정보를 turn 예측에 쓰기 어렵다. Meta Muse Voice Transcribe(closed) 가 보여준 "80 ms soft token + 텍스트 토큰 교차" 방식을 열린 가중치로 재구성하기로 했고([[output-interleaved-streaming-slm-architecture]]), 2026-09-04 에 **IS-SLM 단일 주력**으로 확정했다([[decision-target-architecture]]). 2026-03 의 SoulX-Duplug([[source-soulx-duplug]])가 같은 계열(160 ms 청크, 상태 토큰)로 독립 검증됐지만, 그들은 작은 청크의 스트리밍 ASR 을 포기하고 외부 ASR 을 teacher-forcing 했다. 우리는 스트리밍 ASR 자체를 LLM 이 내게 하는 쪽을 택했고, 그 난도를 `<DELAY_δ>` 손잡이로 다룬다.

### 1.2 핵심 개념
- **청크 = 80 ms(12.5 Hz)**: 인코더 서브샘플링 ×8 의 프레임 하나. 시퀀스의 시간 단위이자 latency 회계 단위([[streaming-causality-and-latency-budget]]).
- **인과성**: 인코더 주의 창 `[56, 0]`(왼쪽 4.48 s, 오른쪽 0) → 전체 인코딩 = 스트리밍 인코딩. 절단 실험으로 lookahead ≤ 80 ms 확인([[output-encoder-causality-audit]]). Qwen3 AuT 는 비인과(420 ms)라 인코더 후보에서 탈락.
- **`<NEXT_AUDIO>` = blank**: RNN-T 의 blank 처럼 "이 청크에서 낼 토큰 없음" 을 뜻하는 라벨 토큰. 청크의 75–80 % 가 이것뿐이라 손실 가중(EN 0.3 / KO 0.15)으로 균형을 맞춘다.
- **δ 조건 토큰 `<DELAY_δ>`**: 토큰을 발화 종료 청크로부터 δ 청크 뒤에 방출하도록 학습(δ∈{2,3,4,6} 무작위). 추론 때 δ 를 고르면 지연–정확도가 즉시 바뀐다(δ 2→4: 지연 +160 ms, EN 오류 −40 %). 이 프로젝트의 고유한 손잡이.
- **강제 정렬 기반 타깃**: 토큰의 종료 시각(Qwen3ForcedAligner)으로 청크 배정 `k = ⌊t_end/80 ms⌋ + δ`. 정렬은 asr-tn-v1 정규화 텍스트 위에서 만든다.
- **mono 입력**([[decision-mono-input]]): 배포 조건이 마이크 하나이므로 화자별 채널 입력은 폐기. 분리 채널 코퍼스는 라벨·혼합 합성에만.
- **평가 3 축**: WER/CER, 방출 지연(p50/p90/p99, 참조보다 80 ms 이상 이른 방출 비율 viol80), 실시간성(청크당 처리 시간 tick, 80 ms 예산).

## 2. 모델 구조

```text
16 kHz mono ─▶ Nemotron 3.5 ASR streaming 0.6B 인코더 (FastConformer 24 층, d 1024, [56,0]) ─▶ 80 ms × 1024
            ─▶ adapter (LayerNorm → Linear 1024→2048 → GELU → Linear 2048→1024, 4.2 M)
            ─▶ Qwen3-ASR-0.6B thinker 텍스트 디코더 (Qwen3 28 층, hidden 1024, 16/8 헤드, FFN 3072, vocab 151,936, lm_head tied)
            입력 = prefix + 청크마다 [AUDIO_k](adapter 출력으로 교체) · 텍스트 토큰… · <NEXT_AUDIO>
```
- 백본 결정([[decision-asr-backbone]]): 인코더 Nemotron(스트리밍 완성도·lookahead 검증), 디코더 Qwen3-ASR thinker(Apache 2.0, 12.5 Hz 정합, 오디오 타워는 버리고 텍스트 디코더만). 관문 U0.5(adapter bridge, [[output-uslm-u05-adapter-bridge]])에서 adapter 가 병목이 아님을 확인.
- 특수 토큰 12 개(`<NEXT_AUDIO>`, `<EMPTY_AUDIO>`, `<DELAY_1..8>`, `<SPK_A/B>`)는 vocab 여유 행을 쓰고, 디코드 때 구조 토큰은 logits −∞ 로 차단.
- 학습 파라미터: C2/D 계열 600.2 M(인코더 동결), E2 1,209 M(인코더 해동, LR 1e-5).
- 세부 규약·라벨·손실·실제 예는 [[output-vapasr-model-and-sequence]].

### 2.1 시퀀스 규약(요약)
```text
prefix  <|im_start|>system⏎<|im_end|>⏎<|im_start|>assistant⏎language {English|Korean}<asr_text><DELAY_δ>
chunk k [AUDIO_k] tok… <NEXT_AUDIO>          (k = 0..K-1, 대부분 tok 없음)
flush   <EMPTY_AUDIO> tok… <NEXT_AUDIO>      (δ 때문에 끝을 넘긴 토큰) + 빈 라운드 1 회
```
손실은 텍스트 토큰과 `<NEXT_AUDIO>` 위치에만, lm_head 는 라벨 위치에서만 계산(메모리). 디코드는 청크마다 `[AUDIO_k]` 를 넣고 `<NEXT_AUDIO>` 가 나올 때까지 greedy(청크당 8 토큰·전체 6·K 토큰 상한으로 폭주 방지).

## 3. 학습 데이터

| 언어 | 코퍼스 | 발화 시간 | 비고 |
|---|---|---|---|
| EN | LibriSpeech 960 | 1,033 h | 챕터 내 발화 연결 스트림(20–30 s) |
| EN | Switchboard | 290 h | 전화 대화, 발화=스트림 |
| EN | VoxPopuli | 521 h | EU 의회, tar 멤버 직접 읽기 |
| EN | Granary-YODAS en129 | 334 h | YouTube 의사 라벨, num2words |
| KO | KsponSpeech 전체 | 1,189 h | 발화=스트림, `(철자)/(발음)` 이중표기 규약 |
| KO | NIKL 일상대화 1,000 h 표본 | 1,451 h | 2021–2025, original_form |
| KO | AI Hub 71631 자유대화 | 159 h | stereo 대화 wav, 화자=채널(에너지 VAD 로 판정), `path#chN` + `src_offset_s` |
| KO | AI Hub 031/033 방송 원음 | 578 h | zip 멤버, 발음전사 |
| 합계 | | **≈ 5.6 k h** | dev: LibriSpeech dev-clean/other, KsponSpeech dev, 71631 VS_02(실외) held-out |

- 제외: MNSC(오디오 부재 90 %), AI Hub 98(라벨 없음), Earnings-22(평가 전용). 
- **텍스트 정규화 asr-tn v1.0 → v1.3**: 사전학습 모델 출력 표기에 맞춘 단일 규약(EN 소문자·숫자 단어, KO 숫자 한글 읽기·간투사 유지). 코퍼스별 파서(nikl, aihub71631, aihubbc)와 11+ 회귀 테스트.
- **데이터 스키마 v1**([[output-dataset-schema-v1]]): manifest `streams.jsonl` + 카드 `dataset.json`, 정렬 `parts/*.jsonl` + `align.json`, textnorm/tokenizer 지문 관문, id 중복 검사. 아카이브(tar/zip) 멤버·다채널·원본 오프셋을 경로 형식으로 표현.
- 정렬: 2 노드 96 shard 로 5 개 신규 manifest 정렬 완료(빈 레코드 ≤1 %).

## 4. 구현

- **모델·학습**: `vapasr/hf/` — `VapAsrForStreamingASR(PreTrainedModel)`, `VapAsrTrainer(Trainer)`(언어별 라운드로빈 버킷 배치, 분산 스트리밍 평가, 선점 콜백), Liger 커널(+24 % 처리량), TensorBoard, 재개·requeue. 원본 `vapasr/uslm/` 과 수치 패리티 검증([[source-hf-trainer-migration]]).
- **추론**: `vapasr/hf/infer.py`(오프라인 스트리밍 디코드·IPykernel 데모), `vapasr/hf/live.py`(cache-aware 스트리밍 인코더 + 청크 단위 LiveSession, 오프라인과 프레임별 동일), `vapasr/hf/live_mlx.py`(Apple silicon: thinker 를 mlx-lm 으로), `experiments/live/`(aiohttp WebSocket 서버 + 브라우저 마이크/파일 UI, TLS).
- **인프라(mxc)**: 4–8 노드 SLURM(선점 잦음), NFS 위 conda env 의 import 지연(16 분) → 노드 로컬 env 스테이징 + 인코더 캐시로 재시작 25 분 → 1–3 분([[source-mxc-model-load-latency]]); NeMo `sync_max_audio_length` 교착 해결; 컨테이너 `sa_tskim_fd`.
- **로컬(M4 Mac)**: 사내 TLS 검사용 CA 번들, `vapasr-local` env(torch MPS + NeMo + mlx-lm), E2 체크포인트 로컬 실행.

## 5. 실험 계보와 결과

| run | 시작점 | 데이터 | 설정 | 결과(select δ=2: dev-clean / dev-other / kspon-dev) | 판정 |
|---|---|---|---|---|---|
| 파일럿(Stage 1) | random adapter + LoRA/full FT | LS-100 + Kspon-100 (230 h) | 6 k step | 0.169 / 0.237 / 0.438 | 학습 가능성 확인, full FT > LoRA([[output-stage1-mono-pilot]]) |
| C2 | C step-500 | LS-960 + Kspon 전체 (1,930 h) | lr 6e-5, 30 epoch, 32 GPU | 0.103 / 0.180 / 0.244 | 데이터 8 배 → EN −39 %, KO −44 %. δ=4 에서 RNN-T 근접([[output-stage2-c2-final-eval]]) |
| D | C2 | + swbd, nikl | lr 4e-5 | (step 7,500 중단) | KO 초반 후퇴 후 회복 |
| D2 | C2 | 8 코퍼스 5.6 k h | lr 4e-5, 10 epoch | 0.126 / 0.212 / 0.306 | **71631 오디오 조립 버그**(`src_offset_s` 누락) + 초반 붕괴. 불채택([[output-stage2-d2-final-eval]]) |
| D4 | C2 | 8 코퍼스(수정 로더) | D2 와 동일 | step 2,000 sentinel 0.469 / 0.566 / 0.460 후 취소 | 붕괴가 데이터 결함이 아니라 레시피(LR 정점) 문제임을 입증 |
| **E2** | C2 | 8 코퍼스 | **인코더 해동** LR 1e-5, thinker 2e-5, 3 epoch | **0.076 / 0.139 / 0.167** (71631-dev 0.292) | 채택([[output-stage2-e2-final-eval]]) |

- **타이밍**: 모든 run 에서 δ 가 지연을 정확히 제어(δ 2/3/4 → p50 ≈ 200/280/360 ms), viol80 ≤ 0.5 %(KO 2–5 %), next_bias 는 항상 해로움. 실시간성은 H200 에서 tick p99 95–120 ms 로 예산(80 ms) 초과였으나 데모 최적화로 해결(§6).
- **실패에서 배운 것**: (a) 새 코퍼스 점검은 manifest 가 아니라 학습 데이터셋 객체 경로로(71631 사례); (b) 새 도메인을 크게 넣을 때 thinker LR 4e-5 는 초반 붕괴(조기 방출) — 2e-5 + 인코더 해동이 안정; (c) 선점 파티션에서는 500 step 저장 + 노드 스테이징이 손실을 줄인다.

## 6. 실시간 데모와 추론 최적화

| 구성(M4 Mac, 청크 80 ms 당) | 인코더 | 디코더 | 합계 p50 | RTF |
|---|---|---|---|---|
| PyTorch MPS 초기 | 44 ms | 53 ms | 97 ms | 1.25 |
| `<NEXT_AUDIO>` 스텝 병합 | 44 | 31 | 75 | 1.10 |
| **thinker MLX fp16(현재)** | 39–44 | 16 | 56 | **0.82** |
| thinker MLX int8 | 43 | 12 | 55 | 0.76 |

- 브라우저 마이크 → WebSocket(Int16 16 kHz, 80 ms) → 서버(인코더 cache-aware 스트리밍 + thinker KV cache) → 청크마다 전사 이벤트. δ 기본 4, UI 에서 변경. 파일 시뮬레이션·입력 장치 선택·진단 표시 포함.
- MLX thinker 는 torch 와 logits 차이 ≤ 0.13, 전사 동일. 남은 병목은 인코더(24 층 Conformer 를 프레임마다 호출).

## 7. Phase 1 판정과 Phase 2 로의 이월

**판정**: 인터리브 스트리밍 ASR 은 (1) 학습 가능하고, (2) 데이터·인코더 적응으로 RNN-T 수준에 접근하며(δ=4 dev-clean 0.049 vs 0.044, kspon-dev 0.128 vs 0.202), (3) 지연을 토큰 하나로 제어할 수 있고, (4) 소비자 하드웨어에서 실시간으로 돈다. 모델이 텍스트를 내는 방식이 "언제 말이 끝났는가" 를 이미 암묵적으로 배우므로, Phase 2 의 turn 예측 헤드가 얹힐 표현으로 충분하다.

**미결·이월**
1. test 셋 보고: δ=2 표본 test(s300/u1000) 반영 완료(E2 0.082 / 0.139 / 0.158 / 0.184, dev 와 ±0.02 이내). E2 δ=4 표본 test 반영 완료(0.054 / 0.107 / 0.125 / 0.136). C2 δ=4 는 측정 중, 전체 셋(SLURM)은 추가 예정.
2. δ=2(160 ms) 정확도: dev-other 0.139 는 RNN-T 오프라인 0.082 와 격차. `<NEXT_AUDIO>` 가중 상향·추가 epoch(E3)·increased context 검토.
3. 인코더 실시간성(MLX 포팅) 과 서버 GPU 에서의 tick p99.
4. Phase 2: 혼합 mono 대화(3-1 전사 → 3-2 `<SPK_A/B>` → 3-3 overlap), VAP·hazard 헤드([[turn-taking-objectives]]), TurnBench 평가([[turn-taking-evaluation-protocol]]). SoulX-Duplug 의 상태 토큰 5 종·LLM 라벨링 파이프라인 참고. → 계획: [[output-phase2-streaming-asr-diarization-plan]](정본), 실행 요약 [[output-phase2-plan]].

## 8. 산출물 위치
- 모델: `/soundai/Model/VAPASR/hf-{C2,D2,E2}/final`, 로컬 `~/Desktop/VAPKT-models/hf-E2-final{,-thinker-mlx}`.
- 데이터: `/soundai/users/tskim/VAPKT-data/data/manifests/<name>/`, 정렬 `align-asr-tn-v1/`.
- 코드: `vapasr/`(data·features·hf·uslm), `experiments/`(s1_build_manifest, s1_align, s3_train_hf, s3_d2_eval, s3_test_eval, live/, probe_*), `slurm/`, `scripts/`.
- 보고서: [[output-stage2-c2-final-eval]], [[output-stage2-d2-final-eval]], [[output-stage2-e2-final-eval]], [[output-vapasr-model-and-sequence]], [[output-dataset-schema-v1]].
