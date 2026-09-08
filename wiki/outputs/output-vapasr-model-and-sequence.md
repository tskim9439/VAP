---
type: output
status: active
created: 2026-09-08
updated: 2026-09-08
summary: 현행 VAP-ASR 스트리밍 모델(Nemotron [56,0] 인코더 → adapter → Qwen3-ASR thinker)의 구조, 80 ms 청크 인터리브 시퀀스·라벨·손실 규약, 스트리밍 디코드와 학습 설정을 코드 기준으로 정리한 보고서
sources:
  - [[output-interleaved-streaming-slm-architecture]]
  - [[output-stage1-mono-pilot]]
  - [[output-asr-tn-v1-spec]]
  - [[source-hf-trainer-migration]]
  - [[source-stage1-mono-run-ab]]
---

# VAP-ASR 모델 구조와 시퀀스 규약 (2026-09-08 현행)

정본 코드: `vapasr/hf/modeling_vapasr.py`(HF 판) = `vapasr/uslm/mono_model.py`(원본, 수치 동일), 시퀀스 `vapasr/data/interleave.py` + `vapasr/uslm/mono_data.py`, 학습 `experiments/s3_train_hf.py`. 설계 배경은 `plans/stage1-mono-pilot.md` §4–§6.

## 1. 한눈에 보기

```text
16 kHz mono wav ─▶ Nemotron 3.5 ASR streaming 0.6B 인코더 (FastConformer, [56,0], 동결)
                   80 ms 프레임 × 1024 차원, K = round(길이 × 12.5)
                ─▶ adapter (LayerNorm → Linear 1024→2048 → GELU → Linear 2048→1024, 학습)
                ─▶ Qwen3-ASR-0.6B thinker 텍스트 디코더 (28 층, full FT)
                   입력 = [prefix] + 청크마다 ( [AUDIO_k] 자리 ← adapter 출력, 텍스트 토큰…, <NEXT_AUDIO> )
                   출력 = 텍스트 토큰 + <NEXT_AUDIO>(= "이 청크는 끝") 를 자기회귀로 예측
```

| 구성 | 크기 | 상태 |
|---|---|---|
| 인코더 Nemotron (24 층 FastConformer, d 1024, 8 헤드, dw-striding ×8) | ≈0.6 B | 동결(fp32, autocast 밖). `set_trainable` 로 해동 가능 |
| adapter | 4.2 M | 학습(증류 초기화, lr 1e-3) |
| thinker Qwen3-ASR-0.6B (28 층, hidden 1024, 16 헤드/8 KV, head 128, FFN 3072, vocab 151,936, lm_head tied) | 596.0 M | 전체 학습(fp32 master + bf16 autocast, lr 6e-5) |
| 학습 파라미터 합계 | **600.2 M** | |

## 2. 인코더: 80 ms 청크가 나오는 곳

- 전처리 128-mel (25 ms 창 / 10 ms 이동) → dw-striding 서브샘플링 ×8 → **프레임 1 개 = 80 ms (12.5 Hz)**. 이것이 시퀀스의 "청크" 단위다.
- 주의 창 `att_context_size=[56, 0]`(chunked_limited): 왼쪽 56 청크(4.48 s), **오른쪽 0** → 인과적. 따라서 전체 파형을 한 번에 인코딩한 결과가 스트리밍 결과와 같고, 학습에서는 온라인으로(캐시 없이) 배치마다 인코더를 돌린다(`vapasr/features/online.py`).
- 스트림 길이 T 초 → K = round(T × 12.5) 프레임(인코더 패딩으로 1–2 프레임 길면 자르고 짧으면 0 패딩).

## 3. 텍스트 쪽: thinker 와 특수 토큰

Qwen3-ASR 의 오디오 타워는 쓰지 않고(삭제) thinker 의 텍스트 디코더만 쓴다. 오디오는 `<|audio_pad|>`(id 151676) 자리의 임베딩을 adapter 출력으로 **교체**해 넣는다(`build()`: `where(is_audio, adapter(feat[chunk_of]), embed(ids))`).

tokenizer 에 추가한 특수 토큰 12 개(임베딩 행렬의 여유 행을 쓰므로 resize 없음, 행은 기존 임베딩 평균 + 0.02·N(0,1) 로 초기화):

| 토큰 | id | 역할 |
|---|---|---|
| `<NEXT_AUDIO>` | 151705 | "이 청크에서 낼 토큰이 더 없다" — RNN-T 의 blank 에 해당. **라벨**(손실) |
| `<EMPTY_AUDIO>` | 151706 | 스트림 종료 후 오디오 자리 대신 넣는 **입력** (손실 없음) |
| `<DELAY_1..8>` | 151709–151716 | prefix 끝에서 방출 지연 δ(청크 수)를 조건화 |
| `<SPK_A>`, `<SPK_B>` | 151707, 151708 | 2-화자용 예약(mono 에서는 미사용, 디코드 시 차단) |

디코드 시 차단 토큰(`blocked`): `<|im_start|>`, `<|im_end|>`, audio start/end/pad, `<asr_text>`, `<EMPTY_AUDIO>`, `<SPK_*>`, `<DELAY_*>` — 모델이 구조 토큰을 텍스트로 내지 못하게 logits 를 −∞ 로 막는다.

## 4. 시퀀스 규약

### 4.1 형태
```text
prefix   <|im_start|>system⏎<|im_end|>⏎<|im_start|>assistant⏎language {English|Korean}<asr_text><DELAY_δ>     (입력만)
chunk 0  [AUDIO_0]  tok tok        <NEXT_AUDIO>
chunk 1  [AUDIO_1]                 <NEXT_AUDIO>        ← 방출 없음(청크의 75–80 %)
…
chunk K-1 [AUDIO_K-1] tok          <NEXT_AUDIO>
flush    <EMPTY_AUDIO> tok tok     <NEXT_AUDIO>        ← δ 때문에 스트림 끝을 넘긴 토큰(남은 것이 있을 때만)
flush    <EMPTY_AUDIO>             <NEXT_AUDIO>        ← 빈 라운드 = "남은 것 없음", 항상 마지막에 하나
```
prefix 는 EN/KO 공통이고 `language …` 만 다르다. `[AUDIO_k]` 는 청크당 정확히 1 개(mono).

### 4.2 토큰이 어느 청크에 붙는가
- 텍스트는 asr-tn-v1 정규화 lexical 텍스트(대소문자·구두점 제거, 숫자 발음형; [[output-asr-tn-v1-spec]]). Qwen3ForcedAligner 가 준 **토큰 종료 시각** t_end 를 쓴다.
- 배정 규칙: `k = ⌊t_end / 80 ms⌋ + δ`. 즉 토큰은 자기 발화가 끝난 청크로부터 δ 청크 뒤에 방출된다. 학습에서는 δ ∈ {2, 3, 4, 6} 을 스트림마다 무작위로 뽑아 `<DELAY_δ>` 로 조건화하고, 평가는 δ = 2(160 ms).
- 청크당 토큰 수 상한(M)은 두지 않는다(2026-09-07 제거). 같은 청크에 여러 토큰이 오면 종료 시각 순으로 모두 낸다.
- k ≥ K 인 토큰(스트림 끝 넘김)은 flush 라운드로 간다. M 이 없으므로 flush 는 "남은 토큰 전부 한 라운드" + "빈 라운드" 로 끝난다.

### 4.3 라벨과 손실
- 위치별 라벨: 텍스트 토큰과 `<NEXT_AUDIO>` 만 라벨(다음 토큰 예측). prefix·`[AUDIO_k]`·`<EMPTY_AUDIO>` 위치는 −100.
- `<NEXT_AUDIO>` 가 라벨의 70–85 % 라 불균형 → 가중 CE: `loss = Σ w·CE / Σ w`, w = next_weight(EN 0.3, KO 0.15) at `<NEXT_AUDIO>`, 1 at 텍스트.
- lm_head 는 **라벨 위치에서만** 계산한다(전 위치 fp32 logits 는 KO bs 48 에서 100 GB 를 넘어 OOM).
- 실제 밀도: LibriSpeech 0.22 토큰/청크, KsponSpeech 0.27 토큰/청크(정렬 캐시 기준) → 대부분의 청크가 `[AUDIO_k] <NEXT_AUDIO>` 두 자리.

### 4.4 실제 예 (kspon-dev `ks-dev-utt-KsponSpeech_620663`, 2.6 s, K=32, δ=2, 시퀀스 길이 91)
`lexical_text`: 그러고 오퍼가 나만 있는 게 아니야

| 청크 k | 시간창 (ms) | 위치별 `입력 → 예측(라벨)` | 토큰 종료 시각 (ms) | 방출 지연 |
|---:|---|---|---|---|
| 0–11 | 0–960 | `[AUDIO_k]`→`<NEXT>` × 12 | – | – |
| 12 | 960–1040 | `[AUDIO_12]`→`그` · `그`→`러` · `러`→`고` · `고`→`<NEXT>` | 807 ×3 | +233 ms |
| 13–15 | 1040–1280 | `[AUDIO_k]`→`<NEXT>` × 3 | – | – |
| 16 | 1280–1360 | `[AUDIO_16]`→`오`⟨1/4⟩ → ⟨2/4⟩ → ⟨3/4⟩ → `가` → `<NEXT>` (바이트 조각 토큰) | 1127 ×4 | +233 ms |
| 20 | 1600–1680 | `[AUDIO_20]`→`나` · `나`→`만` · `만`→`<NEXT>` | 1447 ×2 | +233 ms |
| 22 / 23 | 1760–1920 | `있는` / `게` 각각 1 토큰 + `<NEXT>` | 1607 / 1687 | +233 ms |
| 26 | 2080–2160 | `[AUDIO_26]`→`아니` · `아니`→`야` · `야`→`<NEXT>` | 1927 ×2 | +233 ms |
| 27–31 | 2160–2560 | `[AUDIO_k]`→`<NEXT>` × 5 | – | – |
| flush | 스트림 끝 | `<EMPTY_AUDIO>`→`<NEXT>` (남은 것 없음 → 종료) | – | – |

("방출 지연" = (k+1)·80 ms − t_end. δ=2 이면 이론값 160–240 ms.) 전체 표는 `python experiments/s1_show_sequence.py --manifest kspon-dev --mode utt --delay 2 --dur 2.6` 으로 재현.

## 5. 학습 데이터·배치·최적화 (run C2 기준)

| 셋 | 스트림 수 | 시간 | 길이 중앙값 / p90 / 최대 | 토큰/스트림 중앙값 |
|---|---|---|---|---|
| librispeech-960 (발화 연결 스트림, 무음 삽입) | 126,511 | 1,033 h | 30.1 / 33.6 / 52.5 s | 82 |
| kspon-full (발화 = 스트림) | 619,203 | 1,186 h | 5.4 / 12.7 / 32.8 s | 16 |

- 배치: 길이(K) 버킷, 언어별 크기(EN 12 / KO 48 per GPU), optimizer step 마다 EN/KO 교대. 16 GPU 면 유효 배치 EN 192 / KO 768, epoch = 1,464 step.
- 최적화: AdamW, wd 0.01(임베딩 0), warmup 500 + cosine, grad clip 1.0, thinker lr 6e-5(16 GPU) · adapter 1e-3, gradient checkpointing, bf16 autocast(마스터 fp32), Liger 커널(RMSNorm·SwiGLU·RoPE, +24 % 처리량), DDP(NCCL/IB, 대기·수집은 gloo). 0.93–1.5 s/step(2–8 노드).
- 정렬·정규화 관문: 정렬 산출물의 textnorm fingerprint 가 코드와 major 가 다르면 학습이 거부된다([[decision-asr-tn-v1-freeze]]).

## 6. 스트리밍 디코드 (`stream_decode`)

1. prefix 를 KV cache 에 넣는다. 청크 k 마다 adapter(feat_k) 임베딩 1 개를 넣고 greedy 로 토큰을 낸다. `<NEXT_AUDIO>` 가 나오면 그 청크 종료(`<NEXT_AUDIO>` 임베딩도 cache 에 넣고 다음 청크로).
2. 폭주 방지: 청크당 8 토큰, 스트림 전체 6·K + 64 토큰을 넘으면 `<NEXT_AUDIO>` 강제(`forced` 로 집계). `next_bias` 로 `<NEXT_AUDIO>` logit 을 낮춰 방출을 앞당길 수 있다(평가는 0).
3. 스트림 끝: `<EMPTY_AUDIO>` 를 입력해 flush 라운드를 돌리고, 토큰 없이 `<NEXT_AUDIO>` 가 나오면 종료(최대 8 라운드).
4. 비용: 청크당 forward 1 회 + 방출 토큰 수 만큼 추가 forward. H200 에서 tick p50 ≈ 60 ms / 80 ms 청크(p99 100–150 ms 로 실시간 관문은 아직 미달).
5. 지표: WER/CER(정규화 텍스트), 방출 지연 = (k+1)·80 ms − t_end 를 참조와 정렬(difflib)해 p50/p90/p99, `viol80` = 참조보다 80 ms 이상 이른 방출 비율, tok/chunk, forced 비율, tick ms.

## 7. HF 패키징과 파일 지도

| 역할 | 파일 |
|---|---|
| 설정·모델 | `vapasr/hf/configuration_vapasr.py`, `vapasr/hf/modeling_vapasr.py` (`from_qwen` 새 모델 / `from_legacy` 기존 ckpt / `from_pretrained`) |
| 시퀀스 조립 | `vapasr/data/interleave.py::build_interleaved`, `vapasr/uslm/mono_data.py::build_mono_sequence`·`MonoStreamDataset`·`collate_streams` |
| 학습 | `vapasr/hf/trainer.py`(VapAsrTrainer, PreemptCallback), `vapasr/hf/data.py`, `experiments/s3_train_hf.py`, `slurm/s3_train_hf.sbatch` |
| 원본(수치 동일) | `vapasr/uslm/mono_model.py`, `experiments/s1_train_mono.py`, `slurm/s2_train.sbatch` |
| 산출물 | `/soundai/Model/VAPASR/hf-<RUN>/{checkpoint-<step>/, final/, eval/, tb/, DONE}` |

## 8. 현재 수치와 한계

- 파일럿 230 h(8 GPU, [[source-stage1-mono-run-ab]]): full FT 가 dev-clean WER 0.169 / dev-other 0.237 / kspon-dev CER 0.438 (RNN-T 대조군 0.044 / 0.082 / 0.202). 1,930 h run(C2, 2 노드)이 진행 중이며 sentinel 은 2,000 step 마다.
- 한계: (1) 인코더 동결 — 해동 시 KO bs 48 OOM 이라 bs 축소 또는 인코더 checkpointing 필요, (2) 디코드 tick p99 가 80 ms 를 넘음 — 커널·캐시 최적화 과제, (3) δ 조건화는 학습됐지만 지연 분포 제어(next_bias·δ) 는 파일럿 수준 검증만 됐음.
