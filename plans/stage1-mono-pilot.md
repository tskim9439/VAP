# Stage 1 — 단일 화자 스트리밍 ASR 파일럿: 실행 계획

> `PLAN.md` Stage 1 의 실행 문서. 데이터 구성 · 입력 시퀀스 규약 · 모델 · 학습 설정 · 평가 · 관문 · 실행 순서를 실제 경로와 수치로 못 박는다.
> 상위 원칙(입력은 mono, 관문 분모는 같은 encoder 의 RNN-T)은 `PLAN.md` 와 `wiki/decisions/decision-mono-input.md` 에 있다.
> 갱신: 설계가 바뀌거나 관문을 판정할 때. 일일 실험 일지는 `wiki/tasks/task-uslm-u1-interleaved-asr` (U1a-0 절).

**작성 2026-09-05** · 서버 **mxc** (`sa_tskim`, conda `vapasr`) · 상태: **착수 전**

---

## 1. 목표와 성공 기준

**목표** — `[AUDIO_k] → text 0..M → <NEXT_AUDIO>` 시퀀스로 **단일 화자 mono 스트리밍 ASR 이 학습되는지**를 작은 데이터(EN 100 h + KO 100 h)로 확인한다. 두 축을 본다:

1. **인식** — 학습이 진행되면서 held-out WER/CER 이 **지속 하락**하고, 같은 encoder 의 Nemotron RNN-T `[56,0]` 과의 격차가 좁혀지는 추세인가.
2. **타이밍** — 토큰이 **증거가 들어온 뒤**(evidence-time 위반 0), **설계한 지연**(δ·80 ms 근방)에, **폭주·굶김 없이**(tok/chunk 가 참조 토큰율에 수렴, M 강제 < 5 %) 나오는가.

**파일럿이므로 절대 WER 관문은 두지 않는다.** 절대 성능은 Stage 2(대규모)에서 잰다. 여기서 확인하는 것은 *시퀀스·방출·정렬 파이프라인이 설계대로 동작한다* 는 사실이다.

**성공 판정** (§7 참고) — 인식 4개 세트 모두 하락 추세 + 타이밍 5개 지표 통과 → Stage 2 로. 하나라도 실패하면 §8 이월 절차.

---

## 2. 범위

| 포함 | 제외 (다음 단계) |
|---|---|
| mono 입력, 단일 화자 발화 | 두 화자 혼합, `<SPK_A/B>` (Stage 3) |
| 텍스트 토큰 + `<NEXT_AUDIO>` / `<EMPTY_AUDIO>` / `<DELAY_d>` | `<SPEECH_ONSET|ENDPOINT>`, VAP·hazard·VAD 헤드 (Stage 3) |
| 교사 강제 학습, committed-prefix greedy 디코드 | self-conditioning, revision (Stage 4) |
| LibriSpeech · KsponSpeech 각 ~100 h | 대화 코퍼스(otoSpeech·AI Hub) (Stage 2-2) |
| Nemotron `[56,0]` frozen, adapter + LoRA 학습 | encoder unfreeze (Stage 2 ablation) |

---

## 3. 데이터

### 3.1 코퍼스와 분량

| 코퍼스 | 용도 | 분량 | 발화 수 | 형식 (mxc 실측 2026-09-05) |
|---|---|---|---|---|
| LibriSpeech `train-clean-100` | 학습 EN | 100.6 h | 28,539 · 251 화자 | flac 16 kHz + 챕터별 `*.trans.txt`(대문자) |
| LibriSpeech `dev-clean` / `dev-other` | 모델 선택 EN | 5.4 / 5.3 h | 2,703 / 2,864 | 〃 + jsonl 동봉 |
| **LibriSpeech `test-clean` / `test-other`** | **보고 EN** | 5.4 / 5.1 h | 2,620 / 2,939 | 〃 + jsonl 동봉 |
| KsponSpeech `KsponSpeech_01` 의 절반 | 학습 KO | ~100 h | ~62,000 (01 전체 124,000) | **raw PCM** 16 kHz·16-bit·mono, `train.trn` (`경로 :: 전사`) |
| KsponSpeech `dev.trn` | 모델 선택 KO | ~4 h | 2,545 | 〃 |
| **KsponSpeech `eval_clean` / `eval_other`** | **보고 KO** | ~3.5 / 3.5 h | 3,000 / 3,000 | 〃 (`.wav` 도 RIFF 헤더 없는 raw PCM) + jsonl(정규화된 label) |

경로: `$MXC_LIBRISPEECH_DIR`, `$MXC_KSPONSPEECH_DIR` (공용·읽기 전용). 동봉 jsonl 의 `audio_path` 는 `/DB/...` 라 이 서버에서는 무효 — **id 만 쓰고 경로는 우리가 만든다.** `train_temp_librispeech.jsonl` 은 1,000 행 샘플이므로 학습용 manifest 는 `*.trans.txt` 에서 직접 만든다.

KsponSpeech 파일럿 부분집합은 `KsponSpeech_01/KsponSpeech_0001 ~ 0062`(하위 폴더 순서, 각 1,000 발화)로 고정한다 — 재현 가능하고, 남은 0063~0124 는 Stage 2 확장분이 된다.

### 3.2 스트림 구성 — "발화" 가 아니라 "스트림" 을 학습한다

스트리밍 모델은 20–30 s 짜리 연속 오디오를 봐야 한다. 두 코퍼스 모두 발화 단위 배포이므로 스트림을 만든다.

**LibriSpeech** — 같은 화자·같은 챕터 안에서 발화를 **원래 순서대로 이어 붙인다.** 발화 사이에 무음 `g ~ U(0.3, 1.5) s`(디지털 0 이 아니라 해당 발화 앞뒤 20 ms 의 배경을 늘린 것) 를 넣는다. 스트림 길이 목표 25 s(20–30). 챕터 경계는 넘지 않는다. 스트림 첫 발화 앞에 0.3–1.0 s 무음.

**KsponSpeech** — 발화 평균 5.6 s(965 h / 620 k). 같은 하위 폴더(같은 세션에서 잘린 것) 안에서 **2–5 개 발화를 파일 번호 순으로 이어 붙여** 15–25 s 스트림을 만든다. 발화 간 무음 `U(0.3, 1.0) s`, 앞뒤 무음 0.3–1.0 s. 원문 대화 흐름을 그대로 두어 자발 발화의 맥락이 유지된다.

이 규칙은 학습·선택·보고 세트에 동일하게 적용한다. **보고용 WER/CER 은 추가로 발화 단위**(스트림 = 발화 하나 + 앞뒤 무음)로도 낸다 — 공개된 test-clean / eval_clean 수치와 비교 가능하게.

### 3.3 텍스트 정규화

| 코퍼스 | 학습 타깃 텍스트 | 평가 정규화 |
|---|---|---|
| LibriSpeech | 대문자 → **소문자**, 아포스트로피 유지, 그 외 구두점 없음(원문에 없음) | 동일. WER 은 jiwer, 공백 분리 |
| KsponSpeech | `(철자)/(발음)` → **철자 형** 채택 · `o/ b/ n/ l/` 잡음 표지 제거 · `+`(반복)·`*`(불명)·`/`(간투사 표지) 제거하고 단어는 유지 · 구두점 제거 · 공백 정리 | 동일 규칙 후 **CER 은 공백 제거** 기준. 동봉 `eval_clean.jsonl` 의 label 과 우리 정규화 결과가 일치하는지 100 개 샘플로 확인 |

철자 형을 고르는 이유: thinker LM 의 사전지식이 문자 표기 쪽이고, Stage 3 이후 텍스트가 대화 상태로 쓰이기 때문. 발음 형 CER 은 참고로만 한 번 낸다.

### 3.4 파이프라인 — 5 단계, 전부 mxc 에서

```text
(1) manifest    experiments/s1_build_manifest.py   → $MXC_DATA_MANIFEST_DIR/{librispeech-100,librispeech-dev,librispeech-test,kspon-100,kspon-dev,kspon-eval}/
                  streams.jsonl: {id, corpus, split, segments:[{utt_id, path, offset_s, dur_s, text}], duration_s, lang}
                  발화 duration 은 flac 헤더 / PCM 바이트÷32000. 스트림 오디오는 저장하지 않고 로드 시 조립한다(무음 삽입 규칙은 seed 고정).
(2) 정렬        experiments/u0_align.py --corpus {librispeech,kspon}   (코퍼스 어댑터 2 개 추가: flac 리더, raw PCM 리더)
                  → $MXC_DATA_MANIFEST_DIR/align/<manifest>/<stream id>.jsonl  (발화당 tokens[{id,text,end_time}], 스트림 상대 시각)
                  Qwen3-ForcedAligner($MXC_ALIGNER_DIR), 80 ms 격자. 발화별로 정렬한 뒤 offset_s 를 더해 스트림 시각으로.
(3) 특징 캐시   experiments/extract_features.py --encoder nemotron-c0 --mono
                  → $MXC_DATA_FEATURE_CACHE_DIR/nemotron-c0/<manifest>/<stream id>.npy  (1, T', 1024) fp16, 12.5 Hz
                  Nemotron [56,0], fp32·TF32 off(Stage 0 교훈), 세그먼트 이어붙이기는 기존 코드 그대로(좌측 context 120 s).
(4) QC          experiments/u0_align_qc.py 를 그대로 — M=4·δ=2 에서 이월률·max backlog·불량 발화(동일 종료시각 뭉침) 집계. 기준: 이월 < 3 %, backlog p99 ≤ 4 chunk
(5) 데이터셋    vapasr/uslm/mono_data.py :: MonoStreamDataset  (§4 규약으로 시퀀스 생성)
```

저장 용량: 200 h × 12.5 Hz × 1024 × 2 B ≈ **18 GB** (Lustre). 특징 추출 ≈ 200 h × RTF 0.005 ≈ **1 GPU 시간**, 정렬 ≈ 2–3 시간(단일 워커, 동시 실행 시 `.tmp` 경합 수정본 사용).

### 3.5 학습 전 검증 (필수)

- [ ] PCM 가정 검증: 10 개 파일을 16 kHz·16-bit 로 읽어 `duration ≈ 글자 수 / 평균 발화 속도` 인지, 스펙트로그램에서 음성 대역이 맞는지 확인. **틀리면 이 문서의 KsponSpeech 절 전체가 무효**
- [ ] 정규화 일치: 우리 규칙 vs 동봉 `eval_clean.jsonl` label, 100 개 샘플 diff
- [ ] 정렬 QC 통과 (3.4-(4))
- [ ] 토큰율: 언어별 tok/80 ms 분포(p50/p99). U0 실측 KO p99 0.78 — EN 은 여기서 처음 잰다. p99 > M 이면 M 재검토
- [ ] 특징 캐시 spot-check: 스트림 3 개를 캐시 없이 직접 인코딩해 npy 와 fp32 일치 확인

---

## 4. 입력 시퀀스 구성

현행 `vapasr/data/interleave.py::build_interleaved` 규약을 **화자 태그만 빼고** 그대로 쓴다. 두 채널 코드(`interleave_data.py`, `model.py` 의 `feats (B,2,K,D)`·merge·`audio_spk`)는 쓰지 않고 mono 경로를 새로 둔다.

### 4.1 시퀀스 규약

```text
[prefix]  <|im_start|>system\n<|im_end|>\n<|im_start|>user\n ... language {English|Korean}<asr_text>  <DELAY_d>
chunk 0   [AUDIO_0]  tok tok            <NEXT_AUDIO>
chunk 1   [AUDIO_1]                     <NEXT_AUDIO>          ← 방출 없음 (라벨의 70–85 %)
chunk 2   [AUDIO_2]  tok tok tok tok    <NEXT_AUDIO>          ← M=4 상한, 초과분은 chunk 3 으로 이월
...
chunk K-1 [AUDIO_K-1] tok               <NEXT_AUDIO>
(끝)      tok tok  <EMPTY_AUDIO>                              ← 스트림 종료 후 잔여 flush
```

| 항목 | 값 | 근거 |
|---|---|---|
| chunk | 80 ms = Nemotron 프레임 1 개, 12.5 Hz | `[56,0]` 실측 79.6 ms/frame |
| `[AUDIO_k]` | `<|audio_pad|>` 자리표시자 1 개 → adapter(Nemotron 특징 1 프레임) 임베딩으로 교체 | mono 이므로 chunk 당 **오디오 토큰 1 개** |
| 토큰 배치 | 토큰의 정렬 종료 시각 `t_end` 에 대해 `k = floor(t_end / 0.08) + δ` | 증거를 다 본 뒤 δ chunk 뒤에 낸다 |
| δ (지연) | 학습 `{2, 3, 4, 6}` 무작위, `<DELAY_d>` 로 prefix 에 조건화 · 평가 기본 δ=2 (160 ms) | 추론 시 지연 조절 가능 |
| M | chunk 당 텍스트 토큰 ≤ 4, 초과분은 다음 chunk 로 이월 | U0 실측 KO p99 0.78 tok/80 ms → 여유 5 배 |
| 창 길이 | 25 s (K = 313 chunk) → 시퀀스 ≈ prefix 20 + 313×(1+1) + 텍스트 ≈ 750–900 위치 | H200 에서 bs 8 여유 |
| 손실 | audio 위치·prefix 제외 전 위치 CE. `<NEXT_AUDIO>` 위치 가중치 **`next_weight` = 0.3 시작**, {0.2, 0.3, 0.5} sweep | U1 v0: 1.0 이면 방출 붕괴(blank 지배), v1: 0.3 에서 해소 |
| 특수 토큰 | `<NEXT_AUDIO>` `<EMPTY_AUDIO>` `<DELAY_1..8>` — 임베딩 여유 행에 배치, grad mask 로 해당 행만 학습 | `model.py` 현행 방식 |

`<SPK_A/B>` 는 vocabulary 에 남겨 두되 **이 단계에서는 생성하지 않는다**(`add_spk_tags=False`, 디코드 시 blocked). Stage 3 에서 같은 체크포인트에 켠다.

### 4.2 디코드 (평가)

- chunk k 의 오디오 임베딩 append → 최대 M 토큰 greedy → `<NEXT_AUDIO>` 를 내면 즉시 다음 chunk. KV cache, 위치당 forward 1 회.
- `next_bias` ∈ {0, 1, 2}: `<NEXT_AUDIO>` logit 페널티(RNN-T blank penalty). **건강한 모델은 bias 0 에서 최선**이어야 한다 — 최적 bias 가 0 에서 멀면 학습 목표 문제.
- committed-prefix: 낸 토큰은 고치지 않는다.
- 방출 시각 정의: chunk k 에서 낸 토큰의 시각 = **(k+1)·80 ms** (chunk 오디오를 다 본 뒤).

---

## 5. 모델

```text
Nemotron 3.5 FastConformer [56,0]  frozen, 캐시 특징 (1024-d, 12.5 Hz)
  → Adapter  MLP 1024 → 2048 → 1024  (U0.5 distill-12k 에서 초기화)
  → Qwen3-ASR-0.6B thinker  LoRA r=16, α=32, 대상 q/k/v/o/gate/up/down  (U0.5 에서 초기화)  + 특수 토큰 임베딩 행
```

| 항목 | 값 |
|---|---|
| 초기화 | `u05-asr-distill-12k/ckpt.pt` (adapter + LoRA). **rack4 `/data4/tskim/VAPASR/experiments/uslm/` → mxc `$MXC_CKPT_ROOT/init/` 로 전송 필요** |
| 학습 파라미터 | adapter ≈ 4.2 M + LoRA 14.3 M + 특수 토큰 행 ≈ **18.5 M** (thinker 본체 0.6 B frozen) |
| 12.5 → 13 Hz 리샘플 | U0.5 와 동일하게 nearest 로 13 Hz 로 맞춘다(thinker 의 AuT 분포). 12.5 Hz 그대로는 ablation |
| 정밀도 | bf16 autocast, logits fp32 |
| 대조군 (같은 데이터로 미리 잰다) | ① Nemotron RNN-T `[56,0]` 80 ms 스트리밍 — **관문 분모** ② Qwen3-ASR 오프라인(`$MXC_QWEN_ASR_DIR`) — 참고선 ③ (있으면) Nemotron 모델 카드의 LibriSpeech 수치 |

U0.5 adapter 는 **대화 코퍼스(otoSpeech·AI Hub)** 의 Nemotron 특징으로 학습됐다. 낭독(LibriSpeech)·자발 발화(KsponSpeech)로 옮기면 분포가 다르다 — 초기 손실이 U0.5 보다 높은 것은 정상이며, random-init 대조 run 을 하나 두어 초기화 이득을 확인한다.

---

## 6. 학습 설정

| 항목 | 파일럿 값 | 비고 |
|---|---|---|
| 서버 / GPU | mxc `sa_tskim`, H200 1 장(여유 최대 자동 선택) | 26.9 GB(U1 v2, L 1240) 기준 bs 8 도 여유 |
| step / bs | **6,000 step**, bs 8 (accum 1) | U1 v1 은 12 k 에서 KO 계속 개선 — 파일럿은 추세 확인이 목적이라 절반 |
| lr | adapter 1e-4 · LoRA 1e-4, cosine, warmup 200 | U0.5 최선(lr 1e-4) |
| 언어 샘플링 | EN:KO = 1:1 (batch 단위 교대) | 시간 균형(100 h : 100 h) |
| δ | {2,3,4,6} 균등 | |
| next_weight | 0.3 (주) · 0.2, 0.5 (2 개 짧은 sweep, 2 k step) | |
| 평가 주기 | 1,000 step 마다 dev/dev(선택 세트), 종료 시 test/eval(보고 세트) | 스트리밍 디코드 = 코퍼스당 20 창 + 발화 단위 200 개 |
| 예상 시간 | ≈ 0.5 s/step × 6 k ≈ 50 min + 평가 6 회 × 10 min ≈ **2 시간 / run** | 주 run 1 + sweep 2 + random-init 1 = 4 run ≈ 반나절 |
| 산출물 | `$MXC_CKPT_EXP_DIR/uslm/s1-mono-<tag>/{ckpt.pt, log.jsonl, eval/*.json}` | best(dev 기준) + last 보존 |

로그는 `<NEXT_AUDIO>` 위치 손실과 텍스트 위치 손실을 **분리**해 남긴다(U1 v0 진단이 이것으로 원인을 찾았다).

---

## 7. 평가

### 7.1 음성 인식

| 세트 | 지표 | 모드 | 무엇을 보나 |
|---|---|---|---|
| LibriSpeech dev-clean / dev-other | WER | 스트림(25 s) | 학습 중 추세, 체크포인트 선택 |
| **LibriSpeech test-clean / test-other** | **WER** | 스트림 + **발화 단위** | 최종 보고. 발화 단위는 공개 수치와 비교용 |
| KsponSpeech dev | CER(공백 제거) | 스트림 | 추세, 선택 |
| **KsponSpeech eval_clean / eval_other** | **CER** | 스트림 + 발화 단위 | 최종 보고 |
| 대조 | 같은 세트에서 Nemotron RNN-T `[56,0]` (스트리밍) · Qwen3-ASR 오프라인 | 〃 | **상대 열화 = (ours − RNN-T) / RNN-T** 를 추적 |

전사 정규화는 §3.3. 언어 태그·특수 토큰은 채점 전 제거(Stage 0 에서 `<ko-KR>` 태그가 문자로 세어져 10 pt 과대였던 사고 방지).

### 7.2 타이밍

정렬 종료 시각을 기준으로 방출 시각을 잰다. 가설 토큰과 참조 토큰을 `difflib` 단조 매칭해 **일치한 토큰만** 지연을 계산한다(현행 `latency_stats`).

| 지표 | 정의 | 통과 기준 |
|---|---|---|
| 토큰 지연 p50 / p90 / p99 | `(k+1)·80 ms − t_end` | p50 ∈ δ·80 ms ± 80 ms (δ=2 → 80–240 ms), p99 ≤ 1 s |
| **evidence-time 위반** | 지연 < 0 (증거 전에 방출) | `viol_80ms`(< −80 ms) **= 0**, `viol`(< 0) < 1 % (정렬 오차 허용) |
| δ 추종 | δ = 2, 3, 4 로 각각 평가했을 때 p50 | p50 이 δ 에 따라 단조 증가, 기울기 ≈ 80 ms/δ |
| tok/chunk | 방출 토큰 수 / chunk 수 | 참조 토큰율의 0.9–1.1 배 (bias 0 에서) |
| M 강제 비율 | chunk 당 4 개를 다 채운 chunk 비율 | < 5 % |
| backlog | 이월 토큰의 최대 누적 | p99 ≤ 4 chunk (320 ms) |
| next_bias 최적점 | bias ∈ {0,1,2} 중 WER 최선 | **0** (0 이 아니면 방출 결정이 학습으로 안 잡힌 것) |
| tick 시간 p99 | chunk 당 디코드 시간(LoRA merge 후) | < 80 ms (H200 기준으로 기록, 배포 GPU 별도) |

### 7.3 학습 건강도 (매 평가마다 기록)

- 손실 분리 곡선: `<NEXT_AUDIO>` 위치 vs 텍스트 위치 — 텍스트 손실이 계속 내려가야 한다.
- 방출률 곡선: 학습 중 tok/chunk 가 0 → 참조율로 올라오는 시점.
- 교사 강제 top-1/top-5 (텍스트 위치): U1 v1 진단과 같은 방식. 단독 발화 기준 top-1 이 54–67 % 에서 얼마나 오르는지가 Stage 2 예측치.

---

## 8. 관문과 이월

**통과** = 아래 전부.

- [ ] 인식: 4 개 보고 세트(test-clean, test-other, eval_clean, eval_other) 모두 1 k → 6 k step 에서 단조 하락(잡음 범위 내 정체 없음)
- [ ] 인식: random-init 대조 run 이 U0.5-init 보다 나쁘거나 같음(초기화가 해를 끼치지 않음)
- [ ] 타이밍: §7.2 의 8 개 기준 전부
- [ ] 정규화·정렬·PCM 검증(§3.5) 전부 통과

**실패 시 — 한 번에 하나만 바꾼다** (의심 순서):

| 증상 | 첫 조치 | 다음 조치 |
|---|---|---|
| 방출 0 / bias 최적 ≥ 1 | `next_weight` 0.2, 0.1 | 목표 시각 허용 창(±1 chunk 라벨 스무딩) |
| evidence 위반 > 0 | 정렬 QC 재검(해당 토큰 시각) | δ 최소 3 |
| tok/chunk 폭주 / M 강제 ↑ | 불량 발화 필터 강화(동일 종료시각 뭉침) | M=6 |
| 텍스트 손실 정체, top-1 낮음 | lr, step 2 배 | adapter 2 층 → 3 층, 12.5 Hz 유지 ablation |
| KO 만 / EN 만 나쁨 | 정규화 규칙·토큰율 재확인 | 언어 샘플링 비율 |

두 축 모두 실패하면 Stage 2 로 가지 않고 시퀀스 규약 자체를 재설계한다(`PLAN.md` Stage 1 실패 조건).

---

## 9. 실행 순서

| # | 작업 | 산출물 | 예상 |
|---|---|---|---|
| 1 | U0.5 ckpt rack4 → mxc 전송 (`u05-asr-distill-12k/ckpt.pt`, 수백 MB, scp) | `$MXC_CKPT_ROOT/init/u05-distill-12k.pt` | 10 min |
| 2 | `s1_build_manifest.py` — LibriSpeech(trans.txt 파싱, 챕터 스트림) · KsponSpeech(trn 파싱, PCM 리더, 정규화, 폴더 스트림) | manifests 6 개 + 통계 | 반나절 (코드) |
| 3 | §3.5 검증 1·2 (PCM, 정규화) | 검증 기록 → task 페이지 | 1 h |
| 4 | `u0_align.py` 어댑터 추가 → 정렬 실행 (bg) | align/ 6 개 | 3 h (GPU) |
| 5 | `extract_features.py --mono` → nemotron-c0 캐시 (bg) | features/ 18 GB | 1 h (GPU) |
| 6 | QC(§3.4-(4)) + 토큰율 + 캐시 spot-check | task 페이지 표 | 1 h |
| 7 | 대조군 측정: Nemotron RNN-T `[56,0]` · Qwen 오프라인 on 4 보고 세트 + 발화 단위 | `eval/baselines.json` | 2 h (GPU) |
| 8 | `mono_data.py` + `s1_train_mono.py`(U1 학습 스크립트에서 분기: mono feats, no SPK, 언어 교대 샘플링) + 스모크 200 step | 스모크 통과 | 반나절 (코드) |
| 9 | 주 run(nw 0.3) + random-init 대조 (bg, 병렬 2 GPU) | ckpt·eval | 2 h |
| 10 | sweep nw 0.2 / 0.5 (2 k step) | 〃 | 1 h |
| 11 | 관문 판정, `output-stage1-mono-pilot` 보고서, `PLAN.md` Stage 1 상태 갱신 | 위키 | 2 h |

코드 신규 4 개: `experiments/s1_build_manifest.py`, `vapasr/uslm/mono_data.py`, `experiments/s1_train_mono.py`, `u0_align.py`/`extract_features.py` 어댑터. 기존 두 채널 코드는 건드리지 않는다.

---

## 10. 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| KsponSpeech PCM 형식 가정(16 k·16-bit·LE) 오류 | KO 전체 무효 | §3.5-1 을 **첫 작업**으로. `.wav` 도 RIFF 가 아님을 확인했으므로 헤더에 의존하지 않는다 |
| KsponSpeech 표기 정규화 불일치 | CER 이 공개 수치와 비교 불가 | 동봉 jsonl label 과 100 개 diff. 팀의 `enNORM` 규칙과 다르면 문서화 |
| 낭독(LibriSpeech) → 대화 도메인 격차 | 파일럿 결과가 Stage 2-2 를 과대 예측 | 파일럿 목적은 동작 검증. Stage 2-2 에서 대화 mono 로 적응 |
| U0.5 adapter 의 대화 특징 분포 → 낭독 분포 이동 | 초기 손실 높음, 이득 불명 | random-init 대조 run |
| 한국어 자발 발화 정렬 품질(간투사·반복) | evidence 위반이 정렬 오차로 오염 | `viol`(< 0) 1 % 허용 + `viol_80ms` = 0 이중 기준 |
| H200 bf16 수치가 A100 과 다름 | U0.5 ckpt 재현 차이 | 로드 직후 U0.5 오프라인 WER 을 mxc 에서 1 회 재측정해 기록 |
| 공용 GPU 점유 | run 지연 | 자동 선택 + bg 실행, 2 GPU 이상은 잡지 않는다 |
