# Stage 1 — 단일 화자 스트리밍 ASR 파일럿: 실행 계획

> `PLAN.md` Stage 1 의 실행 문서. 데이터 구성 · 입력 시퀀스 규약 · 모델 · 학습 설정 · 평가 · 관문 · 실행 순서를 실제 경로와 수치로 못 박는다.
> 상위 원칙(입력은 mono, 관문 분모는 같은 encoder 의 RNN-T)은 `PLAN.md` 와 `wiki/decisions/decision-mono-input.md` 에 있다.
> 갱신: 설계가 바뀌거나 관문을 판정할 때. 일일 실험 일지는 `wiki/tasks/task-uslm-u1-interleaved-asr` (U1a-0 절).

**작성 2026-09-05 · 개정 2026-09-05 (사용자 검토 반영: U0.5 초기화 제거, KsponSpeech 연결 금지, dev/test 분리, bias 관문 완화)** · 서버 **mxc** (`sa_tskim`, conda `vapasr`) · 상태: **착수 전**

---

## 1. 목표와 성공 기준

**목표** — `[AUDIO_k] → text 0..M → <NEXT_AUDIO>` 시퀀스로 **단일 화자 mono 스트리밍 ASR 이 학습되는지**를 작은 데이터(EN 100 h + KO 100 h)로, **새 mono 모듈을 random init 에서** 확인한다. 두 축을 본다:

1. **인식** — 학습이 진행되면서 **dev** WER/CER 이 **전체적으로 하락**(best-so-far 갱신)하고, 같은 encoder 의 Nemotron RNN-T `[56,0]` 과의 격차가 좁혀지는 추세인가.
2. **타이밍** — 토큰이 **증거가 들어온 뒤**(evidence-time 위반 0), **설계한 지연**(δ·80 ms 근방)에, **폭주·굶김 없이**(bias 0 에서 방출이 살아 있고 tok/chunk 가 참조 토큰율에 수렴, M 강제 < 5 %) 나오는가.

**파일럿이므로 절대 WER 관문은 두지 않는다.** 6,000 step 결과는 수렴 성능이 아니라 **학습 가능성의 증거**로 해석한다. 절대 성능은 Stage 2(대규모)에서 잰다.

**성공 판정** (§8) — dev 3 세트 하락 추세 + 타이밍 기준 통과 → 보고 세트 최종 1 회 평가 → Stage 2 로. 실패하면 §8 이월 절차.

---

## 2. 범위

| 포함 | 제외 (다음 단계) |
|---|---|
| mono 입력, 단일 화자 발화 | 두 화자 혼합, `<SPK_A/B>` (Stage 3) |
| 텍스트 토큰 + `<NEXT_AUDIO>` / `<EMPTY_AUDIO>` / `<DELAY_d>` | `<SPEECH_ONSET|ENDPOINT>`, VAP·hazard·VAD 헤드 (Stage 3) |
| 교사 강제 학습, committed-prefix greedy 디코드 | self-conditioning, revision (Stage 4) |
| LibriSpeech · KsponSpeech 각 ~100 h | 대화 코퍼스(otoSpeech·AI Hub) (Stage 2-2) |
| Nemotron `[56,0]` frozen, **새 adapter + LoRA 를 random init 에서** 학습 | 기존 U0.5 체크포인트 재사용(하지 않음), encoder unfreeze (Stage 2 ablation) |

---

## 3. 데이터

### 3.1 코퍼스와 분량

| 코퍼스 | 용도 | 분량 | 발화 수 | 형식 (mxc 실측 2026-09-05) |
|---|---|---|---|---|
| LibriSpeech `train-clean-100` | 학습 EN | 100.6 h | 28,539 · 251 화자 | flac 16 kHz + 챕터별 `*.trans.txt`(대문자) |
| LibriSpeech `dev-clean` / `dev-other` | **선택** EN | 5.4 / 5.3 h | 2,703 / 2,864 | 〃 |
| LibriSpeech `test-clean` / `test-other` | **보고** EN (최종 1 회) | 5.4 / 5.1 h | 2,620 / 2,939 | 〃 |
| KsponSpeech `KsponSpeech_01/KsponSpeech_0001 ~ 0062` | 학습 KO | ~100 h (manifest 에서 실측) | 62,000 | **raw PCM** 16 kHz·16-bit LE·mono·headerless, `train.trn` (`경로 :: 전사`) |
| KsponSpeech `dev.trn` | **선택** KO | ~4 h | 2,545 | 〃 |
| KsponSpeech `eval_clean` / **`eval_other-partial[E03314–E06000, n=2687]`** | **보고** KO (최종 1 회) | 2.6 / 3.4 h | 3,000 / **2,687** | 〃 (`.wav` 도 RIFF 헤더 없는 raw PCM). **서버에 E03001–E03313 이 없다** — 공개 3,000 개 수치와 직접 비교하지 말고 대조군도 같은 2,687 개로 잰다. eval PCM 은 끝에 여분 1 바이트(홀수 길이) → 리더가 제외 |

경로: `$MXC_LIBRISPEECH_DIR`, `$MXC_KSPONSPEECH_DIR` (공용·읽기 전용).

- 동봉 jsonl 의 `audio_path` 는 `/DB/...` 라 이 서버에서 무효 — **id 만 참고하고 경로는 우리가 만든다.** `train_temp_librispeech.jsonl` 은 1,000 행 샘플이므로 학습 manifest 는 `*.trans.txt` 에서 직접 만든다.
- KsponSpeech PCM 은 헤더가 없으므로 **바이트 수 ÷ 32,000 = 초**. 홀수 바이트 파일은 로더에서 **마지막 1 바이트를 제외**한다. manifest 생성 시 실제 바이트 기준으로 발화별 시간과 총 시간·발화 수를 기록한다.
- `0001~0062` 는 **재현 가능한 학습 부분집합**으로만 고정한다. 폴더·파일 순서가 화자나 세션을 뜻하지 않으므로 그 이상의 의미를 두지 않는다. 남은 `0063~0124` 는 Stage 2 확장분.

### 3.2 스트림 구성 — 두 코퍼스가 다르다

스트리밍 모델은 연속 오디오를 봐야 하지만, **서로 다른 파일의 음성을 이어 붙이는 것은 화자 동일성이 보장될 때만** 한다.

**LibriSpeech** — 화자·챕터가 파일명에 있으므로 **같은 화자·같은 챕터 안에서 발화를 원래 순서대로 이어 붙인다.** 발화 사이 무음 `g ~ U(0.3, 1.5) s`(디지털 0 이 아니라 인접 발화 앞뒤 20 ms 배경을 늘린 것), 스트림 첫 발화 앞에 0.3–1.0 s 무음. 스트림 길이 목표 25 s(20–30). 챕터 경계는 넘지 않는다. **긴 스트림에서의 동작(KV 누적·이월·지연 유지)은 LibriSpeech 로 검증한다.**

**KsponSpeech** — 파일명·폴더 순서가 화자·세션을 보장하지 않는다. 따라서 **발화 하나 = 스트림 하나**(가변 길이, 평균 5.6 s). 앞뒤에 0.3–1.0 s 무음(padding)만 붙이고, **다른 파일의 음성은 절대 연결하지 않는다.** 짧은 스트림이 많아지는 대신 단일 화자 조건이 지켜진다.
실측 무음 비율은 **19 %**(96.3 h 발화 → 118.7 h 스트림)다. 이번 파일럿은 이대로 간다 — δ 지연 뒤 마지막 토큰을 낼 시간이 필요하고 캐시가 이미 진행 중이다. 단 무해하다고 단정하지 않는다: 학습 로그의 `NEXT_AUDIO` 라벨 비율과 `loss_next / loss_text` 를 기록하고, **방출 붕괴가 나타날 때만** 다음 실험에서 0.2–0.5 s 로 줄인다.

이 규칙은 학습·선택·보고 세트에 동일하게 적용한다. LibriSpeech 보고 WER 은 **스트림 모드와 발화 단위 모드 둘 다** 낸다(발화 단위는 공개 test-clean/other 수치와 비교용). KsponSpeech 는 발화 = 스트림이므로 하나다.

### 3.3 텍스트 정규화와 참조 생성 — 한 규약, 모든 코퍼스 (`vapasr/data/textnorm.py`)

데이터는 계속 추가되므로 코퍼스별 표기를 그대로 두면 코퍼스마다 규칙이 하나씩 늘어난다. **모든 코퍼스의 학습 타깃은 `textnorm.target(text, lang, corpus)` 를 거치고**, 채점은 `score_*` 로 가설·대조군 출력에도 같은 규칙을 건다. 규약은 **사전학습 모델의 출력 표기에 맞춘다** — 2026-09-05 Qwen3-ASR·Nemotron RNN-T 실측(`raw/` 기록 예정):

| | Qwen3-ASR / Nemotron 이 내는 것 | 우리 타깃 규약 | 이유 |
|---|---|---|---|
| EN 숫자 | **단어** ("three hundred dollars", "eleven evenings") | 단어. 코퍼스에 숫자가 있으면 `num2words` 로 변환("$5" → five dollars, "%" → percent) | 모델과 동일, LibriSpeech 와 동일 |
| EN 아포스트로피 | 유지 ("don't", "I've") | 단어 내부 유지, 따옴표 용도는 제거 | 〃 |
| EN 대문자·구두점 | **붙임**(Qwen 은 따옴표·대시까지, Nemotron 은 불규칙 + `<en-US>` 태그) | **타깃에서 제거**(소문자, 구두점 → 공백, 하이픈 → 공백) | LibriSpeech 960 h 에 없어서 붙이려면 라벨을 지어내야 함. 채점에서 양쪽 다 지워 공정하게 |
| KO 숫자 | **한글 읽기** ("십만 원", "이십육 일", "백 명") | KsponSpeech `(철자)/(발음)` 중 **숫자·라틴을 포함한 이중표기만 발음형**, 나머지 철자형 → 한글 전용. 이중표기 밖 숫자는 0.01 % | 모델과 동일, AI Hub(숫자·라틴 0 %)와 동일 |
| KO 구두점·태그 | Qwen 마침표, Nemotron `<ko-KR>` | 제거 | |
| KO 띄어쓰기 | 모델·전사자마다 다름("이십육 일" vs "이십육일") | 원문 유지 | `CER-nospace` 로 흡수 |
| 잡음·간투사 표지 | — | `o/ b/ n/ l/` 제거, 간투사 단어("어", "그")는 유지 | 모델도 간투사를 낸다 |

채점: EN `WER`(jiwer, 위 규칙 후 공백 분리) · KO **`CER-official`**(공백 포함, 공개 수치 비교) + **`CER-nospace`**(내부 비교). 동봉 jsonl 은 기준으로 쓰지 않는다. `.trn` 파서 출력 **100 개 수동 검토**(§3.5).

이 규약이 없던 첫 판(철자형 고정: "30만 원", "3G")은 2026-09-05 에 폐기하고 KsponSpeech manifest·정렬을 `align2/` 에 다시 만든다. LibriSpeech 는 원래 이 규약이라 변화 없음.

### 3.4 파이프라인 — 5 단계, 전부 mxc 에서

```text
(1) manifest    experiments/s1_build_manifest.py   → $MXC_DATA_MANIFEST_DIR/{librispeech-100,librispeech-dev,librispeech-test,kspon-100,kspon-dev,kspon-eval}/
                  streams.jsonl: {id, corpus, split, lang, segments:[{utt_id, path, offset_s, dur_s, text}], duration_s}
                  LibriSpeech: 챕터 내 연결(segments 여러 개). KsponSpeech: segments 1 개(파일 = 스트림).
                  발화 duration 은 flac 헤더 / PCM 바이트÷32000(홀수 바이트 제외). 스트림 오디오는 저장하지 않고 로드 시 조립(무음 규칙 seed 고정).
                  stats.json: 스트림 수·총 시간·길이 분포·(KO) 홀수 바이트 파일 수.
(2) 정렬        experiments/u0_align.py --corpus {librispeech,kspon}   (어댑터 2 개 추가: flac 리더, raw PCM 리더)
                  → $MXC_DATA_MANIFEST_DIR/align/<manifest>/<stream id>.jsonl  (발화당 tokens[{id,text,end_time}], 스트림 상대 시각)
                  Qwen3-ForcedAligner($MXC_ALIGNER_DIR), 80 ms 격자. 발화별로 정렬한 뒤 offset_s 를 더해 스트림 시각으로.
(3) 특징 캐시   experiments/extract_features.py --encoder nemotron-c0 --mono
                  → $MXC_DATA_FEATURE_CACHE_DIR/nemotron-c0/<manifest>/<stream id>.npy  (1, T', 1024) fp16, 12.5 Hz
                  Nemotron [56,0], fp32·TF32 off(Stage 0 교훈), 세그먼트 이어붙이기는 기존 코드 그대로(좌측 context 120 s).
(4) QC          experiments/u0_align_qc.py — M=4·δ=2 에서 이월률·max backlog·불량 발화(동일 종료시각 뭉침) 집계. 기준: 이월 < 3 %, backlog p99 ≤ 4 chunk
(5) 데이터셋    vapasr/uslm/mono_data.py :: MonoStreamDataset  (§4 규약으로 시퀀스 생성)
```

저장 용량: 200 h × 12.5 Hz × 1024 × 2 B ≈ **18 GB** (Lustre). 특징 추출 ≈ 1 GPU 시간, 정렬 ≈ 2–3 시간(단일 워커; 동시 실행은 `.tmp` 경합 수정본으로).

### 3.5 학습 전 검증 (필수, 검증기 `experiments/s1_verify_data.py`)

- [ ] **PCM**: 16 kHz·16-bit LE·mono 로 10 개 파일을 읽어 (a) 길이가 글자 수 / 평균 발화 속도와 맞는지 (b) 스펙트로그램의 음성 대역 (c) 홀수 바이트 파일 수와 처리 확인. **틀리면 KsponSpeech 절 전체가 무효**
- [ ] **`.trn` 파서**: 철자 형 선택·표지 제거 결과 **100 개 수동 검토**. 동봉 jsonl 과의 차이는 기록만 하고 기준으로 쓰지 않는다
- [ ] 정렬 QC 통과 (3.4-(4))
- [ ] 토큰율: 언어별 tok/80 ms 분포(p50/p99). U0 실측 KO p99 0.78 — EN 은 여기서 처음 잰다. p99 > M 이면 M 재검토
- [ ] 특징 캐시 spot-check: 스트림 3 개를 캐시 없이 직접 인코딩해 npy 와 fp32 일치 확인
- [ ] **소규모 overfit** (`--overfit 16 --steps 300`): **EN 16 + KO 16 개 고정, δ=2 고정, 같은 32 개로 학습하고 greedy 디코드.** 시작 시 타깃 보존 assert(라벨 위치 토큰을 이으면 참조 토큰열과 동일). 300 step 뒤 텍스트 top-1 → 1 근처, WER/CER → 0 근처, tok/chunk ≈ 참조율, flush 라운드가 마지막 토큰을 내는지. 전체 데이터로 `--steps 300` 만 돌리는 것은 overfit 이 아니라 smoke 다

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
flush 1   <EMPTY_AUDIO>  tok tok        <NEXT_AUDIO>          ← 스트림 종료: 오디오 자리 대신 <EMPTY_AUDIO>(입력, 손실 없음)
flush 2   <EMPTY_AUDIO>                 <NEXT_AUDIO>          ← 빈 라운드 = "남은 것 없음". 항상 마지막에 하나
```

**flush 규약** — δ 때문에 스트림 끝을 넘긴 토큰(`k ≥ K`)은 오디오 없이 방출해야 한다. `<EMPTY_AUDIO>` 를 오디오 자리 대신 **입력**하고 최대 M 토큰 + `<NEXT_AUDIO>` 를 한 라운드로, `ceil(n_flush / M)` 라운드 뒤 빈 라운드를 하나 더 둔다. 디코드도 같은 라운드를 돌아 토큰 없이 `<NEXT_AUDIO>` 가 나오면 끝낸다(상한 8 라운드). flush 토큰의 chunk 번호는 `K + 라운드`. 이 규약이 없으면 KsponSpeech 는 발화마다 마지막 δ·80 ms 어치가 삭제로 잡히고, 학습 시퀀스는 범위 밖 프레임(`chunk_of = K`)을 gather 한다 — 2026-09-05 검토에서 잡힌 결함.

| 항목 | 값 | 근거 |
|---|---|---|
| chunk | 80 ms = Nemotron 프레임 1 개, 12.5 Hz | `[56,0]` 실측 79.6 ms/frame |
| `[AUDIO_k]` | `<|audio_pad|>` 자리표시자 1 개 → adapter(Nemotron 특징 1 프레임) 임베딩으로 교체 | mono 이므로 chunk 당 **오디오 토큰 1 개** |
| 토큰 배치 | 토큰의 정렬 종료 시각 `t_end` 에 대해 `k = floor(t_end / 0.08) + δ` | 증거를 다 본 뒤 δ chunk 뒤에 낸다 |
| δ (지연) | 학습 `{2, 3, 4, 6}` 무작위, `<DELAY_d>` 로 prefix 에 조건화 · 평가 기본 δ=2 (160 ms) | 추론 시 지연 조절 가능 |
| M | chunk 당 텍스트 토큰 ≤ 4, 초과분은 다음 chunk 로 이월 | U0 실측 KO p99 0.78 tok/80 ms → 여유 5 배 |
| 창 길이 | LibriSpeech 25 s (K ≈ 313) · KsponSpeech 발화 길이 그대로(K ≈ 40–250). 배치는 길이 버킷 | 가변 길이라 padding 낭비를 버킷으로 줄인다 |
| 손실 | audio 위치·prefix 제외 전 위치 CE. `<NEXT_AUDIO>` 위치 가중치 **`next_weight` = 0.3** | U1 v0: 1.0 이면 방출 붕괴(blank 지배), v1: 0.3 에서 해소 |
| 특수 토큰 | `<NEXT_AUDIO>` `<EMPTY_AUDIO>` `<DELAY_1..8>` — 임베딩 여유 행에 배치, grad mask 로 해당 행만 학습 | `model.py` 현행 방식 |

`<SPK_A/B>` 는 vocabulary 에 남겨 두되 **이 단계에서는 생성하지 않는다**(`add_spk_tags=False`, 디코드 시 blocked). Stage 3 에서 같은 체크포인트에 켠다.

### 4.2 디코드 (평가)

- chunk k 의 오디오 임베딩 append → 최대 M 토큰 greedy → `<NEXT_AUDIO>` 를 내면 즉시 다음 chunk. KV cache, 위치당 forward 1 회.
- `next_bias`: `<NEXT_AUDIO>` logit 페널티(RNN-T blank penalty). **bias 0 에서 방출 붕괴·폭주가 없어야 한다**(관문). 최적 bias 는 **dev 에서만** {0, 1, 2} 탐색하고, **bias 0 결과와 dev-최적 bias 결과를 둘 다 보고**한다. `next_weight` 0.3 의 가중 손실은 logit calibration 을 옮길 수 있으므로 최적 bias 가 0 이어야 할 이유는 없다 — 0 인지 여부는 진단값(§7.3).
- committed-prefix: 낸 토큰은 고치지 않는다.
- 방출 시각 정의: chunk k 에서 낸 토큰의 시각 = **(k+1)·80 ms** (chunk 오디오를 다 본 뒤).

---

## 5. 모델

```text
Nemotron 3.5 FastConformer [56,0]  frozen, 캐시 특징 (1024-d, 12.5 Hz)
  → Adapter  MLP 1024 → 2048 → 1024   ← random init
  → Qwen3-ASR-0.6B thinker  LoRA r=16, α=32, 대상 q/k/v/o/gate/up/down  ← random init (LoRA B=0)  + 특수 토큰 임베딩 행 ← random init
```

| 항목 | 값 |
|---|---|
| 초기화 | **전부 random init.** 기존 U0.5 체크포인트는 쓰지 않는다 — 대화 코퍼스·두 채널 입력 규약·다른 방출 목적에 맞춰진 task-specific 파라미터라 재사용 이익보다 해석 혼선이 크다. 기반 모델(Nemotron, Qwen3-ASR thinker)만 공유 |
| 학습 파라미터 | adapter ≈ 4.2 M + LoRA 14.3 M + 특수 토큰 행 ≈ **18.5 M** (thinker 본체 0.6 B frozen) |
| 프레임율 | **12.5 Hz 그대로** — chunk 1 개 = Nemotron 프레임 1 개 = 오디오 토큰 1 개. (U0.5 의 13 Hz 리샘플은 오프라인 ASR 용이었고 interleave 에는 적용하지 않는다; U1 코드도 12.5 Hz) |
| 정밀도 | bf16 autocast, logits fp32 |
| 대조군 (같은 세트에서 미리 잰다) | ① Nemotron RNN-T `[56,0]` 80 ms 스트리밍 — **관문 분모** ② Qwen3-ASR 오프라인(`$MXC_QWEN_ASR_DIR`) — 참고선 ③ (있으면) Nemotron 모델 카드의 LibriSpeech 수치 |

---

## 6. 학습 설정

| 항목 | 파일럿 값 | 비고 |
|---|---|---|
| 서버 / GPU | mxc `sa_tskim`, H200 1 장(여유 최대 자동 선택) | bs 8 도 여유 |
| run | **단일 run** (random init, next_weight 0.3) | sweep 은 관문 실패 시에만(§8) |
| step / bs | **6,000 step**. 배치는 언어별: **EN 2 스트림 · KO 8 스트림** (길이 버킷, accum 1) | LibriSpeech 스트림(≈30 s)이 KsponSpeech(≈5.4 s)의 4–5 배라 같은 bs 는 시간 불균형. 2:8 이 프레임 예산 근사 |
| 언어 교대 | optimizer step 마다 EN, KO 교대 | **"optimizer-step 기준 1:1"** 이지 시간 균형 자체는 아니다 — 위 bs 로 근사 |
| lr / wd | adapter 5e-4 · LoRA 2e-4, cosine, warmup 300, wd 0.01. **임베딩 행렬은 wd 0** | 임베딩은 특수 토큰 행만 grad 지만 AdamW 의 decay 는 전 행(tied lm_head)에 걸린다 → 반드시 별도 그룹 |
| δ | {2,3,4,6} 균등 | |
| 평가 3 종 | ① **sentinel**(1,000 step 마다): dev 3 세트 작은 고정 표본(스트림 20 / 발화 200) — 추세 관찰용 ② **select**(학습 후, `--select`): 마지막 ckpt 3 개를 **큰 고정 표본**(스트림 200 / 발화 1,000, seed 7)으로 평가해 ckpt·bias 선택 → `best.pt` ③ **final**(`--final`): best.pt 로 보고 세트 **전량** 1 회 + δ 추종 | sentinel 20/200 으로는 ckpt 선택이 불안정. test/eval 누수 방지. 축소 실행은 `--smoke-eval` 로만 |
| 예상 시간 | 학습 ≈ 50 min + sentinel 6 회 × 8 min + select 3 × 40 min + final ≈ 1.5 h ≈ **반나절** | |
| 산출물 | `$MXC_CKPT_EXP_DIR/uslm/s1-mono-pilot/{ckpt-<step>.pt, best.pt, results.json, eval/sentinel-*.json, eval/final.json}` | |

로그는 `<NEXT_AUDIO>` 위치 손실과 텍스트 위치 손실을 **분리**해 남긴다(U1 v0 진단이 이것으로 원인을 찾았다).

---

## 7. 평가

### 7.1 음성 인식

| 세트 | 지표 | 모드 | 언제 | 무엇을 보나 |
|---|---|---|---|---|
| LibriSpeech dev-clean / dev-other | WER | 스트림(25 s) | 1 k step 마다 | 추세, 체크포인트·bias 선택 |
| KsponSpeech dev | CER-official · CER-nospace | 발화(=스트림) | 1 k step 마다 | 〃 |
| LibriSpeech test-clean / test-other | WER | 스트림 + **발화 단위** | **최종 1 회** | 보고. 발화 단위는 공개 수치 비교용 |
| KsponSpeech eval_clean / eval_other | CER-official · CER-nospace | 발화 | **최종 1 회** | 보고 |
| 대조 | 같은 세트에서 Nemotron RNN-T `[56,0]` (스트리밍) · Qwen3-ASR 오프라인 | 〃 | 사전 1 회 | **상대 열화 = (ours − RNN-T) / RNN-T** |

전사 정규화는 §3.3. 언어 태그·특수 토큰은 채점 전 제거(Stage 0 에서 `<ko-KR>` 태그가 문자로 세어져 10 pt 과대였던 사고 방지). 모든 인식 결과는 **bias 0** 과 **dev-최적 bias** 두 줄로 낸다.

### 7.2 타이밍 (관문)

정렬 종료 시각을 기준으로 방출 시각을 잰다. 가설 토큰과 참조 토큰을 `difflib` 단조 매칭해 **일치한 토큰만** 지연을 계산한다(현행 `latency_stats`). δ=2, bias 0 기준.

| 지표 | 정의 | 통과 기준 |
|---|---|---|
| **evidence-time 위반** | 지연 < 0 (증거 전에 방출) | **`viol_80ms`(< −80 ms) = 0** — 핵심 안전 지표 · `viol`(< 0) < 1 % (정렬 오차 허용) |
| 방출 생존 | bias 0 에서 tok/chunk | 붕괴(≈0) 도 폭주(> 참조율 2 배) 도 아님 |
| 토큰 지연 p50 / p90 / p99 | `(k+1)·80 ms − t_end` | p50 ∈ δ·80 ms ± 80 ms (δ=2 → 80–240 ms), p99 ≤ 1 s |
| δ 추종 | δ = 2, 3, 4 로 각각 평가했을 때 p50 | δ 에 따라 단조 증가, 기울기 ≈ 80 ms/δ |
| tok/chunk | 방출 토큰 수 / chunk 수 (dev-최적 bias) | 참조 토큰율의 0.9–1.1 배 |
| M 강제 비율 | chunk 당 4 개를 다 채운 chunk 비율 | < 5 % |
| backlog | 이월 토큰의 최대 누적 | p99 ≤ 4 chunk (320 ms) |
| tick 시간 p99 | chunk 당 디코드 시간(LoRA merge 후) | < 80 ms (H200 기준으로 기록, 배포 GPU 별도) |

매칭 토큰이 **0 개**면(방출 붕괴 등) 지연·위반 지표는 `null` 로 두고 `matched=0`, `latency_available=false` 로 명시한다 — 가짜 0 을 넣어 "위반 0" 으로 읽히게 하지 않는다. tick 은 chunk 마다 재서 **p50·p99** 를 낸다(flush 라운드 포함).

### 7.3 건강도 진단 (관문 아님, 매 평가마다 기록)

- 손실 분리 곡선: `<NEXT_AUDIO>` 위치 vs 텍스트 위치 — 텍스트 손실이 계속 내려가야 한다.
- 방출률 곡선: 학습 중 tok/chunk 가 0 → 참조율로 올라오는 시점.
- **dev-최적 bias 가 0 인가** — 0 에서 멀수록 가중 손실이 calibration 을 많이 옮긴 것. Stage 2 에서 next_weight 조정 근거.
- 교사 강제 top-1/top-5 (텍스트 위치): U1 v1 진단과 같은 방식. 단독 발화 기준 54–67 % 였던 값이 random init 에서 어디까지 오는지가 Stage 2 예측치.

---

## 8. 관문과 이월

**통과** = 아래 전부. 판정은 **dev** 로 하고, 통과 후 보고 세트를 1 회 평가한다.

- [ ] 인식: dev 3 세트(dev-clean, dev-other, KsponSpeech dev) 에서 1 k → 6 k step 동안 **best-so-far 가 갱신되며 전체적으로 하락**(엄격한 단조 감소가 아니라 추세). 마지막 2 k step 에서 정체는 허용, 반등은 불허
- [ ] 타이밍: §7.2 의 8 개 기준 전부 (bias 0, δ=2)
- [ ] 검증(§3.5) 6 항목 전부 통과 (PCM, `.trn` 파서 100 개, 정렬 QC, 토큰율, 캐시, overfit)

**판정 기록**

- 2026-09-05 overfit(EN 16 + KO 16, 1,500 step, 옛 규약 데이터): **기능 관문 통과** — 900 step 부터 EN WER 0.000 / KO CER 0.000, tok/chunk = 참조, viol80 = 0, 지연 p50 +201 / +189 ms, p99 +240 / +312 ms, M 강제 0, backlog p99 ≤ 2.85. 1,500 까지 유지.
  **실시간성 관문은 미통과** — tick p99 가 경합 없는 구간에서도 151–186 ms(> 80 ms), 경합 구간 679–755 ms 는 무효. 단독 GPU 에서 별도 측정하기 전까지 미통과 상태로 둔다(§7.2). 개선 후보: LoRA merge 후 CUDA graph, 창 배치 디코드(ragged KV).
- overfit 표본은 seed 셔플 후 **표적 사례를 절반 이상 강제 포함**(KO: 숫자·라틴 이중표기 발화, EN: 발화 경계가 있는 스트림)하고 ID·커버리지를 출력한다. 표적이 없으면 별도 회귀 표본으로 검사한다.
- 2026-09-06 **overfit-v2**(새 규약: `align2/` 선행 공백 + `textnorm`, 900 step, EN 16 중 발화 경계 8 / KO 16 중 숫자 이중표기 8): @900 EN WER 0.000(tok/chunk 0.237 = 참조, p50 +201 / p99 +240 ms, viol80 0) · KO CER 0.000(0.292 = 참조, p50 +202 / p99 +278 ms, viol80 0). **새 규약에서도 기능 관문 통과.** 600 에서 EN 0.574 였던 것은 표본이 전부 긴 다발화 스트림이라 v1 보다 느렸을 뿐 900 에서 수렴. → 6,000-step 파일럿(`s1-mono-pilot`) 착수.

**실패 시 — 한 번에 하나만 바꾼다** (의심 순서):

| 증상 | 첫 조치 | 다음 조치 |
|---|---|---|
| bias 0 에서 방출 0 / 붕괴 | `next_weight` 0.2, 0.1 (2 k step 짧은 sweep) | 목표 시각 허용 창(±1 chunk 라벨 스무딩) |
| bias 0 에서 폭주 / M 강제 ↑ | `next_weight` 0.5 | 불량 발화 필터 강화, M=6 |
| evidence 위반 > 0 | 정렬 QC 재검(해당 토큰 시각) | δ 최소 3 |
| 텍스트 손실 정체, top-1 낮음 | lr 조정, step 2 배 | adapter 2 층 → 3 층, 12.5 Hz 유지 ablation |
| KO 만 / EN 만 나쁨 | `.trn` 정규화·토큰율 재확인 | 언어 샘플링 비율 |
| LibriSpeech 긴 스트림만 나쁨 | 창 길이 15 s 로 축소해 비교 | KV/이월 로직 점검 |

두 축 모두 실패하면 Stage 2 로 가지 않고 시퀀스 규약 자체를 재설계한다(`PLAN.md` Stage 1 실패 조건).

---

## 9. 실행 순서

| # | 작업 | 산출물 | 예상 |
|---|---|---|---|
| 1 | 계획서 개정 (이 문서) — Kspon 연결 금지, dev/test 분리, bias 관문 완화, U0.5 제거 | — | 완료 |
| 2 | mxc 에 현재 코드와 `.env` 동기화 (`scripts/sync-mxc.sh push`, `--delete` 없음) | `/soundai/users/tskim/VAPKT` | 5 min |
| 3 | `experiments/s1_verify_data.py` — PCM 검증기 + `.trn` 파서 + 100 개 검토 출력 | 검증 기록 → task 페이지 | 반나절 |
| 4 | `experiments/s1_build_manifest.py` — LibriSpeech(trans.txt, 챕터 스트림) · KsponSpeech(trn, PCM 바이트 시간, 파일 = 스트림) | manifests 6 개 + stats | 반나절 |
| 5 | `u0_align.py` 어댑터 → 정렬 (bg) · `extract_features.py --mono` → nemotron-c0 캐시 (bg) | align/ 6 개, features/ 18 GB | 3–4 h (GPU) |
| 6 | QC(§3.4-(4)) + 토큰율 + 캐시 spot-check + 토크나이저 일치 테스트(§10) | task 페이지 표 | 1 h |
| 7 | **소규모 overfit** — `--overfit 16 --steps 300` (§3.5). 그 전에 flush 규약·임베딩 wd 수정이 코드에 들어가 있어야 한다(2026-09-05 반영) | overfit-300.json | 30 min |
| 8 | 대조군 측정: Nemotron RNN-T `[56,0]` · Qwen 오프라인 on dev 3 + 보고 세트(eval_other 는 같은 2,687 개) | `eval/baselines.json` | 2 h (GPU) |
| 9 | **random-init 6,000 step 파일럿 (bg)** — sentinel 1 k 마다 | ckpt-*.pt, sentinel-*.json | 1.5 h |
| 10 | `--select 4000,5000,6000` 큰 dev 표본으로 ckpt·bias 선택 → `best.pt` | results.json | 2 h |
| 11 | `--final` 보고 세트 전량 + δ 추종 → 관문 판정 → `output-stage1-mono-pilot` 보고서, `PLAN.md` Stage 1 상태 갱신 | 위키 | 3 h |

코드 신규 5 개: `experiments/s1_verify_data.py`, `experiments/s1_build_manifest.py`, `vapasr/uslm/mono_data.py`, `experiments/s1_train_mono.py`, `u0_align.py`/`extract_features.py` 어댑터. 기존 두 채널 코드는 건드리지 않는다. **체크포인트 전송·초기화 작업은 없다.**

---

## 10. 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| KsponSpeech PCM 형식 가정(16 k·16-bit LE·mono) 오류 | KO 전체 무효 | §3.5-1 을 **첫 작업**으로. `.wav` 도 RIFF 가 아님을 확인했으므로 헤더에 의존하지 않는다. 홀수 바이트는 마지막 1 바이트 제외 |
| `.trn` 정규화 오류 | CER 이 공개 수치와 비교 불가 | 파서 출력 100 개 수동 검토. 동봉 jsonl 은 기준이 아님(발음형 선택 사례) |
| random init 이라 6 k step 안에 방출이 안 살아날 수 있음 | 파일럿 판정 불가 | 소규모 overfit(§3.5-6)으로 파이프라인 결함을 먼저 배제. 그래도 붕괴면 §8 next_weight sweep |
| 낭독(LibriSpeech) → 대화 도메인 격차 | 파일럿 결과가 Stage 2-2 를 과대 예측 | 파일럿 목적은 동작 검증. Stage 2-2 에서 대화 mono 로 적응 |
| KsponSpeech 스트림이 짧음(평균 5.6 s) | 긴 문맥 동작을 KO 로는 못 봄 | 긴 스트림 검증은 LibriSpeech 담당. KO 긴 스트림은 Stage 2-2(AI Hub 대화 mono)에서 |
| 한국어 자발 발화 정렬 품질(간투사·반복) | evidence 위반이 정렬 오차로 오염 | `viol`(< 0) 1 % 허용 + `viol_80ms` = 0 이중 기준 |
| 공용 GPU 점유 | run 지연 | 자동 선택 + bg 실행, 2 GPU 이상은 잡지 않는다 |
| transformers 4.57 의 "incorrect regex pattern" 토크나이저 경고 | 정렬·학습 토큰이 원본과 다르면 LM 사전지식 상실 | **해소(2026-09-05)**: `experiments/s1_tok_check.py` — 로컬 dir 기본 / `fix_mistral_regex=True` / 허브 원본 / qwen_asr processor 네 방식이 EN 150 + KO 150 문장에서 **id 완전 일치**, round-trip 불일치 0. 경고는 이 토크나이저에 해당 없는 오탐 |
