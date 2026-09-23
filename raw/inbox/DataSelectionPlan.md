네. 이 경우에는 **“데이터 포맷 통일 + transcript 검증 + pseudo-label 생성 + 품질 tiering”을 하나의 데이터 정제 파이프라인으로 만드는 것**이 가장 좋습니다.

특히 지금 상황에서는 **제공자 GT를 ground truth라고 가정하지 않는 것**이 핵심입니다. `GT / Qwen3-ASR / Whisper-large-v3`를 모두 후보 hypothesis로 보고, 서로의 agreement와 audio-text consistency를 이용해서 판정하는 구조를 추천합니다.

Qwen3-ASR는 현재 한국어 ASR을 공식 지원하고, 별도의 forced aligner 구현에도 Korean tokenizer가 들어가 있습니다. ([GitHub](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py?utm_source=chatgpt.com "Qwen3-ASR/qwen_asr/inference/qwen3_asr.py at main · QwenLM/Qwen3-ASR · GitHub")) Whisper도 `avg_logprob`, `no_speech_prob`, `compression_ratio`, word timestamp/probability 같은 진단값을 노출하므로 데이터 필터링 teacher로 사용하기 좋습니다. ([GitHub](https://github.com/openai/whisper/blob/main/whisper/decoding.py?utm_source=chatgpt.com "whisper/whisper/decoding.py at main · openai/whisper · GitHub"))

제가 만든다면 아래 구조로 갑니다.

---

# 1. 전체 Pipeline

```text
                ┌──────────────────────────┐
                │ Raw ASR Datasets         │
                │                          │
                │ KsponSpeech / AIHub      │
                │ 자체 DB / LibriSpeech... │
                └────────────┬─────────────┘
                             │
                             ▼
                  ┌───────────────────┐
                  │ ① Data Adapter    │
                  │ dataset-specific  │
                  └─────────┬─────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Unified Manifest      │
                │ + canonical audio     │
                │ + raw transcript      │
                └───────────┬───────────┘
                            │
              ┌─────────────┴───────────────┐
              ▼                             ▼
       ┌─────────────┐               ┌─────────────┐
       │ Qwen3-ASR   │               │ Whisper-v3  │
       └──────┬──────┘               └──────┬──────┘
              │                             │
              └─────────────┬───────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ ② Normalization     │
                 │ comparison_normalize│
                 └──────────┬──────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │ ③ Multi-hypothesis Comparison          │
        │                                        │
        │ GT ↔ Qwen                              │
        │ GT ↔ Whisper                           │
        │ Qwen ↔ Whisper                         │
        └────────────────────┬───────────────────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │ ④ Acoustic / Alignment Quality │
            │ VAD / SNR / clipping / align   │
            └────────────────┬───────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │ ⑤ Adjudicator      │
                  └─────────┬──────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          GOLD           SILVER          REJECT
       Original GT     pseudo-label     / Review
```

이렇게 하면 dataset이 추가되더라도 앞쪽의 `Dataset Adapter` 하나만 추가하면 됩니다.

---

# 2. 가장 먼저 Unified Manifest를 정의

데이터마다 포맷이 다른 문제는 ASR inference 전에 해결해야 합니다.

저라면 JSONL보다 장기적으로는 **Parquet + HuggingFace Dataset** 형태를 추천합니다.

예를 들어 한 utterance를:

```json
{
  "utt_id": "kspon_00000123",
  "dataset": "KsponSpeech",
  "audio_path": "...",
  "audio_sha256": "...",

  "sample_rate": 16000,
  "duration": 4.82,
  "channel": 0,

  "speaker_id": "S00123",
  "language": "ko",

  "gt_raw": "b/ (70%)/(칠 십 퍼센트) 확률이라니...",
  "gt_written": "70% 확률이라니...",
  "gt_spoken": "칠 십 퍼센트 확률이라니...",

  "qwen_raw": null,
  "whisper_raw": null,

  "quality": {},
  "label_source": "provider",
  "quality_tier": null
}
```

### 중요한 점

`gt_raw`**를 절대 overwrite하지 않는 것**을 권합니다.

항상 다음을 별도로 보존합니다.

```text
gt_raw
gt_clean
gt_spoken
gt_written
comparison_norm
training_target
```

KsponSpeech가 이 설계가 필요한 대표적인 사례입니다.

KsponSpeech annotation에는 실제로:

```text
b/ (70%)/(칠 십 퍼센트) 확률이라니
```

처럼 written/phonetic transcript가 함께 들어가고 `b/`, `n/` 같은 noise annotation도 존재합니다. 공개 preprocessing 구현도 phonetic transcription과 spelling transcription을 별도로 선택하도록 되어 있습니다. ([GitHub](https://github.com/sooftware/ksponspeech?utm_source=chatgpt.com "GitHub - sooftware/ksponspeech: Pre-processing KsponSpeech corpus (Korean Speech dataset) provided by AI Hub. · GitHub"))

따라서 dataset ingest 단계에서 이 정보를 날려버리면 안 됩니다.

---

# 3. Normalization을 **두 종류**로 분리해야 합니다

이 부분이 굉장히 중요합니다.

## A. `comparison_normalization`

GT와 Qwen/Whisper가 **같은 말을 인식했는지** 판단하기 위한 normalization.

예:

```text
GT:
"오늘 가격은 (35,000원)/(삼 만 오 천 원)이에요."

Qwen:
"오늘 가격은 35,000원이에요."

Whisper:
"오늘 가격은 삼만 오천 원이에요."
```

raw CER를 계산하면 셋이 굉장히 달라 보입니다.

하지만 실제 recognition content는 동일합니다.

따라서 비교용 normalizer에서는:

```text
Unicode NFC
↓
annotation 제거
↓
punctuation canonicalization
↓
공백 normalize
↓
숫자 equivalence normalization
↓
단위 normalization
```

을 해야 합니다.

---

## B. `training_normalization`

실제로 모델에게 어떤 text style을 학습할 것인가.

이건 완전히 다른 문제입니다.

예를 들어 목표가 production ASR이라면:

```text
"삼만 오천 원"
```

보다는

```text
"35,000원"
```

을 target으로 만들고 싶을 수 있습니다.

따라서:

```text
comparison_norm != training_target
```

로 설계하는 게 좋습니다.

---

# 4. 한국어에서는 CER 하나만 보면 안 됩니다

저라면 최소 세 개를 계산합니다.

```text
CER_surface
CER_semantic
CER_jamo
```

### Surface CER

거의 raw transcript 비교.

```text
35000원
삼만 오천 원
```

→ 상당히 다름.

### Semantic CER

숫자/기호/spacing 등을 equivalence 처리.

```text
35,000원
삼만 오천 원
```

→ 같은 내용으로 취급.

### Jamo CER

한국어 음절 오류에 민감하게 보기 위해:

```text
감사합니다
감사함니다
```

같은 오류를 자모 수준에서도 확인.

그리고 한국어는 띄어쓰기가 annotation source마다 상당히 다르기 때문에 저는 screening metric으로

```text
CER_no_space
```

도 반드시 넣겠습니다.

---

# 5. 핵심은 3-way Agreement

각 sample마다:

```text
G = provider GT
Q = Qwen3-ASR
W = Whisper large-v3
```

라고 하면,

```text
CER(G,Q)
CER(G,W)
CER(Q,W)
```

세 값을 계산합니다.

여기서 굉장히 유용한 패턴이 나옵니다.

---

## Case 1

```text
GT      : 안녕하세요 저는 김태수입니다
Qwen    : 안녕하세요 저는 김태수입니다
Whisper : 안녕하세요 저는 김태수입니다
```

```text
G-Q ≈ 0
G-W ≈ 0
Q-W ≈ 0
```

→ **GOLD**

GT를 그대로 사용.

---

## Case 2

```text
GT      : 오늘 회의는 세 시입니다
Qwen    : 오늘 회의는 네 시입니다
Whisper : 오늘 회의는 네 시입니다
```

```text
G-Q = high
G-W = high
Q-W ≈ 0
```

이 패턴이 매우 중요합니다.

### 높은 확률로 GT annotation error

즉

```text
Qwen == Whisper != GT
```

라면 provider GT를 의심합니다.

하지만 **바로 Qwen/Whisper label로 교체하지 않습니다.**

다음 단계인 alignment 검증으로 넘깁니다.

---

# 6. Forced Alignment를 추가

여기서 Qwen3 Forced Aligner를 상당히 유용하게 사용할 수 있습니다.

공식 Qwen3-ASR 코드의 forced aligner에는 명시적으로 Korean tokenizer가 구현되어 있고, audio/text pair에 timestamp를 붙일 수 있습니다. ([GitHub](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py?utm_source=chatgpt.com "Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py at main · QwenLM/Qwen3-ASR · GitHub"))

각 candidate를 모두 align합니다.

```text
Audio
 │
 ├── GT      → forced alignment
 ├── Qwen    → forced alignment
 └── Whisper → forced alignment
```

예:

```text
Audio:    오늘 회의는 네 시입니다

GT:
오늘      0.1–0.5
회의는    0.5–1.0
세        ????
시입니다  1.5–2.1

Qwen:
오늘      0.1–0.5
회의는    0.5–1.0
네        1.1–1.4
시입니다  1.5–2.1
```

Qwen transcript가 훨씬 자연스럽게 align되면:

```text
Qwen + Whisper consensus
+
Qwen transcript alignment good
+
GT alignment bad
```

가 됩니다.

이 경우 **GT 오류일 확률이 상당히 높습니다.**

---

# 7. Alignment score도 하나의 feature로 저장

Qwen aligner wrapper는 timestamp를 반환하지만 명시적인 scalar confidence를 바로 반환하지는 않습니다. ([GitHub](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py?utm_source=chatgpt.com "Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py at main · QwenLM/Qwen3-ASR · GitHub"))

그래서 저는 다음을 직접 계산하겠습니다.

```text
alignment_coverage
duration_per_token
zero_duration_ratio
long_token_ratio
inter_token_gap
timestamp_monotonicity
audio_coverage
```

더 깊게 수정할 수 있다면 Qwen aligner 내부의:

```python
logits = self.model.thinker(**inputs).logits
output_ids = logits.argmax(dim=-1)
```

부분에서 argmax만 쓰지 말고 ([GitHub](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py?utm_source=chatgpt.com "Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py at main · QwenLM/Qwen3-ASR · GitHub"))

```text
max probability
top1-top2 margin
entropy
```

를 뽑아서 **alignment confidence**를 만들 수 있습니다.

이건 꽤 쓸 만한 quality feature가 될 가능성이 높습니다.

---

# 8. Whisper confidence도 적극 활용

Whisper는 기본적으로 다음 값을 제공합니다.

```text
avg_logprob
no_speech_prob
compression_ratio
```

([GitHub](https://github.com/openai/whisper/blob/main/whisper/decoding.py?utm_source=chatgpt.com "whisper/whisper/decoding.py at main · openai/whisper · GitHub"))

word timestamp를 활성화하면 word-level probability도 활용할 수 있습니다. Whisper 코드 자체에서도 낮은 word probability와 비정상 duration을 hallucination detection에 사용합니다. ([GitHub](https://github.com/openai/whisper/blob/main/whisper/transcribe.py?utm_source=chatgpt.com "whisper/whisper/transcribe.py at main · openai/whisper · GitHub"))

따라서:

```json
"whisper": {
    "text": "...",
    "avg_logprob": -0.21,
    "no_speech_prob": 0.002,
    "compression_ratio": 1.21,
    "mean_word_prob": 0.93
}
```

처럼 모두 저장하는 게 좋습니다.

단,

> `Whisper avg_logprob > X이면 좋은 데이터`

식의 단일 threshold는 추천하지 않습니다.

도메인별 calibration이 필요합니다.

---

# 9. Audio 자체 품질도 따로 평가

Transcript가 맞더라도 ASR training data로 나쁜 audio가 있습니다.

예를 들어:

```text
심한 clipping
음성이 너무 작음
긴 silence
두 명 동시 발화
background TV
music
잘못 segment됨
1초 audio에 20자 transcription
```

입니다.

그래서 별도의 acoustic quality stage가 필요합니다.

저라면 최소:

```text
duration
speech_duration
speech_ratio
silence_ratio

RMS / loudness
clipping_ratio
SNR estimate

number_of_speakers
overlap_ratio

chars/sec
syllables/sec
words/sec
```

를 봅니다.

특히 아주 강력한 heuristic이:

```text
text_length / speech_duration
```

입니다.

예:

```text
0.8초 audio
GT = "안녕하세요 오늘 날씨가 정말 좋은 것 같습니다"
```

면 거의 무조건 segmentation/annotation 오류입니다.

---

# 10. 최종 Quality Score는 이렇게

하나의 score로 만들더라도 내부 feature는 모두 남겨야 합니다.

예를 들면:

Q=w1Ateacher+w2AGT+w3Falign+w4CASR+w5QaudioQ = w\_1 A\_{teacher} + w\_2 A\_{GT} + w\_3 F\_{align} + w\_4 C\_{ASR} + w\_5 Q\_{audio}

여기서

### Teacher agreement

Ateacher=1−CER(Q,W)A\_{teacher}=1-CER(Q,W)

### GT agreement

AGT=1−CER(G,Q)+CER(G,W)2A\_{GT} = 1-\\frac{CER(G,Q)+CER(G,W)}{2}

### Alignment

```text
F_align
```

### ASR confidence

```text
Whisper avg_logprob
Whisper word prob
Qwen token logprob
```

### Audio quality

```text
SNR
speech_ratio
clipping
overlap
```

입니다.

하지만 저는 처음부터 neural classifier로 합치기보다는 **rule-based adjudicator**로 시작하겠습니다.

---

# 11. 실제로는 4개 Tier가 좋습니다

## Tier A — Gold

```text
Q ↔ W CER < 5%
G ↔ Q CER < 5%
G ↔ W CER < 5%

alignment good
audio good
```

### 사용

```text
GT 그대로 training
```

---

## Tier B — GT correction candidate

```text
Q ↔ W CER < 5%

BUT

G ↔ Q > 15%
G ↔ W > 15%
```

즉:

```text
Qwen ≈ Whisper
Qwen ≠ GT
Whisper ≠ GT
```

### 사용

pseudo-GT 후보.

---

## Tier C — ambiguous

예:

```text
Qwen ≈ GT
Whisper != GT
```

또는

```text
Qwen != Whisper
GT는 중간
```

### 사용

```text
human review
또는 training 제외
```

---

## Tier D — bad

```text
Qwen != Whisper != GT
+
alignment bad
+
audio bad
```

### 사용

```text
reject
```

제가 예시로 5%/15%를 적었지만 **이 숫자는 calibration을 위한 초기값일 뿐**입니다.

---

# 12. 가장 효율적인 Human Review 전략

전체 데이터를 사람이 확인할 필요가 없습니다.

예를 들어 5,000시간이 있다면:

```text
전체
│
├── Tier A  70% → 자동 accept
│
├── Tier B  15% → 자동 수정 후보
│
├── Tier C  10% → human review
│
└── Tier D   5% → discard
```

그리고 Tier B/C에서 각각 수백\~수천 sample만 사람이 듣습니다.

이를 기반으로:

```text
P(GT correct | features)
P(Qwen correct | features)
P(Whisper correct | features)
P(reject | features)
```

classifier를 학습할 수 있습니다.

처음에는 XGBoost/LightGBM 정도면 충분합니다.

input feature:

```text
CER_GT_Qwen
CER_GT_Whisper
CER_Qwen_Whisper

CER_jamo_*
CER_semantic_*

Whisper_avg_logprob
Whisper_mean_word_prob
Whisper_no_speech_prob

Qwen_logprob

alignment_GT
alignment_Qwen
alignment_Whisper

speech_ratio
SNR
clipping_ratio

duration
chars_per_sec
```

그리고 target:

```text
KEEP_GT
USE_QWEN
USE_WHISPER
REJECT
```

이렇게 됩니다.

이게 결국 **Data Quality Classifier**가 됩니다.

---

# 13. 중요한 문제: Qwen + Whisper만으로 consensus를 만들면 bias가 생김

이 부분은 상당히 조심해야 합니다.

```text
Qwen == Whisper
```

라고 해서 반드시 정답은 아닙니다.

두 모델 모두 인터넷 기반 speech/text 데이터의 영향을 받았고, 특히:

```text
고유명사
숫자
외래어
code switching
전문 용어
```

에서 비슷한 correction을 할 수 있습니다.

따라서 가장 좋은 구성은 결국:

```text
Provider GT

Teacher 1:
Qwen3-ASR-1.7B

Teacher 2:
Whisper-large-v3

Teacher 3:
독립 구조의 Korean ASR
```

입니다.

태수님 상황에서는 오히려 **현재 사용 중인 streaming FastConformer 계열 모델**을 세 번째 judge로 넣는 것이 상당히 좋습니다.

즉:

```text
             ┌── Qwen3-ASR
Audio ───────┼── Whisper
             └── FastConformer
```

AR decoder 계열만 두 개 놓는 것보다 **CTC/RNNT 계열이 하나 섞이는 게 ensemble diversity 측면에서 좋습니다.**

---

# 14. 처음부터 pseudo-label로 GT를 덮어쓰지 않는 게 중요

예:

```text
label_provider = GT
label_qwen
label_whisper
label_consensus
label_selected
```

를 모두 남깁니다.

그리고:

```text
label_selected = provider
label_source   = provider
```

또는

```text
label_selected = ...
label_source   = teacher_consensus
```

처럼 provenance를 남겨야 합니다.

나중에 Qwen4-ASR 같은 더 좋은 모델이 나오면 **raw부터 다시 검증**할 수 있습니다.

---

# 15. 제가 추천하는 최종 Manifest

대략 이런 형태가 좋습니다.

```text
utterance_id
dataset
dataset_version

audio_path
audio_hash
sample_rate
duration
channel
speaker_id

language

# original
gt_raw
gt_spoken
gt_written

# teacher
qwen_text
whisper_text
fastconformer_text

# normalized comparison
gt_cmp
qwen_cmp
whisper_cmp
fastconformer_cmp

# agreement
cer_gt_qwen
cer_gt_whisper
cer_qwen_whisper
semantic_cer_*
jamo_cer_*

# ASR quality
qwen_logprob
whisper_logprob
whisper_no_speech
whisper_word_prob

# alignment
gt_alignment_score
qwen_alignment_score
whisper_alignment_score

# acoustic
snr
speech_ratio
clipping_ratio
overlap_ratio

# decision
quality_score
quality_tier

label_selected
label_source

filter_reason
pipeline_version
```

특히

```text
pipeline_version
```

을 반드시 넣으세요.

normalization rule이 바뀌거나 teacher model version이 바뀌면 재현성 문제가 생기기 때문입니다.

---

# 16. 태수님 상황에서는 이 구조가 가장 현실적입니다

제가 구현 순서를 정한다면:

```text
Phase 1
Dataset Adapter
        ↓
Canonical Manifest
        ↓
Korean Comparison Normalizer

Phase 2
        ↓
Qwen3-ASR-1.7B
Whisper-large-v3
        ↓
3-way CER matrix

Phase 3
        ↓
Whisper confidence
Qwen forced alignment
audio quality
        ↓
Rule-based Tiering

Phase 4
        ↓
500~2000개 human validation
        ↓
threshold tuning

Phase 5
        ↓
Quality Classifier
        ↓
자동 데이터 정제
```

가 좋겠습니다.

### 특히 저는 초기에 이 판단 matrix를 핵심으로 쓰겠습니다.


| GT    | Qwen | Whisper | 판단                                |
| ----- | ---- | ------- | --------------------------------- |
| 일치    | 일치   | 일치      | **GT 유지**                         |
| 불일치   | 일치   | 일치      | **GT 오류 후보 → alignment 검사**       |
| 일치    | 일치   | 불일치     | **GT 유지 후보**                      |
| 일치    | 불일치  | 일치      | **GT 유지 후보**                      |
| 모두 다름 |      |         | **Review / Reject**               |
| GT 없음 | 일치   | 일치      | **pseudo-label strong candidate** |
| GT 없음 | 불일치  | 불일치     | **Reject / Review**               |


그리고 **여기에 forced alignment가 최종 tie-breaker로 들어가는 구조**가 가장 깔끔합니다.

---

## 한 단계 더 나아가면

이 파이프라인은 단순 preprocessing tool이 아니라 사실상 **ASR Data Factory**로 만들 수 있습니다.

```text
                ASR Data Factory
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    Ingestion     Validation      Labeling
        │              │              │
 Dataset Adapter   Multi-ASR      GT / Pseudo
        │           Consensus       Label
        │              │              │
        └──────────────┼──────────────┘
                       ↓
                Unified Dataset
                       ↓
               ┌───────┴───────┐
               ↓               ↓
            ASR Train       VAP-ASR Train
```

태수님이 지금 진행하고 있는 **VAP-ASR / streaming ASR 데이터 구축에도 그대로 재사용할 수 있는 구조**입니다. 특히 VAP-ASR은 transcript correctness뿐 아니라 segment boundary가 중요하므로 `forced alignment + VAD boundary quality`까지 같이 저장해두면 이후 `<NEXT_AUDIO>/<DELAY_n>` sequence를 만들 때도 유용합니다.

원한다면 다음 단계로 제가 **이걸 실제 Python 프로젝트 수준으로 구체화해서** `dataset adapter → normalizer → Qwen/Whisper inference → scorer → adjudicator → parquet export` **디렉터리 구조와 class/interface 설계까지** 만들어드리는 게 좋겠습니다.