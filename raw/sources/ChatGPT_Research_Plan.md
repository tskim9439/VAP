가능합니다. 이 목표라면 연구를 **“VAP + ASR” 두 모델을 병렬로 붙이는 방식보다, 하나의 streaming speech representation에서 transcription과 future conversational dynamics를 동시에 예측하는 모델**로 설계하는 것이 좋습니다.

Meta Muse Voice Transcribe가 좋은 출발점인 이유도 같습니다. Muse는 80 ms 단위로 audio를 soft token 하나로 만들고, 매 순간 `<|next_audio|>`를 내며 더 들을지 아니면 text token을 생성할지를 autoregressive model이 결정합니다. 여기에 diarization과 `<|speech_endpoint|>`를 special token으로 통합했고, 최종적으로 RL을 이용해 WER와 transcription delay를 함께 최적화합니다. ([Meta AI Research][1])

다만 **Muse의 endpointing과 VAP는 같은 문제가 아닙니다.** 이 차이를 연구의 핵심 novelty로 가져가는 것을 추천합니다.

---

# 제안 연구 주제

**Streaming Conversational Projection ASR**

> A bilingual streaming speech model for simultaneous transcription and predictive turn-taking.

목표를 수식으로 쓰면 매 시점 \(t\)에서 입력된 streaming audio \(x_{\leq t}\)로부터 동시에

$$
P(Y_{\text{text}}\mid x_{\leq t})
$$

와

$$
P(A_{t:t+H}\mid x_{\leq t}, Y_{\leq t})
$$

를 예측하는 모델입니다.

여기서 두 번째 항은 단순 endpoint가 아니라 앞으로 약 2초 동안의 **speaker activity / floor transition / interruption / backchannel 가능성**입니다.

즉,

```text
                     Streaming Audio
                           │
                           ▼
                Streamable Speech Encoder
                           │
                   shared representation
                  ┌────────┴─────────┐
                  │                  │
                  ▼                  ▼
            Streaming ASR     Turn Projection
                  │                  │
           partial transcript       ├─ Future VAP
                  │                  ├─ End-of-turn
                  └──── semantic ────┤
                        state        ├─ Interruption
                                     └─ Backchannel
```

이 방향은 Muse의 **streaming transcription + endpointing**과 원래 VAP의 **future projection**을 합치는 형태입니다.

---

# 1. 기존 연구에서 정확히 어디까지 왔나

현재 가장 중요한 비교 대상은 네 가지입니다.

| 모델                    | Streaming ASR | Semantic 정보 | 미래 Turn 예측 |        KO/EN | 의미                          |
| --------------------- | ------------: | ----------: | ---------: | -----------: | --------------------------- |
| Original VAP          |             ❌ |          약함 |      **✅** |            △ | 미래 activity 예측              |
| Muse Voice Transcribe |         **✅** |       **✅** | △ endpoint | multilingual | 가장 가까운 product architecture |
| JAL-Turn              |        ASR 병렬 |       **✅** | Hold/Shift | multilingual | acoustic+linguistic 결합      |
| 제안 모델                 |         **✅** |       **✅** |      **✅** |    **KO+EN** | VAP + streaming ASR         |

특히 2026년 JAL-Turn은 pretrained ASR encoder와 linguistic feature를 결합하고, ASR와 turn prediction을 병렬로 수행하는 접근을 제안했습니다. ASR encoder를 freeze해 추가 latency를 최소화하는 방향이라 상당히 중요한 baseline입니다. ([arXiv][2])

또 최근 **Next-Turn**은 binary endpoint 대신

$$
\text{time-to-next-speech-onset}
$$

을 직접 예측하는 duration-aware objective를 제시했고, streaming endpoint accuracy에서 개선을 보였습니다. 이 objective도 VAP와 아주 잘 맞습니다. ([arXiv][3])

따라서 새 모델에서 단순히 `EOT=0/1`만 예측하는 것은 피하는 게 좋습니다.

---

# 2. 핵심 연구 hypothesis

제가 논문의 중심 hypothesis를 잡는다면 다음 세 가지입니다.

1. **ASR pretrained streaming encoder는 CPC 기반 VAP보다 turn-taking에 유리한 representation을 제공한다.**
2. **incremental linguistic state를 acoustic VAP와 결합하면 mid-turn pause와 true EOT를 더 잘 구분할 수 있다.**
3. **binary EOT보다 future activity + time-to-next-turn을 jointly predict하면 더 빠르면서 false positive가 낮은 turn-taking이 가능하다.**

이 세 개가 각각 깔끔한 ablation으로 연결됩니다.

---

# 3. 가장 먼저 할 일: Streaming ASR backbone 선정

현재 기준으로 후보를 두 개만 남기는 것을 추천합니다.

## 후보 A — Nemotron 3.5 ASR Streaming 0.6B

제가 **첫 baseline으로 가장 추천**합니다.

구조는

> Cache-Aware FastConformer + RNNT

이고, 24-layer streaming encoder입니다.

무엇보다 inference chunk를

$$
80,\ 160,\ 320,\ 560,\ 1120\text{ ms}
$$

중 선택할 수 있고, encoder self-attention 및 convolution state를 cache하기 때문에 이미 처리한 audio를 재계산하지 않습니다. 한국어 `ko-KR`와 영어가 모두 transcription-ready tier입니다. ([Hugging Face][4])

즉:

```text
80~320 ms audio
      ↓
Cache-aware FastConformer
      ↓
h_t
 ┌────┴─────┐
 │          │
RNNT      VAP
 │          │
text     turn
```

가 매우 자연스럽습니다.

**빠른 수렴 및 engineering feasibility를 확인하기에는 이게 최고입니다.**

---

# 4. 하지만 main research backbone은 Qwen3-ASR-0.6B도 매우 매력적

Muse와 가장 비슷한 쪽은 오히려 Qwen입니다.

Qwen3-ASR의 AuT encoder는 FBank을 **8× downsampling하여 정확히 12.5 Hz**, 즉 **80 ms당 하나의 speech representation**을 만듭니다. Meta Muse 역시 80 ms당 soft audio token 하나를 사용합니다. ([arXiv][5])

구조적으로:

```text
Muse
audio ── 80 ms ── soft token ── LLM
                         │
                         ├ listen
                         └ write


Qwen3-ASR
audio ── 80 ms ── AuT embedding ── Qwen3
                              │
                              └ text
```

입니다.

Qwen3-ASR-0.6B의 AuT encoder는 약 180M 규모이고, dynamic attention window를 1–8초 사이에서 학습해 streaming과 offline을 하나의 모델로 지원합니다. 한국어와 영어가 모두 공식 지원 언어입니다. ([arXiv][5])

또 Apache 2.0이고 공식 fine-tuning recipe까지 공개되어 있습니다. ([GitHub][6])

따라서 연구 진행은:

> **Nemotron → feasibility baseline**

후

> **Qwen3-ASR → Muse-like main model**

순서를 추천합니다.

---

# 5. 제안하는 최종 architecture

원 VAP처럼 두 화자 audio가 분리되어 있다고 우선 가정합니다.

```text
Speaker A audio ──┐
                  │
                  ▼
           Shared Streaming
             ASR Encoder
                  │
                  ├─────────────── hA_t
                  │
Speaker B audio ──┤
                  │
                  ▼
           Shared Streaming
             ASR Encoder
                  │
                  └─────────────── hB_t

              hA_t       hB_t
                │          │
                └────┬─────┘
                     ▼
             Cross-Speaker
               Attention
                     │
              conversational
                  state z_t
              ┌──────┼─────────┐
              ▼      ▼         ▼
             VAP    EOT     Next-Turn
             Head   Head       Head
```

그리고 transcription은 동일한 encoder에서:

```text
hA_t ── RNNT / LLM Decoder ── partial transcript
```

를 생성합니다.

---

# 6. Acoustic feature만 쓰지 말고 ASR의 linguistic state까지 넣어야 합니다

이게 중요한 부분입니다.

예를 들어

> “그런데 제가 말씀드리고 싶은 것은…”

뒤에서 잠깐 멈춘 것과

> “네, 알겠습니다.”

뒤에서 멈춘 것은 acoustic silence만으로는 구분하기 어렵습니다.

따라서:

$$
z_t =
F(h_t^{audio}, h_t^{ling})
$$

로 만들어야 합니다.

Nemotron이라면

* FastConformer encoder state
* RNNT predictor state

를 이용합니다.

Qwen3-ASR이라면

* AuT hidden state
* incremental Qwen decoder hidden state

를 이용할 수 있습니다.

중요한 것은 **ASR 결과 문자열을 다시 다른 text model에 넣지 않는 것**입니다.

그렇게 하면

```text
Audio
 ↓
ASR
 ↓
text
 ↓
TurnGPT
```

라는 cascade가 되어 latency가 커집니다.

대신:

```text
             ┌─ ASR
audio → h_t ─┤
             └─ Turn prediction
```

으로 하고, 내부 decoder state만 공유합니다.

JAL-Turn의 acoustic-linguistic joint modeling과 비슷하지만 이를 **future projection**까지 확장하는 것입니다. ([arXiv][2])

---

# 7. Turn-taking objective는 3개를 동시에 쓰는 것을 추천

핵심은 기존 VAP objective입니다.

미래 2초:

$$
[0,0.2],
[0.2,0.6],
[0.6,1.2],
[1.2,2.0]
$$

에 대해 두 speaker의 activity를 예측합니다.

따라서:

$$
L_{\mathrm{VAP}}
=
CE(y_{\mathrm{256}},\hat y)
$$

입니다.

그런데 여기에 두 objective를 추가합니다.

### Time-to-next-turn

앞으로 상대 speaker가 언제 발화를 시작할지:

$$
\tau_t
=
t_{\text{next speaker onset}}-t
$$

를 예측합니다.

예:

```text
<160 ms
160–320
320–640
640–1280
1280–2560
>2560
```

의 discrete classification으로 하는 것이 좋습니다.

Next-Turn의 duration-aware objective와 유사합니다. ([arXiv][3])

### Semantic event

별도로

$$
P(\mathrm{EOT}),
P(\mathrm{HOLD}),
P(\mathrm{INTERRUPT}),
P(\mathrm{BACKCHANNEL})
$$

을 auxiliary head로 둡니다.

따라서 전체:

$$
\boxed{
L =
\lambda_{ASR}L_{ASR}
+
\lambda_{VAP}L_{VAP}
+
\lambda_{\tau}L_{next-turn}
+
\lambda_{event}L_{event}
+
\lambda_{VAD}L_{VAD}
}
$$

정도로 시작합니다.

---

# 8. Dataset 구성

앞서 조사한 데이터와 정확히 연결됩니다.

### 한국어

가장 중요한 것은 AI Hub의 **감정이 태깅된 자유대화**입니다.

성인 3,000시간 + 청소년 3,000시간이고, 각 화자를 mono로 따로 녹음한 후 하나의 16 kHz stereo conversation으로 제공됩니다. 따라서 VAP training에 거의 이상적입니다. ([AI 허브][7])

처음부터 6,000시간을 쓰지 말고:

```text
KO Stage 1
AIHub Adult      500 h
AIHub Teen       500 h

→ 총 1,000 h
```

로 시작할 것을 권합니다.

ASR adaptation이 추가로 필요하다면 KsponSpeech를 사용합니다.

---

### 영어

우선순위는:

```text
otoSpeech       104 h
SpokenWOZ       249 h
CANDOR          850 h
```

입니다.

TurnBench의 otoSpeech는 EOT/Interruption supervision이 있어서 특히 유용합니다. TurnBench 자체는 30시간 규모의 dual-channel human conversation test이며 causal EOT/INT를 recall, FPR, latency로 평가합니다. ([TurnBench][8])

CANDOR는 1,656개의 unscripted dyadic conversation, 850시간 이상이라 self-supervised VAP training에 매우 좋습니다. ([PMC][9])

SpokenWOZ도 249시간의 실제 human-to-human conversation을 제공합니다. ([SpokenWOZ][10])

---

# 9. 굉장히 유용한 trick: Qwen ForcedAligner로 training target 생성

여기가 꽤 좋은 연구 포인트입니다.

Qwen3-ForcedAligner는 **한국어와 영어를 모두 지원**하고, 시간 index를 **80 ms 단위로 예측**합니다. ([arXiv][5])

따라서:

```text
conversation audio
       +
transcript
       │
       ▼
Qwen3 ForcedAligner
       │
       ▼
word/character timestamp
       │
       ├─ incremental ASR target
       ├─ semantic completion target
       └─ Muse-style emission timing target
```

을 자동 생성할 수 있습니다.

특히 Qwen AuT와 Muse 모두 80 ms rate라서 매우 깔끔합니다.

예를 들어:

```text
0 ms       audio
80 ms      audio
160 ms     audio
240 ms     "안"
320 ms     audio
400 ms     "녕"
480 ms     audio
560 ms     "하세요"
...
```

와 같은 **audio/text interleaving training sequence**를 만들 수 있습니다.

이게 나중에 Muse style architecture로 가는 핵심 bridge입니다.

---

# 10. 연구 단계는 이렇게 가는 게 가장 안전합니다

하나의 거대한 모델부터 만들기보다는 다음 **6단계**로 진행하는 것을 권합니다.

1. **Frozen encoder probing.** Original VAP의 CPC, Nemotron 3.5 encoder, Qwen3 AuT를 모두 freeze하고 동일한 작은 VAP head만 학습합니다. 이 실험만으로 “ASR representation이 VAP에 정말 도움이 되는가?”를 확인할 수 있습니다.

2. **Streaming ASR + VAP multitask baseline.** Nemotron 3.5를 사용해 RNNT는 그대로 유지하고 VAP head만 추가합니다. 처음에는 encoder freeze, 이후 top 4~8 layers만 unfreeze합니다.

3. **Joint acoustic-linguistic turn predictor.** RNNT predictor 또는 Qwen LLM hidden state를 turn predictor에 입력해 acoustic-only 대비 semantic gain을 확인합니다.

4. **Duration-aware future prediction.** 기존 VAP 256-class에 time-to-next-speaker objective를 추가합니다. EOT/HOLD/INT/BACKCHANNEL은 auxiliary supervision으로 사용합니다.

5. **Bilingual joint training.** 한국어와 영어를 1:1 시간 비율로 단순 sampling하지 말고 temperature sampling을 사용합니다. 이전 multilingual VAP 연구에서도 단일 언어 모델의 cross-language generalization은 좋지 않았지만 multilingual training을 하면 각 언어의 monolingual 모델에 가까운 성능을 얻을 수 있었습니다. ([ACL 앤솔로지][11])

6. **Muse-style adaptive emission.** 마지막 단계에서만 `<next_audio>`와 유사한 WAIT/EMIT policy를 추가하고 transcription correctness와 latency를 동시에 최적화합니다. Muse 역시 이 단계에서 WER와 delay reward를 RL로 결합합니다. ([Meta AI Research][1])

---

# 11. Stage 1 실험이 특히 중요합니다

전체 프로젝트에서 첫 번째 논문 가치가 있는 결과가 사실 이겁니다.

동일한 VAP decoder에:

```text
CPC
vs
WavLM
vs
Nemotron Streaming FastConformer
vs
Qwen3-ASR AuT
```

만 바꿉니다.

그리고 encoder는 전부 freeze합니다.

그러면 다음 질문에 답할 수 있습니다.

> **“ASR를 위해 대규모 pretraining된 streaming representation 자체가 conversational future prediction capability를 가지고 있는가?”**

이 결과가 좋으면 이후 연구 방향에 강한 근거가 생깁니다.

---

# 12. 평가 지표도 기존 VAP accuracy만 보면 안 됩니다

최근 TurnBench 결과가 매우 참고할 만합니다.

현재 TurnBench EOT 기준 VAP는:

* Recall 0.845
* FPR 0.055
* median latency 368 ms

로 강력한 baseline입니다. 동시에 사람은 실제 turn transfer를 발화 종료 약 **151 ms 전부터 준비**하기 때문에 아직 개선 여지가 상당합니다. ([TurnBench][8])

따라서 최종 평가를 네 축으로 나누겠습니다.

| 능력            | Metric                             |
| ------------- | ---------------------------------- |
| EN ASR        | WER                                |
| KO ASR        | CER                                |
| Streaming ASR | TTFT, Time-to-Final, revision rate |
| EOT           | Recall @ FPR≤0.1/0.15              |
| Interruption  | Recall @ FPR≤0.1/0.15              |
| Timing        | latency p50/p90                    |
| Prediction    | EOT probability at −600/−300/0 ms  |
| Backchannel   | Precision / Recall / F1            |
| Efficiency    | RTF, GPU memory, chunk latency     |

특히

$$
P(EOT \mid x_{\leq t})
$$

를 실제 EOT **300~600 ms 전**에 얼마나 잘 올리는지 보는 것이 중요합니다.

---

# 13. 한국어 benchmark는 직접 하나 만드는 게 좋습니다

여기가 논문의 또 하나의 contribution이 될 수 있습니다.

현재 TurnBench는 영어 중심입니다.

AI Hub에서 예를 들어 **10~30시간**을 별도 분리한 뒤 사람이

```text
EOT
HOLD
BACKCHANNEL
INTERRUPTION
```

을 annotation합니다.

가능하면 annotator 3명 + 2/3 consensus 방식으로 TurnBench protocol을 따라갑니다.

그러면:

> **Korean TurnBench**

또는 내부 benchmark가 만들어집니다.

특히 한국어는

```text
-요
-습니다
-는데
-고
-면
-서
```

같은 어미와 clause-final prosody가 turn projection에 매우 중요한 역할을 하기 때문에 영어와 상당히 흥미로운 차이가 나올 가능성이 있습니다.

---

# 14. 제가 생각하는 가장 중요한 ablation

논문을 생각한다면 아래 축이 중요합니다.

```text
Acoustic representation
CPC vs Nemotron vs Qwen-AuT

Semantic information
Audio only
vs + ASR encoder
vs + incremental linguistic state

Objective
Binary EOT
vs VAP
vs VAP + Next-turn
vs VAP + Next-turn + Event

Training
Frozen
vs LoRA
vs full fine-tuning

Language
KO
EN
KO+EN

Chunk
80 / 160 / 320 / 640 ms
```

특히 **80/160/320 ms latency-quality curve**는 꼭 보여줘야 합니다.

---

# 15. 최종적으로는 Muse보다 한 단계 더 나아가는 모델을 목표로 할 수 있습니다

Muse의 현재 conceptual output은 대략:

```text
LISTEN
TEXT
SPEECH_ONSET
SPEECH_ENDPOINT
SPEAKER
```

입니다. ([Meta AI Research][12])

우리가 원하는 것은:

```text
LISTEN
TEXT

CURRENT:
SPEECH_ONSET
SPEECH_ENDPOINT

PREDICTIVE:
HOLD
YIELD
INTERRUPT
BACKCHANNEL
TIME_TO_NEXT_TURN
FUTURE_VOICE_ACTIVITY
```

입니다.

즉 Muse가

> **“무엇을 듣고 있으며 언제 끝났는가?”**

를 잘 푸는 모델이라면,

제안 모델은

> **“무엇을 말하고 있으며, 앞으로 누가 언제 말할 것인가?”**

까지 예측합니다.

이 차이는 실제 **Full-Duplex Speech Agent**에서는 꽤 큽니다.

---

# 제가 권하는 실제 개발 로드맵

가장 현실적인 경로는 이렇습니다.

**Phase A — 1st baseline**

```text
Nemotron 3.5 Streaming ASR
            ↓
    frozen FastConformer
            ↓
       VAP Head
```

한국어 AIHub + otoSpeech/CANDOR로 학습합니다.

여기서 기존 VAP와 비교합니다.

**Phase B — Semantic VAP**

```text
 FastConformer state ───────┐
                            ├─ Fusion → VAP
 RNNT predictor state ──────┘
```

이게 첫 번째 핵심 연구 결과가 될 가능성이 높습니다.

**Phase C — Qwen3-ASR**

```text
                 ┌─ Streaming transcript
80ms AuT token ──┤
                 └─ Conversational projection
                        ↑
                    Qwen state
```

이 단계부터 Muse와 직접 비교할 만한 architecture가 됩니다.

**Phase D — Unified Listen/Write/Turn model**

```text
80 ms audio token
       ↓
    Decoder
       │
       ├ <WAIT>
       ├ text token
       ├ <EOT>
       ├ <HOLD>
       ├ <BACKCHANNEL>
       └ <INTERRUPT>
```

그리고 최종적으로:

$$
R =
R_{ASR}
\cdot
R_{latency}
\cdot
R_{turn}
$$

형태의 RL/post-training을 검토합니다.

---

## 가장 추천하는 연구 순서

프로젝트를 처음부터 **“Muse clone”**으로 시작하지 않는 것이 중요합니다.

가장 성공 확률 높은 순서는:

> **Streaming ASR representation이 VAP에 도움이 되는지 검증
> → Acoustic + linguistic VAP
> → bilingual VAP
> → unified streaming transcription
> → Muse-style adaptive emission**

입니다.

그리고 **첫 구현은 Nemotron 3.5**, **최종 연구 모델은 Qwen3-ASR-0.6B 기반**으로 가는 것을 현재 기준으로 가장 추천합니다. Nemotron은 streaming engineering이 이미 해결되어 있고, Qwen은 Muse와 동일한 **12.5 Hz / 80 ms speech representation**을 가지면서 한국어·영어, streaming, forced alignment, fine-tuning까지 모두 공개되어 있어서 연구 자유도가 훨씬 높습니다. ([Hugging Face][4])

특히 이 방향은 이전에 이야기했던 **“ASR token을 지금 downstream LLM에 commit해도 되는가?”**까지 자연스럽게 확장할 수 있습니다. `VAP/Next-turn prediction`과 `ASR commit/revision prediction`을 동일한 conversational state에서 예측하면, 단순 turn detector가 아니라 **Full-Duplex Speech-LM 앞단의 unified streaming perception model**이 됩니다.

[1]: https://research.meta.ai/blog/introducing-muse-voice-transcribe "Introducing Muse Voice Transcribe | Meta AI Research"
[2]: https://arxiv.org/abs/2603.26515?utm_source=chatgpt.com "JAL-Turn: Joint Acoustic-Linguistic Modeling for Real-Time and Robust Turn-Taking Detection in Full-Duplex Spoken Dialogue Systems"
[3]: https://arxiv.org/abs/2606.18094?utm_source=chatgpt.com "Next-Turn: Duration-Aware Streaming Endpoint Detection via Time-to-Next-Speech-Onset Prediction"
[4]: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b?utm_source=chatgpt.com "nvidia/nemotron-3.5-asr-streaming-0.6b · Hugging Face"
[5]: https://arxiv.org/html/2601.21337v2 "Qwen3-ASR Technical Report"
[6]: https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/README.md?utm_source=chatgpt.com "Qwen3-ASR/finetuning/README.md at main · QwenLM/Qwen3-ASR · GitHub"
[7]: https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=&currMenu=115&dataSetSn=71632&topMenu=&utm_source=chatgpt.com "AI-Hub"
[8]: https://turnbench.sesame.com/?utm_source=chatgpt.com "TurnBench"
[9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10065445/?utm_source=chatgpt.com "The CANDOR corpus: Insights from a large multimodal dataset of naturalistic conversation - PMC"
[10]: https://spokenwoz.github.io/?utm_source=chatgpt.com "SpokenWOZ"
[11]: https://aclanthology.org/2024.lrec-main.1036/?utm_source=chatgpt.com "Multilingual Turn-taking Prediction Using Voice Activity Projection - ACL Anthology"
[12]: https://research.meta.ai/blog/introducing-muse-voice-transcribe?utm_source=chatgpt.com "Introducing Muse Voice Transcribe | Meta AI Research"
