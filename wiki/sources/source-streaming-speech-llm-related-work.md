---
type: source
status: active
created: 2026-09-07
updated: 2026-09-07
summary: Streaming SpeechLLM·RNN-T 7개 연구를 방출 정책, 정렬, 지연, 재현성, VAPASR 적용성으로 비교
sources:
  - [[source-muse-voice-transcribe]]
  - [[output-interleaved-streaming-slm-architecture]]
  - [[output-stage1-mono-pilot]]
---

# Streaming SpeechLLM 및 저지연 ASR 관련 연구 비교

## 질문과 범위

VAPASR의 `80 ms audio soft token → text 0..M → <NEXT_AUDIO>` 구조가 기존 연구와
어떻게 연결되며, RNN-T 수준의 정확도와 예측 가능한 token timing을 얻기 위해 무엇을
가져와야 하는지 비교한다. 직접적인 decoder-only SpeechLLM뿐 아니라, 방출 시점을
학습하는 transducer 연구도 포함한다.

비교 대상은 다음 일곱 개다.

1. Samsung AI Center Cambridge의 **Streaming Speech-to-Text Translation with a SpeechLLM**
2. **Speech ReaLLM**
3. **Streaming Speech Recognition with Decoder-Only Large Language Models and Latency Optimization**
4. **Moshi**
5. Meta **Muse Voice Transcribe**
6. **Alignment Restricted Streaming RNN-T**
7. **Minimum Latency Training of Sequence Transducers**

Muse는 논문이 아니라 공식 기술 소개다. 공개된 수치와 설명은 유용하지만, 논문과 같은
재현성·검증 수준으로 취급하지 않는다.

## 먼저 읽을 결론

- **구조적으로 가장 가까운 공개 논문은 Speech ReaLLM**이다. RNN-T의 blank에 해당하는
  제어 토큰을 decoder-only 모델이 예측한다. 다만 240 ms 입력 단위와 960 ms future
  context를 사용하므로 VAPASR의 80 ms strict-streaming 지연 증거로는 쓸 수 없다.
- **VAPASR와 가장 닮은 공개 설명은 Muse**다. 80 ms soft token과 `<|next_audio|>`가
  거의 같은 규약이다. 그러나 weights·학습 세부사항이 닫혀 있어 설계 선례이지 재현 가능한
  baseline은 아니다.
- **현재 hard-δ 목표를 개선하는 가장 직접적인 선례는 Ar-RNN-T와 MinLT**다. 전자는
  단일 정렬점 대신 허용 window 안의 경로를 남기고, 후자는 기대 지연을 loss에 직접 넣는다.
  둘 다 RNN-T lattice에 정의되어 있으므로 decoder-only CE에 그대로 복사할 수는 없다.
- **학습된 read/write 정책의 대안은 Wan et al.의 MoChA 방식**이다. hard alignment보다
  유연하지만 정책 모듈과 segmentation 실패 모드가 추가된다. 단순한 window/soft target보다
  먼저 도입할 이유는 아직 없다.
- **Samsung 연구의 핵심 가치는 번역 성능 자체보다 wait-token의 silence 강건성과
  early-exit 정책**이다. 640 ms chunk와 1–2초 번역 지연은 80 ms ASR 목표와 직접 비교할
  수 없다.
- **Moshi는 turn-taking과 full-duplex 설계의 선례**다. 200 ms는 시스템 대화 지연이며,
  부록의 streaming ASR text는 2초 고정 지연을 사용한다. VAPASR token latency와 같은
  수치로 인용하면 안 된다.

## 구조와 학습 방식 비교

| 연구 | 주 과제 | audio/text 시간 표현 | 방출 결정 | 정렬·학습 목표 | VAPASR와의 관계 |
|---|---|---|---|---|---|
| Samsung Intermixed SpeechLLM (2026) | streaming speech-to-text translation | 640 ms chunk당 encoder vector 8개를 넣고 마지막 vector에서 LLM 판정 | text 또는 `Wait`; 선택적 early-exit wait/emit head | source word forced alignment + LLM 생성 phrase translation alignment의 단일 경로 CE | `<NEXT_AUDIO>`와 가장 가까운 번역 선례. early exit가 runtime 후보 |
| Speech ReaLLM (2024) | continuous streaming ASR | 240 ms speech embedding 뒤 text 0개 이상 | text 또는 `BLANK` | 외부 CTC 정렬 한 경로의 CE | 현재 구조의 가장 직접적인 공개 ASR 선례 |
| Wan et al. (2026) | decoder-only streaming ASR | MoChA가 누적 speech embedding을 가변 segment로 분할 | 별도 read/write policy가 다음 token 시점 결정 | streaming/non-streaming 공동 학습 + policy용 minimum-latency objective | hard-δ 이후 검토할 learned segmentation 대안 |
| Moshi (2024) | full-duplex speech-to-speech dialogue | 사용자·모델 audio를 병렬 fixed time grid로 모델링하고 text를 audio token 앞에 둠 | adaptive wait보다 PAD/EPAD로 고정 시간축 유지 | time-aligned inner-monologue text와 audio token 생성 | overlap·turn-taking에는 직접적, ASR timing에는 간접적 |
| Muse Voice Transcribe (2026) | streaming ASR·diarization·endpointing | 80 ms당 soft token 1개 | text 또는 `<|next_audio|>`; stream 끝은 `<|empty_audio|>` | 지도학습 세부 비공개, WER와 delay를 결합한 RL 설명 | clock과 제어 규약이 가장 유사하나 closed black box |
| Ar-RNN-T (2021) | 저지연 streaming ASR | 표준 RNN-T lattice | blank/non-blank 경로 합 | 외부 alignment 주변 `[left, right]` window 밖 경로 제거 | 단일 hard-δ를 window marginalization으로 바꾸는 근거 |
| Transducer MinLT (2022) | 저지연 streaming ASR | transducer lattice의 diagonal별 지연 | lattice 전체의 방출 확률 | transducer loss + 미분 가능한 expected-latency loss | 명시적 latency loss를 설계할 이론적 선례 |

## 결과와 비교 가능성

| 연구 | 보고된 결과 | 지연 수치의 뜻 | 공개성 | 직접 비교 시 주의점 |
|---|---|---|---|---|
| Samsung | 약 3,700 h, EN→KO/FR에서 offline에 가까운 COMET, 약 1–2 s latency | translation의 average logical latency | 논문 공개, in-house 3B LLM·일부 데이터 비공개 | WER가 아닌 번역 품질이며 chunk가 640 ms |
| Speech ReaLLM | LibriSpeech 80M: test-clean 3.0%, test-other 7.4%; Llama-7B: 4.2%/9.5% | 실시간 실행을 보고하지만 encoder에 960 ms right context | 논문 공개, 완전한 공식 구현·recipe 여부는 별도 확인 필요 | causal encoder의 token delay와 동등하지 않음 |
| Wan et al. | AISHELL-1/2 CER 5.1%/5.5%; latency objective로 평균 token delay 62.5% 감소 | 평균 token generation delay | 논문 공개 | Mandarin CER이며 VAPASR와 데이터·encoder·언어가 다름 |
| Moshi | theoretical 160 ms, 실사용 약 200 ms | full-duplex 시스템 반응 지연 | code·weights 공개 | streaming ASR text는 2 s 고정 지연이어서 200 ms와 분리해야 함 |
| Muse | 공식 소개에서 transcription WER 3.1%, adaptive-delay Pareto 주장 | time-to-final 중심의 제품 지표 | API만 공개, weights·recipe 비공개 | test set과 세부 지연 분포가 없어 논문 baseline으로 부적합 |
| Ar-RNN-T | LibriSpeech·내부 데이터에서 WER/방출 지연 조절, LSTM 학습 처리량 4배 | token end-time delay와 endpoint latency를 구분 보고 | 논문 공개 | 숫자는 모델·buffer·endpoint 조건별로 달라 하나의 대표값으로 축약하면 안 됨 |
| Transducer MinLT | WSJ causal Conformer-T: 220→27 ms, WER 열화 0.7%; Ar-RNN-T 110 ms, FastEmit 67 ms | lattice에서 정의한 평균 emission latency | 논문 공개 | 작은 WSJ 실험이며 p99·service time을 뜻하지 않음 |

이 표의 latency 수치는 서로 순위를 매길 수 없다. 최소한 다음 네 가지를 분리해야 한다.

1. encoder가 사용하는 **future context / algorithmic look-ahead**
2. acoustic evidence end에서 token이 나올 때까지의 **emission delay**
3. 한 tick을 처리하는 **wall-clock service time과 p99 deadline miss**
4. utterance 종료부터 최종 결과 또는 응답까지의 **endpoint/system latency**

VAPASR는 이 네 항목을 각각 보고해야 한다. 현재의 `viol80`, token delay p50/p90/p99,
tick p99는 이 구분을 지향하지만, 삭제 token을 `미방출`로 포함한 deadline recall과
PCM 입력부터 출력까지 end-to-end 측정이 더 필요하다.

## 논문별 까다로운 평가

### 1. Samsung: Intermixed SpeechLLM

Samsung AI Center Cambridge가 제안한 실제 명칭은 **Intermixed SpeechLLM**이다. text
token과 Wait token을 한 autoregressive sequence에서 예측하며, Wait 다음에는 새 speech
chunk가 들어온다. 이것은 VAPASR의 `<NEXT_AUDIO>`와 사실상 같은 제어 추상화다.

번역에서는 목적어 순서가 바뀌기 때문에 word-to-word 정렬만으로 emission 시점을 만들기
어렵다. 논문은 source forced alignment와 Qwen3-14B가 만든 phrase-level bilingual
alignment를 결합한다. VAPASR는 단일 언어 ASR이므로 이 복잡성은 필요 없지만, forced
alignment가 유일한 정답은 아니라는 점과 alignment 오류를 보수적으로 처리하는 방식은 참고할
수 있다.

가장 유용한 결과는 5초 선행 무음에서 fixed wait 정책이 무너지는 반면 learned Wait가
강건하다는 점이다. 또 LLM의 앞쪽 layer에서 wait/emit만 먼저 판단해 Wait이면 나머지 layer를
건너뛰는 early-exit는 VAPASR tick p99 개선 후보가 된다. 다만 640 ms deadline 안에서의
에너지 절약 설계이므로 80 ms deadline에서 그대로 충분하다고 볼 수는 없다.

### 2. Speech ReaLLM

decoder-only 모델이 매 speech embedding 뒤에 text를 0개 이상 내고 `BLANK`로 다음 audio를
요청한다. 현재 VAPASR 규약의 학술적으로 가장 가까운 형태다. continuous audio와 명시적
endpointing 없이도 동작하고, LibriSpeech에서 전통 ASR에 근접한 수치를 보였다는 점은
“구조적으로 불가능하다”는 반론을 약화한다.

그러나 80M 모델은 최대 900 epoch를 학습했고, encoder segment는 1.92초이며 960 ms의
right context를 포함한다. VAPASR의 1,930 h full-FT가 아직 필요하다는 판단에는 힘을 주지만,
strict-causal 80 ms timing의 증거는 아니다. 또한 외부 CTC 정렬의 단일 경로 CE라서 현재
hard-δ와 같은 정렬 경직성을 공유한다.

### 3. Wan et al.: MoChA read/write streaming LLM-ASR

causal speech encoder 뒤의 MoChA policy가 “더 읽을지, 지금 쓸지”를 결정하고, write 시점에
누적 segment와 직전 token을 LLM에 넣는다. 고정 80/240/640 ms마다 대형 LLM을 호출하지
않아도 된다는 점이 runtime에는 매력적이다. 논문의 latency objective도 방출 경계를 더 이르게
학습한다.

반면 별도 policy와 threshold가 추가되어 ASR 오류와 segmentation 오류를 분리해야 한다.
현재 VAPASR는 이미 방출 중심과 token rate를 학습했으므로, 먼저 hard-δ를 허용 window와
soft target으로 완화한 뒤에도 정확도·지연 Pareto가 막힐 때 비교하는 편이 실험 해석이 쉽다.

### 4. Moshi

Moshi는 사용자와 모델의 audio stream을 동시에 모델링하고, time-aligned text를 audio token보다
먼저 예측하는 Inner Monologue를 쓴다. 겹침, interruption, interjection을 turn segmentation 없이
다루므로 Stage 3 이후 turn-taking에는 가장 중요한 선례 중 하나다.

하지만 Moshi의 200 ms는 대화 시스템 latency다. 논문 부록의 streaming ASR 파생 모델은 text에
2초 고정 지연을 둔다. 따라서 “Moshi도 200 ms ASR이므로 VAPASR의 200 ms token delay와 같다”는
결론은 틀리다. 현재 Stage 2의 WER·token timing 관문보다는 이후 병렬 화자 stream과 turn event
설계에 사용한다.

### 5. Muse Voice Transcribe

공식 설명상 80 ms soft token, `<|next_audio|>`, `<|empty_audio|>`와 text/event token의 통합은
VAPASR 목표와 가장 가깝다. diarization과 speech onset/endpoint를 special token으로 추가하고,
마지막에는 WER와 delay reward를 결합한 RL로 adaptive delay를 만든다.

그러나 기술 보고서, 학습 데이터, 평가 set, weights가 공개되지 않았다. 3.1% WER는 가능성의
상한 신호일 뿐, 현재 RNN-T 4.4%/8.2%와 공정하게 비교할 baseline은 아니다. 당장은 API
black-box 평가와 설계 아이디어만 사용하고, RL은 1,930 h supervised scaling과 windowed target
실험 뒤로 둔다. 자세한 내용은 [[source-muse-voice-transcribe]]에 있다.

### 6. Alignment Restricted RNN-T

표준 RNN-T는 가능한 audio-text alignment를 합산하지만 늦게 방출하는 경로에도 확률을 둘 수
있다. Ar-RNN-T는 외부 정렬 주변의 left/right buffer 안에 있는 경로만 합산해 WER와 token
delay를 통제한다. 하나의 정답 시점만 강제하지 않고 범위 안의 여러 경로를 보존한다는 점이
현재 hard-δ보다 원칙적으로 낫다.

VAPASR에는 transducer lattice가 없으므로 구현은 달라야 한다. 현실적인 첫 ablation은
`[t_end+δ_min, t_end+δ_max]` 안의 여러 emission 시점에 확률을 나누는 soft target 또는
window 안 최선 시점의 loss를 쓰는 것이다. 조기 방출 금지선과 지연 허용선을 분리해야 한다.

### 7. Minimum Latency Training

MinLT는 transducer lattice 각 diagonal의 expected latency를 계산해 원래 loss와 함께
최적화한다. 단순히 “일찍 내라”는 gradient를 일괄 강화하기보다, 정확도 확률분포 안에서 기대
지연을 직접 줄인다는 점이 핵심이다.

decoder-only VAPASR에서는 각 audio tick의 `P(text)`와 `P(<NEXT_AUDIO>)`로 첫 올바른 token
방출 시점의 기대값을 구성해야 한다. 이는 구현·수치 안정성이 필요한 연구 항목이므로 즉시
1,930 h recipe에 섞지 않는다. 먼저 데이터 스케일링 곡선을 고정한 뒤, windowed target 다음
단계로 둔다.

## VAPASR 실험에 반영할 순서

1. **Stage 2A 기준선은 유지한다.** `asr-tn-v1.0.0`, 1,930 h, full thinker FT,
   80 ms clock, hard-δ를 한꺼번에 바꾸지 않고 스케일링 효과를 측정한다.
2. **정확도 격차가 남으면 windowed target을 먼저 비교한다.** Ar-RNN-T에서 가져오되
   decoder-only CE에 맞는 soft/window loss로 구현한다.
3. **그 다음 expected-latency loss를 추가한다.** 평균 지연뿐 아니라 `viol80`, p99,
   미방출 deadline recall을 함께 본다.
4. **learned MoChA read/write는 별도 구조 ablation으로 둔다.** 단순 목표 함수 변경보다
   개선 폭이 클 때만 복잡성을 감수한다.
5. **runtime에는 Samsung식 early-exit를 후보로 둔다.** 다만 먼저 deferred-control fusion과
   단일-stream 전용 GPU 측정으로 실제 병목을 확정한다.
6. **turn-taking 확장은 Moshi·Muse에서 가져온다.** Stage 2의 lexical ASR이 통과한 뒤
   audio-clock hidden state에 onset/endpoint/VAP head를 붙이고, display text 생성은 별도
   보조 목표로 유지한다.

## 원문

- Parcollet et al. (2026), [Streaming Speech-to-Text Translation with a SpeechLLM](https://arxiv.org/abs/2605.14766)
- Seide et al. (2024), [Speech ReaLLM — Real-time Streaming Speech Recognition with Multimodal LLMs by Teaching the Flow of Time](https://arxiv.org/abs/2406.09569)
- Wan et al. (2026), [Streaming Speech Recognition with Decoder-Only Large Language Models and Latency Optimization](https://arxiv.org/abs/2601.22779)
- Défossez et al. (2024), [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)
- Meta AI Research (2026), [Introducing Muse Voice Transcribe](https://research.meta.ai/blog/introducing-muse-voice-transcribe)
- Mahadeokar et al. (2021), [Alignment Restricted Streaming Recurrent Neural Network Transducer](https://arxiv.org/abs/2011.03072)
- Shinohara and Watanabe (2022), [Minimum latency training of sequence transducers for streaming end-to-end speech recognition](https://www.isca-archive.org/interspeech_2022/shinohara22_interspeech.html)

## 관련 내부 문서

- [[output-interleaved-streaming-slm-architecture]] — 현재 통합 구조와 실패 조건
- [[output-stage1-mono-pilot]] — Stage 1 실측과 Stage 2 관문
- [[source-muse-voice-transcribe]] — Muse 공식 소개 상세
- [[streaming-causality-and-latency-budget]] — 프로젝트 latency 회계
- [[turn-taking-objectives]] — 이후 turn-taking 목표
