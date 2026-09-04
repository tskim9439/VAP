---
type: source
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: Meta Muse Voice Transcribe — 80ms soft token 기반 streaming ASR+diarization+endpointing 통합 모델, closed weights
url: https://research.meta.ai/blog/introducing-muse-voice-transcribe
observed: 2026-09-03
---

# Muse Voice Transcribe (Meta Superintelligence Labs)

## 무엇인가

streaming ASR + diarization(20+ speaker) + endpointing 을 **하나의 autoregressive
multimodal 모델** 로 푸는 실시간 음성 인식 모델. Muse Spark 계열.

## 작동 방식

- audio 를 **80 ms 단위(12.5 chunk/s)** 로 받아 각각을 soft token 하나로 변환.
- 매 chunk 마다 "더 들을지 / text token 을 낼지" 를 결정한다.
  계속 들으면 `|next_audio|` 를 예측하고, 그 자리에 실제 다음 audio chunk 가 들어간다.
  스트림이 끝나면 `|empty_audio|`.
- endpointing 은 `|speech_onset|` / `|speech_endpoint|` special token.
- diarization·endpointing 을 streaming ASR 과 **함께 학습** 하고, ASR reward 위에
  각각의 reward 를 얹는다.
- 최종적으로 **WER + delay 를 결합한 RL** 로 adaptive delay 를 최적화.
  최종 transcription WER 3.1%.

## 연구 계획에 미치는 영향 (중요)

**weights 가 공개되지 않는다.** Meta Model API 의 `muse-voice-transcribe-1.0`
로만 접근 가능하며 ($3.00 / 1,000 audio min = $0.18/h), self-host·on-prem 경로가
없다. 같은 해 Muse Glimmer 는 공개했으므로 이는 의도적 선택이다.

따라서:

- Muse 를 **backbone 으로 쓰거나 fine-tune 하거나 ablation 할 수 없다.**
- 쓸 수 있는 것은 (a) **architecture 아이디어**(80 ms soft token, listen/write
  policy, special token 통합, delay-aware RL) 와 (b) **API 를 통한 black-box
  비교**(WER, endpoint latency) 뿐이다.
- "Muse-style adaptive emission" 단계는 Muse 재현이 아니라 **Qwen3-ASR 위에서의
  독자 구현** 이어야 한다. → [[decision-asr-backbone]]

## 핵심 차이

Muse 는 **"무엇을 듣고 있으며 언제 끝났는가"** 를 푼다. 이 볼트의 연구 목표인
[[streaming-conversational-projection-asr]] 는 **"앞으로 누가 언제 말할 것인가"**
까지 예측한다. 이 구분이 novelty 의 근거다.
