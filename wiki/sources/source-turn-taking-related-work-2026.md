---
type: source
status: active
created: 2026-09-03
updated: 2026-09-03
summary: 2026년 turn-taking 관련 연구 3편 — JAL-Turn, Next-Turn, DualTurn의 기여와 baseline 위치
observed: 2026-09-03
---

# 2026 turn-taking 관련 연구 묶음

## JAL-Turn (arXiv 2603.26515, 2026-03-27)

Guangzhao Yang 외. *Joint Acoustic-Linguistic Modeling for Real-Time and Robust
Turn-Taking Detection in Full-Duplex Spoken Dialogue Systems.*

- pretrained acoustic representation + linguistic feature 를 **cross-attention** 으로 결합.
- **hold vs shift** 저지연 예측. ASR 과 **병렬** 실행되어 추가 latency 가 없다.
- 실제 dialogue 에서 turn-taking 라벨을 뽑는 **자동 데이터 파이프라인**.
- 다국어 benchmark + 일본어 고객센터 데이터에서 평가.

**이 볼트와의 관계**: [[acoustic-linguistic-fusion]] 의 직접적 선행 연구.
차이는 JAL-Turn 이 hold/shift **이진 분류** 에 머무는 반면, 이 연구는
**future projection** 까지 간다는 점이다. 반드시 baseline 으로 재현해야 한다.

## Next-Turn (arXiv 2606.18094, 2026-06-16)

Tristan Tsoi 외. *Duration-Aware Streaming Endpoint Detection via
Time-to-Next-Speech-Onset Prediction.*

- **time-to-next-speech-onset 을 학습 목표**로 삼는다.
- **320 ms 이내 endpoint accuracy 에서 최강 baseline 대비 +25.9%p (절대값).**
- duration-aware objective 는 표준 binary EPD 를 **대체가 아니라 보완** 한다
  (joint training 이 유효).
- 동기: 화자는 hesitation·disfluency 때문에 turn 중간에도 멈춘다.

**이 볼트와의 관계**: [[turn-taking-objectives]] 의 τ head 근거.
"binary EOT 만 쓰지 말라" 는 초안의 판단은 이 논문이 뒷받침한다.

## DualTurn (arXiv 2603.08216, 2026-03-09) — **초안이 놓친 핵심 경쟁 연구**

Shangeth Rajaa. *Learning Turn-Taking from Dual-Channel Generative Speech
Pretraining.*

- dual-channel 대화 오디오에서 **양쪽 화자의 미래 audio 를 autoregressive 하게 생성**
  하는 방식으로 pretraining. **수동 라벨 불필요.**
- 이후 5개 agent action 으로 fine-tune. 0.5B.
- **VAP 대비 agent action 예측 weighted F1 0.633 vs 0.389.**
- 3.1B audio-text 모델보다 word-level turn 예측 AUC 우세 (0.930 vs 0.880).
- turn boundary 를 더 일찍 예측하면서 interruption 은 더 적다.

**왜 중요한가**: "ASR pretraining 이 turn-taking representation 에 좋다" 는 이 연구의
가설과 **정면으로 경쟁하는 대안 가설** 이다 — DualTurn 은 ASR 이 아니라
**generative dual-channel pretraining** 이 답이라고 주장한다. 게다가 VAP 를 큰 폭으로
이겼다. 따라서:

1. DualTurn 은 **필수 baseline** 이다. VAP 만 이기는 결과는 더 이상 충분하지 않다.
2. [[question-asr-representation-vs-ssl-for-vap]] 의 비교 축에
   **DualTurn encoder 를 반드시 포함** 해야 한다.
3. 두 pretraining 목표는 상호 배타적이지 않다 — 결합이 오히려 novelty 가 될 수 있다.
