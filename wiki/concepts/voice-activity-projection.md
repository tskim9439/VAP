---
type: concept
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: 미래 2초 구간의 양 화자 음성 활동을 256-class로 예측하는 turn-taking 모델 — 이 연구의 출발 baseline
sources:
  - [[source-chatgpt-research-plan]]
  - [[source-turnbench]]
---

# Voice Activity Projection (VAP)

## 정의

매 시점 `t` 에서 **앞으로 2초 동안 두 화자가 각각 언제 말할지** 를 예측하는 모델.
미래 2초를 4개 구간으로 나눈다:

```text
[0, 0.2]  [0.2, 0.6]  [0.6, 1.2]  [1.2, 2.0]   (초)
```

화자 2명 × 구간 4개 = **8 bit → 256 class** 의 CE 문제로 푼다.

```
L_VAP = CE(y_256, ŷ)
```

원 VAP 는 CPC encoder 위에서 동작하며 **50 Hz (20 ms/frame)** 이다.
화자별 오디오가 분리되어 있다고 가정한다 (dual-channel).

## 왜 endpointing 과 다른가

이것이 이 연구의 novelty 근거다.

| | 묻는 것 | 시점 |
|---|---------|------|
| Endpointing (VAD/EPD) | "지금 발화가 끝났는가?" | 현재 |
| VAP | "앞으로 2초간 누가 말할 것인가?" | **미래** |

[[source-muse-voice-transcribe]] 의 `|speech_endpoint|` 는 앞의 것이다.
detection 은 사후 반응이고 projection 은 사전 예측이므로, 사람 수준의
**턴 종료 −151 ms 선점**([[source-turnbench]])에 도달하려면 projection 이 필요하다.

## 현재 위치

[[source-turnbench]] 기준 VAP 는 EOT·INT 양 트랙에서 여전히 최강 baseline 이다
(EOT recall 0.845 @ FPR 0.055, p50 368 ms). 그러나:

- 사람은 −151 ms 에 이미 움직인다 → **약 520 ms 의 격차**.
- [[source-turn-taking-related-work-2026]] 의 **DualTurn 이 agent action 예측에서
  VAP 를 크게 앞섰다** (F1 0.633 vs 0.389). VAP 만 이기는 결과는 더 이상 충분하지 않다.

## 알려진 약점

1. **semantic 정보 부재.** CPC 는 음향만 본다. "그런데 제가 말씀드리고 싶은 것은…"
   뒤의 멈춤과 "네, 알겠습니다." 뒤의 멈춤을 음향만으로 구분하기 어렵다.
   → [[acoustic-linguistic-fusion]]
2. **class imbalance.** 256-class CE 는 "둘 다 침묵" / "A 계속 발화" 패턴이 지배한다.
   EOT·interruption 은 희귀 이벤트다. → 이벤트 단위 평가 필수
   ([[turn-taking-evaluation-protocol]]).
3. **binary 하지 않은 시간 정보를 버린다.** 언제 시작할지의 *거리* 정보가
   256-class 안에 구간으로만 뭉뚱그려진다. → [[turn-taking-objectives]] 의 τ head.

## 관련

- [[streaming-conversational-projection-asr]] — 이 연구가 제안하는 확장
- [[turn-taking-objectives]] — objective 설계
- [[streaming-causality-and-latency-budget]] — 80 ms backbone 으로 옮길 때의 제약
