---
type: output
status: active
created: 2026-09-03
updated: 2026-09-04
summary: 검증된 연구 기반 계획 v2이며 모델 단계는 IS-SLM과 최종 Nemotron–adapter–Qwen thinker backbone 결정으로 대체됨
sources:
  - [[source-chatgpt-research-plan]]
  - [[source-turnbench]]
  - [[source-turn-taking-related-work-2026]]
  - [[source-conversation-corpora]]
---

# 연구 계획 v2 — Streaming Conversational Projection ASR

> **2026-09-04 갱신**: 목표 모델은 [[output-interleaved-streaming-slm-architecture]] (IS-SLM) 단일 주력. 아래 3절·4절의 Stage 2–3(RNN-T 위 VAP head 융합)은
> 폐기되고 U1–U3 으로 대체됐다 → [[decision-target-architecture]]. Stage 1(표현 비교)·Phase 3–4(objective·한국어)는 유효.
> backbone은 Nemotron `[56,0]` → new adapter → Qwen3-ASR thinker로 최종 확정했으며, Qwen AuT로의 계획된 교체는 없다 → [[decision-asr-backbone]].

[[source-chatgpt-research-plan]] 초안을 사실 검증한 뒤 수정한 계획.

## 요약

초안의 **방향은 타당하다.** endpointing 과 projection 을 구분한 것,
strict-streaming encoder 우선, cascade 회피, 단계적 검증이라는 원칙은 옳다. 다만 이후 결정으로
Nemotron → Qwen AuT 순차 교체는 폐기하고 동일한 Nemotron–adapter–Qwen thinker backbone을 유지한다.
인용한 자료도 **전부 실재**했다 (초안이 지어낸 참고문헌은 없었다).

바꿔야 할 것은 방향이 아니라 **위험 통제와 범위**다. 다섯 가지다.

| # | 문제 | 조치 |
|---|------|------|
| 1 | **DualTurn 누락** — VAP 를 F1 0.633 vs 0.389 로 이긴 경쟁 가설 | 필수 baseline + 비교 축에 추가 |
| 2 | **causality/lookahead 미검증** — 비교 자체가 무효화될 수 있음 | 모든 모델링에 선행하는 감사 |
| 3 | **Muse 는 closed weights** | backbone 후보에서 제외, API 비교로 격하 |
| 4 | **AI Hub 재배포 제약** | 벤치마크 공개 범위를 어노테이션 레이어로 재설계 |
| 5 | **범위가 논문 2~3편치** | Paper 1 / Paper 2 로 명시 분할 |

---

## 1. 유지하는 것 (초안이 옳았던 부분)

- **주제 설정.** [[voice-activity-projection]] 의 future projection 과
  Muse 의 endpointing 은 다른 문제이며 그 차이가 novelty 다. 타당하다.
- **cascade 회피.** ASR 텍스트를 별도 text 모델에 넣지 않고 내부 state 만 공유한다.
  → [[acoustic-linguistic-fusion]]
- **binary EOT 지양.** future activity + time-to-next-turn 을 함께 예측한다.
  [[source-turn-taking-related-work-2026]] 의 Next-Turn 이 뒷받침한다 (+25.9%p).
- **고정 backbone + 단계적 학습 전략.** Nemotron `[56,0]`과 Qwen thinker 사이 adapter를 먼저 검증하고 같은 그래프를 끝까지 유지한다. → [[decision-asr-backbone]]
- **한국어 데이터 판단.** AI Hub 감정 태깅 자유대화가 stereo + 발화 타임스탬프를
  갖는다는 것은 **확인되었다.** VAP 학습에 적합하다는 판단은 맞다.
- **Qwen3-ForcedAligner 활용.** 80 ms 해상도가 AuT·Muse 와 정확히 일치한다.

## 2. 반드시 고칠 것

### 2.1 DualTurn 을 baseline 에 넣는다 (가장 시급)

[[source-turn-taking-related-work-2026]] 의 **DualTurn** (arXiv 2603.08216) 은
dual-channel **generative** pretraining 으로 VAP 를 크게 앞섰다.
이 연구의 H1("ASR pretraining 이 좋다")과 **정면 경쟁하는 대안 가설**이다.

- **VAP 만 이기는 결과는 더 이상 논문이 되지 않는다.**
- [[question-asr-representation-vs-ssl-for-vap]] 의 비교를
  `ASR vs SSL vs generative dual-channel` 3자 구도로 확장한다.
- 두 pretraining 은 배타적이지 않다. **결합이 오히려 novelty 가 될 수 있다.**

### 2.2 causality 감사를 모든 것에 앞세운다

→ [[streaming-causality-and-latency-budget]]

절단 실험으로 실효 lookahead 를 측정한다. lookahead > 320 ms 면
VAP 의 368 ms 를 **구조적으로** 이길 수 없고 [[decision-asr-backbone]] 이 뒤집힌다.
코드 몇 줄이면 되는 일이 전체 프로젝트의 전제를 결정한다.

frame rate 하락(50 Hz → 12.5 Hz, **4배**)도 함께 회계한다.
`latency = frame_period/2 + lookahead + compute`.

### 2.3 Stage 1 실험의 교란 변수를 통제한다

초안의 "encoder 만 바꾼다" 는 **ASR pretraining 효과와 모델 크기·데이터량 효과를
구분하지 못한다.** 다음을 추가한다: 공통 frame rate 리샘플, probe 용량 고정,
크기 대조군(WavLM Base vs Large), **random-init floor**, encoder 별 pretraining
시간·lookahead 명시. → [[question-asr-representation-vs-ssl-for-vap]]

### 2.4 τ 를 생존분석으로 정식화한다

초안의 `τ_t = t_next_onset − t` bucket 분류는 화자 지정, overlap, 동일 화자 재개,
그리고 **right-censoring** 에서 모호하다. **discrete-time hazard** 로 바꾸면
censoring 이 원리적으로 처리되고 `P(EOT before Δ)` 를 임의 시점에 계산할 수 있어
평가 지표와 직결된다. → [[turn-taking-objectives]]

### 2.5 손실 균형과 ASR 회귀 가드레일

5개 손실의 스케일이 다르다. 고정 λ 대신 **uncertainty weighting / GradNorm**.
그리고 **unfreeze 후 WER/CER 회귀를 항상 함께 보고**한다 — turn 성능이 올라도
ASR 이 망가지면 통합 모델의 존재 이유가 없다.

### 2.6 누락된 baseline 3종

초안에 없다. 리뷰어가 반드시 요구한다.

1. **VAD + silence threshold** — 산업 표준. 이걸 못 이기면 아무 의미가 없다.
2. **cascade (ASR → 텍스트 turn 모델)** — 초안은 측정 없이 "느리다"고 기각했다.
   **실제로 재서** joint 가 번 것을 정량화한다.
3. **DualTurn**, **JAL-Turn**.

### 2.7 이벤트 라벨을 먼저 검증한다

AI Hub·CANDOR 에는 EOT/HOLD/INT/BACKCHANNEL 라벨이 **없다.** 휴리스틱으로 유도해야
하는데, 검증 없이 쓰면 결론이 오염된다. 다행히 **otoSpeech 는 사람이 라벨했으므로
정답지가 있다.** → [[question-event-label-derivation-validity]]

### 2.8 벤치마크 공개 범위 재설계

AI Hub 재배포 제약 때문에 오디오 포함 공개가 불가능하다.
**"Korean TurnBench"가 아니라 "어노테이션 레이어 + 평가 툴킷"** 으로 포지셔닝한다.
→ [[decision-korean-benchmark-release-scope]]

### 2.9 데이터 규모를 낮춰 시작한다

초안은 KO 1,000시간을 권했지만 **Stage 1 probing 에는 50–100시간이면 충분하다.**
encoder 순위는 그 규모에서 안정된다. 결론을 먼저 내고 데이터를 늘린다.
추가로 **frozen encoder 출력을 디스크에 캐시** 하면 probing 실험이 며칠에서 분 단위로 줄어든다.

---

## 3. 논문 분할

초안의 6단계는 한 편에 담기지 않는다. 명시적으로 나눈다.

### Paper 1 — "Do streaming ASR representations project conversational futures?"

- Stage 1 encoder probing (통제된 비교, DualTurn 포함)
- Stage 2–3 acoustic-linguistic fusion + WER 가드레일
- latency-quality curve (80/160/320/640 ms) + lookahead 회계
- 한국어 어노테이션 레이어 + 한국어/영어 대조 분석
- **핵심 주장**: representation 선택이 turn projection 에 미치는 영향을
  교란 없이 처음으로 분리했다 + 한국어 평가 자원

### Paper 2 — Unified streaming perception model

- Qwen3-ASR 기반 통합 모델, τ hazard head, event head
- bilingual joint training
- Muse-style adaptive emission (WAIT/EMIT), delay-aware RL
- **Paper 1 결과가 H1 을 지지할 때만 착수한다.**

Stage 6 RL 은 Paper 2 의 마지막이거나 future work 다.

---

## 4. 수정된 로드맵

```text
Phase 0  검증 (9월)          데이터 접근 · causality 감사 · 코퍼스 확보
   │                         ⚠ 여기서 막히면 전부 막힌다
Phase 1  Stage 1 (10월)      encoder probing + baseline 재현 + latency curve
   │                         → Paper 1 의 핵심 결과
Phase 2  Fusion (11월)       multitask VAP + linguistic state + 손실 균형
Phase 3  Objective (12월)    τ hazard head + event head
Phase 4  한국어 (12~1월)     어노테이션 설계 → 수행
Phase 5  Bilingual/robustness (2월) temperature sampling + 잡음 강건성
```

## 5. 평가

→ [[turn-taking-evaluation-protocol]] ([[source-turnbench]] 규약 그대로 채택)

**최소 성공 기준**: 동일 FPR 에서 VAP 와 같은 recall 을 유지하며 p50 latency 를
유의하게 낮춘다. 목표 지점은 사람의 −151 ms 와 VAP 의 368 ms 사이.

## 6. 실행

17개 태스크로 분해했다. → [[todo]]

가장 먼저 할 3개 (모두 p0, 서로 독립이라 병렬 가능):

1. [[task-verify-aihub-stereo-and-access]] — 데이터가 없으면 아무것도 못 한다
2. [[task-audit-encoder-causality-lookahead]] — 결과가 backbone 결정을 뒤집을 수 있다
3. [[task-reproduce-vap-turnbench-baseline]] — 비교 기준선이 없으면 측정이 무의미하다
