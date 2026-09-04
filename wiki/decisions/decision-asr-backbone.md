---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-10-22
created: 2026-09-03
updated: 2026-09-04
summary: IS-SLM 최종 backbone은 Nemotron 3.5 FastConformer [56,0] → new adapter → Qwen3-ASR-0.6B-hf thinker LM이며 계획된 encoder 교체는 없다
sources:
  - [[source-nemotron-3-5-asr-streaming]]
  - [[source-qwen3-asr]]
  - [[source-muse-voice-transcribe]]
---

# 결정: streaming ASR backbone 선정

## 맥락

[[streaming-conversational-projection-asr]] 는 streaming ASR encoder 위에 turn
projection head 를 얹는다. backbone 선택이 frame rate, lookahead, 라이선스,
fine-tuning 자유도를 한꺼번에 결정한다.

## 검토한 선택지

| 선택지 | 장점 | 단점 |
|--------|------|------|
| **Nemotron 3.5 ASR Streaming 0.6B** | streaming engineering 완성. chunk 5단계(80~1120 ms)로 latency curve 즉시 확보. RNNT predictor state 명시적. ko-KR transcription-ready (7.12% WER @1.12s) | 라이선스 **OpenMDW-1.1** (파생물 배포 조건 확인 필요). LLM decoder 없음 |
| **Qwen3-ASR-0.6B** | **Apache 2.0**. AuT 180M, **12.5 Hz/80 ms 로 Muse 와 동일**. 공식 fine-tuning recipe. ForcedAligner 동반. LLM decoder hidden state 사용 가능 | **causality/lookahead 미문서화**. streaming 실측 필요 |
| Muse Voice Transcribe | 목표 architecture 그 자체 | **closed weights, API 전용.** fine-tune·ablation 불가 → **후보에서 제외** |
| 원 VAP CPC 유지 | 50 Hz 고해상도, 검증된 baseline | semantic 정보 없음. 연구 가설 자체를 검증할 수 없음 |

## 이전 결정 이력 (superseded)

아래 2단계안은 2026-09-04 최종 결정으로 대체되었다. 현재 실행 기준은 문서 하단의
**IS-SLM 최종 backbone** 절이다.

**2단계로 간다.**

1. **Phase A–B: Nemotron 3.5** — feasibility 와 [[acoustic-linguistic-fusion]] 검증.
   RNNT predictor state 가 명시적이라 H2 를 가장 적은 엔지니어링으로 시험할 수 있다.
2. **Phase C 이후: Qwen3-ASR-0.6B** — main 연구 모델. Apache 2.0 과 80 ms 정합성이
   장기적으로 결정적이다.

Muse 는 **backbone 이 아니라 (a) 설계 아이디어 출처, (b) API black-box 비교 대상**
으로만 쓴다.

## 조건 — 이 결정은 무조건적이지 않다

**[[task-audit-encoder-causality-lookahead]] 의 결과에 종속된다.**

- 두 encoder 중 하나라도 실효 lookahead 가 **320 ms 를 넘으면** 그 backbone 으로는
  [[source-turnbench]] 의 VAP 368 ms 를 구조적으로 이길 수 없다.
- 그 경우 [[streaming-causality-and-latency-budget]] 의 대응 3안(저해상도 semantic
  path + 50 Hz acoustic path 병합)으로 설계를 변경하고 이 결정을 `superseded` 로 바꾼다.

### 당시 결과 / 파급

- Phase A–B 코드는 NeMo 생태계에, Phase C 는 Transformers 생태계에 묶인다.
  **두 번 구현하는 비용**을 감수하는 대신 위험을 분산한다.
- 데이터 파이프라인과 평가 코드는 **backbone 에 독립적으로** 설계해야 이 비용이 줄어든다.
  → [[task-build-vap-target-pipeline]]
- 라이선스: 최종 공개 모델이 Nemotron 파생이면 OpenMDW-1.1 조건을 따라야 한다.
  Qwen 기반이면 Apache 2.0 으로 자유롭다. **이것이 main model 을 Qwen 으로 두는
  또 하나의 이유다.**

## 2026-09-03 감사 결과 — 조건 발동 (Qwen)

[[output-encoder-causality-audit]]:

- **Nemotron: 확인.** `[56,0]` 80 ms chunk 에서 lookahead ≤ 80 ms, `[56,1]` ≤ 160 ms.
  Phase A–B 는 그대로 진행하되 **chunk 는 80/160 ms 만 쓴다** (`[56,3]` 이상은 320 ms 관문 초과).
- **Qwen3 AuT: 조건 위반.** transformers/sdpa 경로는 마스크 미적용으로 비인과(발화 전체),
  의도된 블록 모드도 1 s 블록에서 lookahead 평균 420 ms(0–800) + 블록 간 좌측 context 부재.

### 수정된 결정

1. **Phase A–B (Paper 1): Nemotron 단일 backbone.** 변경 없음.
2. **Phase C (Paper 2) 의 Qwen 채택은 [[task-qwen-aut-causal-adaptation]] 결과에 종속.**
   작은 conv chunk 에서의 WER 열화와 causal/chunked fine-tune 타당성이 확인되어야 main backbone 이 된다.
   확인되지 않으면 Paper 2 도 Nemotron(또는 다른 causal encoder) 위에서 진행한다.
3. Stage 1 probing 에 AuT 를 넣을 경우 **1 s 블록 모드 + lookahead 0–800 ms 를 표에 명시** 한다.
   sdpa 기본 경로 특징은 사용 금지.
4. Qwen 의 당장의 가치는 **ForcedAligner(한국어 80 ms 정렬)** 와 **12.5 Hz 토큰 설계 참고** 다.
5. **단, 마스크 주입 실험이 길을 열었다** ([[output-encoder-causality-audit]] 추가 실험): 프레임 causal 마스크에서도
   lookahead 80 ms 에 단어 대부분이 보존된다(WER 23.5 %, 단일 발화). causal fine-tune 으로 회복되면 Qwen 은
   Paper 2 backbone 으로 복귀한다. 그래서 [[task-qwen-aut-causal-adaptation]] 을 p1 로 올린다.

## 2026-09-04 — IS-SLM 시작 backbone (사용자 질문에 대한 결정)

IS-SLM 이 단일 주력이 되면서 "backbone" 은 encoder + LLM 쌍이다. 근거는 지금까지의 실측이다.

| 후보 | 장점 | 단점 / 실측 |
|---|---|---|
| **Qwen3-ASR-0.6B 통째** (AuT + thinker) | LLM 이 **이미 AuT 임베딩으로 ASR 을 하도록 학습됨** → interleaving·`<NEXT_AUDIO>` 기계만 얹으면 됨(Muse 레시피와 가장 가까움). ForcedAligner 가 같은 tokenizer. Apache 2.0. INT 최강(0.945) | 기본 경로 비인과 → **마스크 주입 필수**. chunked-causal 1 s 블록·8 s 창에서 WER +5.9 %, lookahead 0–800 ms. **실외 잡음에 취약**(AI Hub 실외 CE 5.8–6.5 vs Nemotron 3.3) |
| Nemotron FastConformer `[56,0]` + Qwen3-0.6B 텍스트 LLM + 새 projector | causal 실측 ≤80 ms, 12.5 Hz, ko-KR ready, **잡음 강건**, 좌측 4.5 s context | LLM 이 이 임베딩으로 ASR 하는 법을 **처음부터** 배워야 함(SLAM-ASR 식, 360 h 로는 위험). NeMo↔Transformers 두 생태계. OpenMDW-1.1 |
| CPC (50 Hz) | 타이밍 최강, 5 M | ASR 정보 없음 — encoder 로는 부적합, **U3 사이드 브랜치**로만 |

**결정 (2026-09-04 사용자 확정) — 최종 backbone**

```text
Nemotron 3.5 FastConformer [56,0]  (causal ≤80 ms, 12.5 Hz, 잡음 강건, ko-KR)
        → new adapter (Nemotron 1024-d → thinker 오디오 임베딩 공간, 12.5→13 Hz 리샘플)
        → Qwen3-ASR-0.6B-hf thinker LM (LoRA)  — 오디오 임베딩 → 텍스트 ASR 사전지식
```

이 조합을 **최종**으로 한다. 제안했던 "AuT + thinker 시작 → U3 교체" 는 어차피 U3 에서 같은 adapter 작업을 요구하므로, 그 위험을 앞당겨
지는 대신 첫날부터 진짜 스트리밍(≤80 ms)과 잡음 강건성을 얻는다.

1. **U0.5 adapter bridge test (관문, GPU ≈ 1 일)** — 캐시된 Nemotron·Qwen audio tower 특징(205 h 동일 대화)으로 adapter 를 **표현 증류**로 초기화
   (Nemotron → 기존 Qwen audio tower의 thinker 입력 임베딩 회귀, cosine + L2, 라벨 불필요) → 오프라인 ASR 손실로 adapter + thinker LoRA 짧게 학습 →
   WER 을 AuT+thinker 오프라인, Nemotron RNN-T 와 비교. **10–15 % 이내면 통과.** 회귀 잔차가 크면 조기 경고.
2. 통과 시 U1(interleaved streaming ASR) 로 간다. 실패하면 adapter 용량·시간 정렬·증류 목표·encoder/thinker unfreeze 범위를 재설계한다.
   **AuT+thinker로 자동 교체하지 않는다.** backbone 변경은 실험 관문에 내장된 fallback이 아니라 별도의 사용자 결정이 필요한 범위 변경이다.
3. 세부: 12.5 → 13 Hz nearest 리샘플 기본(12.5 그대로는 ablation), 두 화자는 화자별 인코딩 → merge → adapter → joint chunk token 1개,
   LLM 0.6B 로 시작(p99 tick < 80 ms), 50 Hz CPC 사이드 브랜치는 U3 에서 dense 헤드에.
4. 라이선스: OpenMDW-1.1(Nemotron) + Apache 2.0(Qwen3-ASR) — 배포 시 두 조건 병기.
5. [[task-qwen-aut-causal-adaptation]]은 주 경로에서 취소한다. Qwen AuT 결과는 비교 근거로 보존하지만 causal 적응이나 향후 encoder 교체를 일정에 넣지 않는다.

## 재검토


2026-10-22 — [[task-stage1-encoder-probing]] 결과가 나오는 시점.
