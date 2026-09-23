# Streaming ASR Semantic Commit 데이터 구축 및 학습 계획

## 1. 목표

본 프로젝트의 목표는 한국어와 영어 구어체 음성을 대상으로 하는 **streamable ASR 모델**이 실시간으로 전사를 생성하면서, 단순한 문장부호 복원이 아니라 **현재까지의 전사 결과를 의미적으로 확정(commit)해도 되는 시점**을 함께 예측하도록 학습하는 것이다.

핵심 목표는 다음과 같다.

1. 실시간 스트리밍 ASR 전사
2. 구어체의 머뭇거림, 반복, self-repair, filler에 강건한 semantic segmentation
3. 의미적으로 충분한 정보가 형성된 시점의 탐지
4. 별도 classification head 없이 **sequence token 방식**으로만 학습
5. 한국어/영어를 하나의 공통 semantic commit 개념으로 학습
6. 이후 downstream LLM prefill, turn-taking, full-duplex speech system과 연결 가능하도록 설계

---

# 2. 핵심 설계 원칙

## 2.1 Auxiliary head를 사용하지 않는다

Semantic boundary를 별도의 classifier/head로 예측하지 않는다.

ASR의 autoregressive sequence 안에 semantic event token을 직접 삽입한다.

예:

```text
오늘
오전에
회의를
했어요
<SEM_END>
<NEXT_AUDIO>
...
```

모델이 학습해야 하는 경쟁 관계는 다음과 같다.

\[
P(y_t = \texttt{<SEM_END>} \mid audio_{\le t}, y_{<t})
\]

대

\[
P(y_t = \texttt{<NEXT_AUDIO>} \mid audio_{\le t}, y_{<t})
\]

즉 모델은 매 시점마다 사실상 다음 중 하나를 결정한다.

```text
더 듣는다       → <NEXT_AUDIO>
의미를 확정한다 → <SEM_END>
```

---

## 2.2 Semantic End와 Turn End는 분리한다

두 event는 의미가 다르다.

```text
<SEM_END>
```

- 지금까지 생성된 발화가 독립적으로 이해 가능한 semantic unit
- 이후 발화가 나오더라도 기존 의미를 수정할 가능성이 충분히 낮음
- downstream LLM 등에 irreversible commit해도 되는 시점

```text
<TURN_END>
```

- 화자가 실제 turn을 종료한 시점
- acoustic/conversational event

따라서 아래와 같은 sequence가 가능하다.

```text
오늘 오전에 회의를 했어요 <SEM_END>
그리고 점심을 먹었어요 <SEM_END>
그 다음 회사로 돌아왔어요 <SEM_END> <TURN_END>
```

반대로 발화를 중단했지만 의미가 완성되지 않은 경우:

```text
그게 그러니까... <TURN_END>
```

처럼 `<SEM_END>` 없이 `<TURN_END>`만 나올 수 있다.

---

# 3. `<SEM_END>`의 정의

`<SEM_END>`는 단순한 문장 끝이나 punctuation boundary가 아니다.

다음 두 조건을 모두 만족해야 한다.

\[
Complete(prefix_t)=1
\]

\[
Stable(prefix_t \mid future)=1
\]

즉,

> 현재 prefix가 독립적인 의미 단위를 형성하며, 이후 발화가 추가되더라도 이미 확정한 의미를 수정·취소·재해석할 가능성이 충분히 낮은가?

를 기준으로 한다.

예:

```text
오늘 오후에 회의를 했어요 <SEM_END>
```

하지만:

```text
오늘 오후에 회의를 했는데
```

는 일반적으로 아직 commit하지 않는다.

또한:

```text
내일...
아니 오늘 오후에
```

같은 self-repair를 고려하면 `내일` 뒤에 긴 pause가 있더라도 `<SEM_END>`를 넣지 않는다.

---

# 4. Annotation 기본 원칙

## 4.1 원본 구어체를 최대한 보존한다

ASR 학습용 transcript에서는 다음을 임의로 제거하거나 정규화하지 않는다.

- filler
- repetition
- hesitation
- self-repair
- word fragment
- discourse marker

예:

```text
어 저는 그 내일 아니 오늘 오후에 병원에 갔다가요
음 회사로 바로 갈 것 같아요
```

를

```text
저는 오늘 오후에 병원에 갔다가 회사로 바로 갈 것 같아요
```

로 clean-up하지 않는다.

Semantic annotation은 speech fidelity를 유지한 상태에서 추가한다.

```text
어 저는 그 내일 아니 오늘 오후에 병원에 갔다가요
음 회사로 바로 갈 것 같아요 <SEM_END>
```

---

# 5. Transcript Annotation 규칙

아래 규칙을 LLM teacher와 human annotator가 공통으로 사용한다.

## Rule 1. 실제 발화를 보존한다

filler, 반복, self-repair를 임의로 삭제하거나 교정하지 않는다.

```text
어 제가 그...
```

를 그대로 유지한다.

---

## Rule 2. Semantic completeness가 있을 때만 `<SEM_END>`를 넣는다

```text
오늘 회의했어요 <SEM_END>
```

---

## Rule 3. Future-stability가 확보되어야 한다

이후 발화가 기존 의미를 수정할 가능성이 높다면 commit하지 않는다.

```text
내일...
아니 오늘
```

중간에는 `<SEM_END>`를 넣지 않는다.

---

## Rule 4. Silence 또는 hesitation 자체는 boundary가 아니다

```text
병원에 갔다가...
```

뒤에 긴 pause가 있더라도 의미가 불완전하면 WAIT이다.

---

## Rule 5. Filler 직후에는 원칙적으로 boundary를 두지 않는다

```text
음...
어...
그...
```

는 `<SEM_END>`가 아니다.

---

## Rule 6. Self-repair 내부에는 boundary를 두지 않는다

```text
금요일 아니 목요일
```

에서 `금요일` 뒤에 `<SEM_END>`를 넣지 않는다.

---

## Rule 7. Repetition 또는 stutter 내부에는 boundary를 두지 않는다

```text
내 내일
그 그 사람이
```

중간에는 commit하지 않는다.

---

## Rule 8. Continuation을 요구하는 연결 구조 뒤에는 boundary를 두지 않는다

한국어 예:

```text
-고
-다가
-면서
-려고
-니까
-아서 / -어서
-면
-지만
-거나
-도록
```

영어 예:

```text
because ...
if ...
when ...
although ...
I think that ...
the person who ...
```

단, 형태소 또는 접속사 자체를 hard rule로 사용하지는 않는다.

---

## Rule 9. 문법 형태보다 pragmatic completion을 우선한다

한국어:

```text
저는 그렇게 생각 안 하는데요. <SEM_END>
```

처럼 `-는데요`라도 실제 화행이 완결되면 commit할 수 있다.

영어:

```text
Probably tomorrow. <SEM_END>
```

처럼 완전한 문장이 아니어도 대화 문맥에서 답변이 완결되면 commit할 수 있다.

---

## Rule 10. 보문·인용·embedding 내부에는 boundary를 두지 않는다

```text
그 사람이 온다고 생각해요
```

에서:

```text
그 사람이 온다고 <SEM_END>
```

는 금지한다.

---

## Rule 11. Discourse marker는 다음 semantic unit에 포함한다

```text
오늘 회의했어요 <SEM_END>
그리고 점심을 먹었어요 <SEM_END>
```

`그리고`, `근데`, `but`, `and`, `so` 등은 원칙적으로 다음 unit에 포함한다.

---

## Rule 12. Enumeration은 항목 의미가 완결된 경우에만 분리한다

```text
첫째 가격이 싸고 둘째 속도가 빠릅니다
```

에서 단순히 `싸고` 뒤에 `<SEM_END>`를 넣지 않는다.

---

## Rule 13. Fragment라도 pragmatic하게 완결되면 commit할 수 있다

한국어:

```text
Q: 몇 시에요?
A: 세 시쯤이요 <SEM_END>
```

영어:

```text
Q: When?
A: Tomorrow morning <SEM_END>
```

---

## Rule 14. Turn은 끝났지만 의미가 불완전할 수 있다

```text
그게 그러니까... <TURN_END>
```

이 경우 `<SEM_END>`는 넣지 않는다.

---

## Rule 15. 애매하면 commit하지 않는다

Semantic commit은 **Precision 우선**으로 설계한다.

\[
Cost(False\ Positive) \gg Cost(False\ Negative)
\]

Premature commit은 downstream LLM에 이미 잘못된 prefix를 넘길 수 있기 때문에, 조금 늦게 commit하는 것보다 위험하다.

---

# 6. Annotation Metadata

최종 모델은 `<SEM_END>`만 학습하더라도 annotation 단계에서는 더 풍부한 metadata를 유지한다.

예:

```json
{
  "text": "어 저는 내일 아니 오늘 오후에 병원에 갔다가 회사에 갈 것 같아요",
  "words": [
    {"id": 0, "text": "어"},
    {"id": 1, "text": "저는"},
    {"id": 2, "text": "내일"},
    {"id": 3, "text": "아니"},
    {"id": 4, "text": "오늘"}
  ],
  "disfluencies": [
    {
      "type": "filler",
      "start": 0,
      "end": 0
    },
    {
      "type": "self_repair",
      "reparandum": [2, 2],
      "repair": [3, 4]
    }
  ],
  "semantic_boundaries": [
    {
      "after_word": 12,
      "confidence": 0.97
    }
  ]
}
```

중요한 원칙은 **LLM에게 수정된 transcript를 생성하게 하지 않는 것**이다.

항상 word/eojeol index 기반으로 annotation을 반환하게 한다.

---

# 7. 전체 LLM Relabeling Pipeline

```text
                    Audio
                      │
                      ▼
             Original Transcript
                      │
                      ▼
          Transcript QC / Consensus
                      │
                      ▼
              Word/Eojeol Indexing
                      │
                      ▼
       ┌──────────────────────────┐
       │ Stage A                  │
       │ Full-context Annotation  │
       │                          │
       │ - filler                 │
       │ - repetition             │
       │ - self-repair            │
       │ - SEM_END candidates     │
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ Stage B                  │
       │ Causal Prefix Judge      │
       │                          │
       │ SAFE / WAIT / UNCERTAIN  │
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ Stage C                  │
       │ Future Stability Judge   │
       │                          │
       │ STABLE / REVISION        │
       └────────────┬─────────────┘
                    │
                    ▼
           Confidence Filtering
                    │
                    ▼
              <SEM_END>
                    │
                    ▼
        Streaming Training Sequence
```

---

# 8. Stage A — Full-context Annotation

Stage A는 전체 transcript를 보고 semantic unit 후보를 생성한다.

입력 예:

```text
[000] 어
[001] 저는
[002] 내일
[003] 아니
[004] 오늘
[005] 오후에
[006] 병원에
[007] 갔다가요
[008] 음
[009] 회사로
[010] 바로
[011] 갈
[012] 것
[013] 같아요
[014] 그리고
[015] 저녁에는
[016] 친구를
[017] 만나려고요
```

출력:

```json
{
  "semantic_boundaries": [13, 17],
  "fillers": [[0, 0], [8, 8]],
  "repairs": [
    {
      "reparandum": [2, 2],
      "repair": [3, 4]
    }
  ]
}
```

Stage A는 전체 문맥을 볼 수 있기 때문에 **좋은 segmentation 후보를 만드는 역할**만 한다.

Stage A 결과를 바로 `<SEM_END>` GT로 사용하지 않는다.

---

# 9. Stage B — Causal Prefix Judge

Stage A에서 생성한 candidate마다 미래 transcript를 제거하고 prefix만 보여준다.

예:

```text
저는 오늘 오후에 병원에 갔다가
```

판단:

```text
WAIT
```

반면:

```text
저는 오늘 오후에 병원에 다녀왔어요
```

판단:

```text
SAFE
```

출력 형식:

```json
{
  "decision": "SAFE"
}
```

가능하면 자유형 reasoning은 저장하지 않는다.

Stage B의 목적은:

> 실제 streaming 시점에서 미래를 모르는 모델이 지금 commit하는 것이 타당한가?

를 검증하는 것이다.

---

# 10. Stage C — Future Stability Judge

Prefix만으로 완결되어 보이더라도 실제 미래 발화에서 self-repair가 발생할 수 있다.

예:

```text
PREFIX:
내일 갈게요

FUTURE:
아 아니 오늘 갈게요
```

결과:

```json
{
  "relation": "REVISION",
  "type": "SELF_REPAIR"
}
```

반면:

```text
PREFIX:
내일 갈게요

FUTURE:
김 대리도 같이 간다고 하네요
```

결과:

```json
{
  "relation": "STABLE",
  "type": "ADDITIONAL_INFORMATION"
}
```

최종 조건은 개념적으로 다음과 같다.

\[
SEM\_END_i =
A_i \land B_i \land \neg C_i
\]

- \(A_i\): Full-context semantic boundary
- \(B_i\): prefix-only에서도 SAFE
- \(C_i\): future가 기존 의미를 수정함

---

# 11. Future Window

Stage C에서 future를 무한히 보여줄 필요는 없다.

초기 설정:

```text
최대 10~15 어절
또는
다음 semantic candidate
또는
약 2~3초 상당의 transcript
```

중 먼저 도달하는 것을 사용한다.

Self-repair가 실제로 얼마나 멀리 발생하는지는 데이터 분석 후 조정한다.

---

# 12. 한국어 Annotation 특성

한국어에서는 다음 표현을 strong continuation cue로 활용할 수 있다.

```text
-고
-다가
-면서
-려고
-니까
-아서 / -어서
-면
-지만
-거나
-도록
```

하지만 hard rule로 쓰지 않는다.

예:

```text
제가 병원에 갔다가
→ WAIT
```

```text
제가 병원에 갔다가 회사에 왔어요
→ SAFE
```

반대로:

```text
저는 별로 그렇게 생각 안 하는데요
→ SAFE 가능
```

처럼 종결 화행으로 사용되는 경우가 있기 때문이다.

---

# 13. 영어 Annotation 특성

영어에서는 다음 유형을 별도 hard-negative category로 관리한다.

## Subordinate clause

```text
because I thought...
→ WAIT
```

```text
if we go tomorrow...
→ WAIT
```

## Filled pause

```text
I think we should...
uh...
probably wait.
```

중간에는 commit하지 않는다.

## Repair

```text
Let's meet Friday—
no, Thursday.
```

`Friday` 뒤에는 `<SEM_END>`를 넣지 않는다.

## Pragmatic fragment

```text
Q: When?
A: Tomorrow morning <SEM_END>
```

## Discourse marker

```text
That's probably fine <SEM_END>
but I think we should check again <SEM_END>
```

---

# 14. 한국어와 영어의 공통 Token 정책

언어별 token을 만들지 않는다.

사용하지 않음:

```text
<SEM_END_KO>
<SEM_END_EN>
```

사용:

```text
<SEM_END>
```

Semantic commit 자체를 language-independent event로 정의한다.

```text
오늘 오전에 회의를 했어요 <SEM_END>

I had a meeting this morning <SEM_END>
```

두 event는 동일한 의미를 가진다.

---

# 15. Code-switching

한국어/영어 혼합 발화도 별도 고려한다.

예:

```text
이번 meeting은 제가 보기에는
다음 주로 postpone 하는 게 좋을 것 같아요 <SEM_END>
```

언어 category:

```text
KO
EN
MIXED
```

MIXED sample은 특정 언어 전용 heuristic보다 multilingual teacher agreement를 우선한다.

---

# 16. Hard-negative Set

일반 annotation만으로는 premature commit 사례가 충분하지 않을 수 있으므로 의도적으로 hard negative를 구성한다.

## Hesitation

```text
오늘 제가...
→ WAIT
```

## 연결 구조 + long pause

```text
병원에 갔다가...
→ WAIT
```

## Repair

```text
이번 주 금요일...
아니 목요일에
→ WAIT
```

## Repetition

```text
그 그 사람이
→ WAIT
```

## Filler

```text
저는 그...
어...
→ WAIT
```

## Long silence + incomplete semantics

```text
그래서 제가 생각하기에는
[1.5 sec silence]
→ WAIT
```

## Superficially complete + later correction

```text
내일 갈게요
...
아 아니 오늘 갈게요
```

이 category는 특히 높은 비율로 포함한다.

---

# 17. Transcript 품질이 낮은 DB 처리

Original GT가 항상 정답이라고 가정하지 않는다.

가능하면 다음 구조를 사용한다.

```text
                    Audio
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
 Original GT      ASR Teacher A   ASR Teacher B
       │              │              │
       └──────────────┼──────────────┘
                      ▼
             Transcript Consensus
                      │
             quality / disagreement
                      │
                      ▼
             Semantic Relabeling
```

LLM이 audio를 직접 보지 않는다면 새로운 transcript를 자유 생성하게 하지 않는다.

다음과 같이 candidate selection 문제로 제한한다.

```text
A / B / C / UNCERTAIN
```

---

# 18. H200 140GB 기준 LLM 구성

전제:

- GPU: NVIDIA H200 140GB
- 기본적으로 단일 GPU inference를 고려
- annotation throughput이 중요
- 라이선스 제한은 모델 선정 기준에서 제외
- 단계별 batch job으로 모델을 교체하여 사용

## Main Teacher

### Qwen3.8-27B-FP8

주요 역할:

- Stage A Full-context annotation
- 영어 Stage B
- 한국어/영어 Stage C
- multilingual/MIXED sample 처리

선정 이유:

- 27B급으로 H200에서 매우 여유 있는 inference
- FP8 사용 시 weight memory가 작아 large batch / KV cache 확보 가능
- KO/EN 공통 teacher로 활용 가능
- 100B+ 모델을 간신히 올리는 것보다 annotation throughput 측면에서 유리

권장 context:

```text
8K ~ 16K
```

대부분의 utterance/session annotation에서는 매우 긴 context를 사용할 필요가 없다.

---

## Korean Critic

### EXAONE 4.5 33B

주요 역할:

- 한국어 Stage B Causal Commit Judge
- 한국어 pragmatic completion 검증
- `-는데요`, `-고요` 등 문법적으로 continuation처럼 보이지만 화용적으로 완결되는 표현 검증
- Qwen과의 disagreement 생성

출력은 최소화한다.

```json
{"decision":"SAFE"}
```

또는:

```json
{"decision":"WAIT"}
```

---

## Independent Judge

### gpt-oss-120b

전체 데이터에 사용하지 않는다.

다음 sample에만 선택적으로 사용한다.

```text
Qwen vs EXAONE disagreement
low confidence
repair ambiguity
pragmatic fragment
code-switch ambiguity
```

즉 tie-breaker / independent judge 역할이다.

---

# 19. 추천 모델 실행 구조

동시에 여러 모델을 H200에 올리기보다 단계별 batch job으로 처리한다.

```text
전체 Dataset
    │
    ▼
[Job 1]
Qwen3.8-27B-FP8
Stage A
    │
    ▼
candidate 저장
    │
    ▼
[Job 2]
EXAONE 4.5 33B
KO Stage B
    │
    ▼
판단 저장
    │
    ▼
[Job 3]
Qwen3.8-27B-FP8
EN Stage B + KO/EN Stage C
    │
    ▼
agreement / disagreement
    │
    ▼
[Job 4]
gpt-oss-120b
disagreement only
    │
    ▼
Final Annotation
```

---

# 20. Qwen3.8-27B 다운로드

BF16/기본 모델:

```bash
pip install -U huggingface_hub

hf download Qwen/Qwen3.8-27B \
    --local-dir /models/Qwen3.8-27B
```

H200 annotation inference 목적이라면 FP8 버전을 우선 고려한다.

```bash
hf download Qwen/Qwen3.8-27B-FP8 \
    --local-dir /models/Qwen3.8-27B-FP8
```

필요하면 Hugging Face authentication:

```bash
hf auth login
```

---

# 21. 최종 Streaming Training Sequence

Offline annotation:

```text
어 저는 내일 아니 오늘 오후에 병원에 갔다가요
음 회사로 바로 갈 것 같아요 <SEM_END>
그리고 저녁에는 친구를 만나려고요 <SEM_END> <TURN_END>
```

Streaming interleave는 개념적으로:

```text
audio_0
<NEXT_AUDIO>
audio_1
<NEXT_AUDIO>
...
어
...
회사로
바로
갈
것
같아요
<SEM_END>
<NEXT_AUDIO>
...
그리고
...
만나려고요
<SEM_END>
<TURN_END>
```

ASR 모델 내부에는 semantic-specific auxiliary classifier를 추가하지 않는다.

---

# 22. 데이터 Sampling

한국어와 영어를 단순히 시간 기준으로 concatenate하지 않는다.

가능하면 semantic event 기준으로 sampling 균형을 맞춘다.

초기 목표 예:

\[
N_{\text{SEM,KO}} : N_{\text{SEM,EN}}
\approx 1:1
\]

서비스 비율에 따라 이후 조정한다.

또한 hard-negative category도 언어별로 균형을 맞춘다.

한국어:

- 연결어미
- hesitation
- filler
- self-repair
- repetition
- pragmatic `-는데요/-고요`

영어:

- subordinate clause
- hesitation
- filler
- self-repair
- pragmatic fragment
- discourse marker

---

# 23. Confidence Filtering

단일 LLM의 self-reported confidence는 신뢰하지 않는다.

대신 다음 신호를 조합한다.

- Stage A teacher agreement
- Stage B SAFE/WAIT agreement
- Stage C future stability
- language-specific critic agreement
- disfluency overlap 여부
- transcript quality
- alignment quality

최종 sample을 3등급으로 관리한다.

## A-grade

```text
Stage A agreement
Stage B SAFE
Stage C STABLE
no future repair
high transcript confidence
```

초기 학습에 사용한다.

## B-grade

```text
model disagreement
SAFE / UNCERTAIN
minor transcript uncertainty
```

초기에는 제외하고 이후 curriculum 또는 추가 검증에 사용한다.

## C-grade

```text
future repair
severe disagreement
poor alignment
poor transcript quality
```

학습에서 제외한다.

---

# 24. Evaluation Metrics

WER만으로 평가하지 않는다.

## 24.1 Semantic Boundary Precision

\[
Precision=
\frac{Correct\ predicted\ SEM\_END}
{All\ predicted\ SEM\_END}
\]

가장 중요한 metric 중 하나다.

---

## 24.2 Semantic Boundary Recall

실제 semantic boundary를 얼마나 놓치는지 측정한다.

---

## 24.3 Premature Commit Rate

\[
PCR =
\frac{Commits\ later\ invalidated\ by\ future}
{All\ commits}
\]

본 프로젝트에서 매우 중요한 핵심 KPI이다.

---

## 24.4 Commit Latency

실제 semantic completion 시점과 `<SEM_END>` 생성 시점의 차이:

\[
Latency =
t_{\text{predicted SEM\_END}}
-
t_{\text{reference completion}}
\]

---

## 24.5 Language-wise Metrics

반드시 분리해서 본다.

```text
KO Precision / Recall / PCR / Latency
EN Precision / Recall / PCR / Latency
MIXED Precision / Recall / PCR / Latency
```

---

## 24.6 Disfluency-category Metrics

다음 category별 성능도 따로 측정한다.

```text
filler
repetition
self-repair
long hesitation
subordinate/connected clause
pragmatic fragment
code-switching
```

---

# 25. Human Gold Evaluation Set

대규모 자동 relabeling에 들어가기 전에 최소 500~1,000개의 high-quality human gold sample을 만든다.

포함 비율을 의도적으로 조절한다.

예:

```text
normal completion          20%
filler                     10%
long hesitation            10%
repetition                 10%
self-repair                15%
connected/subordinate      15%
pragmatic fragment         10%
code-switch / 기타         10%
```

이 세트를 이용하여 다음을 비교한다.

```text
Qwen3.8-27B
EXAONE 4.5 33B
gpt-oss-120b
```

평가 결과를 바탕으로 각 stage의 모델을 최종 확정한다.

---

# 26. 모델 선정의 핵심 원칙

가장 큰 LLM 하나로 모든 데이터를 처리하지 않는다.

이번 task의 특성은:

```text
Semantic Segmentation
+
Discourse / Pragmatic Classification
+
NLI-like Future Stability
```

이므로,

```text
강한 27B급 multilingual main teacher
+
언어 특화 critic
+
다른 family의 independent judge
```

구조가 효율적이다.

H200 140GB에서는 특히:

> 100B+ 모델 하나를 메모리 한계까지 사용하기보다  
> Qwen3.8-27B급 모델을 큰 batch로 처리하고 모델 간 agreement를 활용하는 것을 우선한다.

---

# 27. 최종 권장 구성

```text
                         KO / EN / MIXED
                               │
                               ▼
                       Transcript QC
                               │
                               ▼
                     Word/Eojeol Indexing
                               │
                               ▼
                    Qwen3.8-27B-FP8
                 Full-context Stage A
                               │
                               ▼
                     SEM_END Candidates
                               │
                  ┌────────────┴────────────┐
                  │                         │
                 KO                        EN
                  │                         │
                  ▼                         ▼
        EXAONE 4.5 33B              Qwen3.8-27B
        Causal KO Critic            Causal EN Judge
                  │                         │
                  └────────────┬────────────┘
                               │
                               ▼
                    Qwen3.8-27B-FP8
                   Future Stability
                               │
                     STABLE / REVISION
                               │
                     disagreement only
                               │
                               ▼
                       gpt-oss-120b
                       Tie-break Judge
                               │
                               ▼
                          <SEM_END>
                               │
                               ▼
                 Streaming ASR Training
```

---

# 28. 다음 구현 순서

1. `<SEM_END>` 및 `<TURN_END>` token 규약 확정
2. KO/EN human gold set 500~1,000개 구축
3. Qwen3.8-27B-FP8 Stage A prompt 구현
4. EXAONE 기반 Korean causal judge 구현
5. Qwen 기반 English causal judge 구현
6. Future Stability prompt 구현
7. 모델 agreement 및 A/B/C grade filtering 구현
8. word timestamp와 semantic boundary alignment
9. `<SEM_END>`를 streaming interleave sequence에 삽입
10. 초기 학습
11. `SEM_END Precision`, `PCR`, `Commit Latency` 평가
12. hard-negative mining
13. 재학습 및 threshold/data policy 조정

---

# 29. 핵심 연구 가설

본 연구의 핵심 가설은 다음과 같다.

> 구어체 streaming ASR에서 문장 경계를 punctuation restoration 문제로 정의하는 것보다,  
> **현재 transcript prefix를 의미적으로 irreversible하게 commit할 수 있는지를 autoregressive event token으로 학습하는 것이**  
> real-time spoken interaction에 더 적합하다.

최종적으로 ASR 모델은 단순 transcription 모델이 아니라:

```text
Streaming Audio
      │
      ▼
Streaming ASR
      │
      ├── lexical tokens
      ├── <NEXT_AUDIO>
      ├── <SEM_END>
      └── <TURN_END>
      │
      ▼
Stable Semantic Prefix
      │
      ▼
Downstream LLM Prefill / Turn Taking / Full-Duplex Agent
```

형태의 **semantic-aware streaming front-end** 역할을 수행하게 된다.
