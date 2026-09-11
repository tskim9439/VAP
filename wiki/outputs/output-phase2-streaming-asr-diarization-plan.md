---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 v1.2 — 무음·단독·겹침 블록과 화자별 start/end_of_turn 공동 생성, 라벨·상태·학습·평가 계약
contributors:
  - tskim
sources:
  - '[[output-phase1-report]]'
  - '[[output-stage2-e2-final-eval]]'
  - '[[decision-mono-input]]'
  - '[[output-vapasr-model-and-sequence]]'
  - '[[source-hf-trainer-migration]]'
  - '[[source-conversation-corpora]]'
  - '[[output-vap-target-pipeline]]'
  - '[[turn-taking-objectives]]'
  - '[[turn-taking-evaluation-protocol]]'
  - '[[source-soulx-duplug]]'
  - '[[source-muse-voice-transcribe]]'
  - '[[output-phase2-final-plan]]'
  - '[[output-phase2-plan-critique]]'
  - '[[output-phase2-critique-response]]'
  - '[[output-phase2-block-and-turn-label-spec]]'
---

# Phase 2 개발 계획: Streaming ASR & Speaker Diarization / Turn-Taking

작성 기준: 2026-09-11. 사용자가 제시한 최대 2화자 목표를 구체화한 **실행 제안서**다. 아래 새 구조·수치 관문·예산은 제안이며, 구현 완료나 성능 보장이 아니다.

개정 v1.1(2026-09-11): [[output-phase2-plan-critique]]의 P1–P9를 검토하여 그룹 직렬화, 의미 학습 트랙, 조기 speaker memory 실험, ASR 강화 트랙, 중간 산출물 기준을 반영했다. 채택 범위·반론·계산 정정은 [[output-phase2-critique-response]], 달력·보유 데이터는 [[output-phase2-plan]]을 참조한다.

**개정 v1.2(사용자 추가 요구): 화자별 `<start_of_turn>`·`<end_of_turn>`을 전사와 함께 생성하는 필수 목표로 추가했다.** 무음·한 화자·겹침·맞장구·지연 전사·EOF의 블록 예시, 상태 머신, 라벨 생성과 loss 명세는 [[output-phase2-block-and-turn-label-spec]]이 정본이다. v1.1의 ‘턴 출력은 head 중심, 상태 토큰은 추후’ 범위를 이 부분에서 대체한다.

## 1. 목표와 범위

하나의 마이크에서 들어오는 16 kHz mono 스트림을 받아, **최대 두 화자의 말을 실시간으로 전사하고 화자별로 귀속하며, 발화 내용과 운율·대화 이력으로 앞으로의 턴 교대를 예측**한다. 두 사람이 동시에 말해도 각자의 전사가 남아야 한다.

| 요구 | 모델이 제공할 결과 | 검증 질문 |
|---|---|---|
| 실시간 ASR 강화 | 누적 lexical 전사, 방출 시각, 안정적인 처리 지연 | E2 정확도를 유지하면서 실제 입력 속도를 따라가는가? |
| Speaker diarization | 대화 내 고정 A/B ID, 화자별 활동 구간 | 누가 언제 말했으며, 긴 침묵 뒤에도 같은 ID인가? |
| Overlap 인식 | A와 B의 동시 활동 확률 | 동시에 말함을 감지하는가? |
| Overlap 전사 | A/B 각각의 전사와 시간 정보 | 작은 목소리를 누락하거나 큰 목소리를 복제하지 않는가? |
| Turn-taking | 화자별 미래 활동, 턴 양도·유지 가능성, 끼어들기·맞장구 이벤트 | 침묵 길이뿐 아니라 문맥을 이용해 다음 행동을 예상하는가? |

**Diarization과 turn-taking은 서로 다른 목표**다. 현재 화자 변경 검출만으로 미래 턴 예측을 달성했다고 판정하지 않는다. `둘 다 발화 중`과 `A가 주도하고 B가 맞장구 중`도 구분한다.

범위는 기존 EN/KO를 유지한 0–2화자다. A/B는 세션 내 익명 ID이며 실명 인식이나 세션 간 화자 인증은 포함하지 않는다. 분리된 음성 파형 생성, 번역, 응답 생성/TTS, 3명 이상 처리는 이번 필수 산출물이 아니다. 파형 분리를 하지 않고 두 화자의 전사를 직접 생성하는 것을 주 경로로 한다. 영어 대소문자·구두점 복원은 이후 과제로 유지한다.

## 2. 출발점과 기존 계획과의 관계

E2는 Nemotron streaming encoder → adapter → Qwen3-ASR thinker 구조이며, 약 5.6k h 학습과 encoder 해동을 거쳤다. HF Trainer 및 M4 MLX 실시간 실행이 있다. [[output-phase1-report]], [[source-hf-trainer-migration]]

| select 결과 | E2 δ=2 | E2 δ=4 | 기록된 RNN-T 참고값 |
|---|---:|---:|---:|
| dev-clean WER | 7.6% | 4.9% | 4.4% |
| dev-other WER | 13.9% | 10.3% | 8.2% |
| Kspon dev CER | 16.7% | 12.8% | 20.2% |
| AIHub 71631 dev CER | 29.2% | 23.4% | 미측정 |

출처: [[output-stage2-e2-final-eval]]. 표는 select 표본이며 전체 test 결과가 아니다. 해당 보고서의 RNN-T는 ‘오프라인 기준’으로 표기되어 있어 기존 `PLAN.md`의 `[56,0]` 실시간 동등 조건과 일치하는지 재확인해야 한다. E2는 encoder도 바뀌었으므로 ‘같은 encoder 가중치’ 비교로 표현하지 않는다.

기존 `PLAN.md` Stage 2 관문(상대 오류 열화 ≤10%, evidence 위반 0)은 아직 완전 통과로 볼 수 없다. 본 문서의 Phase 2는 기존 Stage 3의 다화자·턴 예측과 Stage 4의 실시간 안정화 일부를 묶은 사용자 지정 개발 범위다. **미달 항목을 이월한 탐색 개발**로 계획하고, Phase 1 통과를 소급 선언하거나 기존 관문을 자동 폐기하지 않는다. 탐색 진입과 최종 성능 채택을 별도 관문으로 둔다.

E2 체크포인트·tokenizer·TN 지문·평가 split을 고정한다. Q1/Q2의 전사 품질은 δ=4를 최초 주 평가점, δ=2를 저지연 평가점으로 삼는다. Q3의 turn 실험은 δ=2를 첫 실행 조건으로 하고 δ=4를 같은 평가 pack에서 필수 비교한다. δ=4 전사와 δ=2 turn 결과는 서로 다른 실행 조건임을 명시한다. E2의 ASR 방출 타이밍 학습은 turn-taking의 의미 이해를 입증하지 않는다.

## 3. 권장 모델 구조

```text
16 kHz mono 혼합 음성
        ↓
E2 Nemotron streaming encoder [56,0] → adapter
        ↓
Qwen3-ASR thinker: [AUDIO_k] → 화자 태그·start/end_of_turn·전사 → <NEXT_AUDIO>
        ├─ 생성 경로: <SPK_A/B> + turn markers + lexical text → 화자별 turn 전사 버퍼
        └─ h_audio[k]: 현재 오디오 위치의 causal hidden state
              ├─ 현재 활동: [P(A active), P(B active)]
              ├─ 미래 활동: VAP 256-class
              └─ 후속 단계: 화자별 onset hazard / semantic-event heads
```

분리 채널은 라벨·정렬·훈련용 혼합 생성에만 사용한다. 추론에는 mono만 제공하고, 정답 VAD·정답 화자·깨끗한 채널·완성 전사·전체 파일 통계를 넣지 않는다. 이는 [[decision-mono-input]]을 유지한다. 화자별 활동 헤드는 별도 외부 diarization 시스템이 아닌 공유 모델의 보조 출력이다.

### 3.1 왜 이 구성을 먼저 시험하는가

기존 thinker와 80 ms 시간축, `<NEXT_AUDIO>` 규약을 재사용한다. 겹침 전사는 한 decoder에서 청크 시간순으로, 같은 청크 안에서는 화자별 그룹으로 직렬화하는 것을 첫 후보로 한다(§4.2). 다화자 토큰을 단일 출력 경로에 직렬화하는 선행 근거는 [t-SOT](https://arxiv.org/abs/2202.00842)다. 다만 원 논문의 시간순 토큰 직렬화와 이번 그룹화는 다른 규약이다. virtual channel과 세션 내 고정 화자 ID도 같지 않으므로 A/B 추적 능력은 별도로 검증한다.

오디오마다 예측하는 활동 헤드가 있어야 텍스트가 없는 무음·호흡·겹침에서도 상태를 낼 수 있다. 기존 `<SPK_A/B>`는 현재 HF 디코드에서 차단되어 있고 활동·turn 헤드도 없다. 예약 토큰이 있다는 이유로 기능 구현이 완료된 것은 아니다.

### 3.2 Audio-clock 헤드와 의미 정보

`h_audio[k]`는 `[AUDIO_k]`를 처리한 직후, **같은 청크의 전사를 생성하기 전** 상태로 고정한다. 현재까지의 오디오와 이전 청크까지 방출한 텍스트를 볼 수 있다. 현재 청크의 gold 토큰이나 미래 발화를 attention으로 볼 수 없게 한다. 활동·turn 출력은 텍스트 생성 여부와 무관하게 80 ms마다 계산한다.

이 선택은 turn 출력이 ASR flush를 기다리는 문제를 줄이지만, 최근 확정 전사는 늦게 들어온다. δ는 목표 방출 지연이며 실제 전사 지연은 청크 양자화·오류·backlog도 포함한다. 현재 오디오 정보까지 δ만큼 낡은 것은 아니다. Q3에서는 δ=2/4별 자유실행 history로 head를 학습·평가하고, 오디오 상태에서 아직 방출하지 않은 단어/완결성의 prefix probe도 보조 진단한다. 같은 청크 전사를 본 뒤의 헤드는 후속 ablation이며 추가 생성 시간을 가용 시각에 포함한다.

초기 헤드는 작은 MLP로 두고 E2 표현의 유효성을 확인한다. 활동은 2개 sigmoid로 두어 `00/10/01/11`을 모두 허용한다. 중첩 확률은 초기에는 `VAP`의 현재 상태 대체가 아니라 현재 활동 출력으로 평가하며, 두 sigmoid의 곱은 독립 가정이므로 필요 시 4상태 공동 분포 헤드와 비교한다.

Q1부터 `speaker_state` 인터페이스와 장기 침묵 뒤 복귀 평가 pack을 준비한다. no-memory 기준선을 먼저 확보하고, 같은 checkpoint에서 **2-slot memory on/off**를 Q1 말–Q2 초에 비교하여 Q4까지 결정을 미루지 않는다. [t-vector 연구](https://arxiv.org/abs/2203.16685)는 token-level 화자 표현의 참고다.

memory 첫 후보는 `q_k = speaker_projection(encoder_k)`의 화자별 EMA/소형 exemplar bank다. 이전 상태로 현재 활동을 예측한 뒤, 단독 활동 confidence·비중첩·최소 누적 관측 조건을 만족할 때만 갱신하여 다음 청크부터 사용한다. 무음·겹침·불확실 구간에서는 갱신을 막고, 초기 동시 발화에는 uninitialized 슬롯과 추후 bootstrap을 허용한다. 훈련·검증에서도 예측 activity 기반 갱신을 사용하며 gold 갱신은 oracle 진단으로만 둔다.

원 encoder feature 평균은 화자뿐 아니라 음소·채널도 담으므로 speaker embedding으로 보장되지 않는다. 같은 화자의 다른 내용에서 안정적인지 probe하고 필요하면 화자 대조 손실을 추가한다. A/B 슬롯은 slot ID를 보존한 concat/projection 또는 slot-conditioned query로 읽고, 단순 `m_A+m_B` 합으로 정체성을 잃지 않게 한다. E2 경로는 zero-init gated residual로 시작한다. 이 방식은 구현 제안이며 추가 연산·지연·memory drift를 실측한다. [Streaming Sortformer](https://arxiv.org/abs/2507.18446)는 선택된 프레임 cache의 선행 근거이지 단순 EMA의 성능 보증이 아니다.

## 4. 화자·전사·시간 계약

### 4.1 A/B 정체성

- 최초로 식별 가능한 화자를 A, 다음 새 화자를 B로 배정하고 세션 중 유지한다. A/B를 남녀·채널 번호·큰 목소리와 연결하지 않는다.
- 시작부터 겹쳐 식별이 모호하면 임시 슬롯으로 시작하고 confidence를 기록한다. 주 지표는 최초 출력 기준이다. 나중에 ID를 고쳤다면 수정 횟수·확정 지연도 보고한다.
- 학습의 channel→speaker 매핑, 전사 태그, 활동·VAP·hazard 라벨은 **같은 permutation**을 사용한다. 무작위 channel 순서와 gain을 바꾸어 지름길 학습을 검사한다.
- 독립 crop은 관측 prefix 내 최초 화자 기준으로 일관되게 재매핑한다. carry 학습·장문 평가는 세션 ID를 유지한다. 미래 단독 구간이나 전체 파일 clustering 결과로 온라인 A/B를 정하지 않는다.
- PIT를 쓰는 실험도 permutation은 crop/세션 단위로 고정하며 청크마다 최적 permutation을 바꾸지 않는다. 전사 손실과 활동 손실을 따로 permutation하면 ID 의미가 충돌한다.

### 4.2 겹침 전사의 직렬화

각 화자의 lexical 전사를 독립적으로 tokenize·정렬하고, E2의 `k=floor(t_end/0.08)+δ`로 청크에 배정한다. **G1 기본 후보**는 각 청크 안에서 A의 토큰을 모두 낸 뒤 B의 토큰을 내며, 화자 내부 순서를 보존한다. 빈 화자 블록은 생략한다. 원 토큰 시각은 라벨 metadata에 보존한다.

```text
[AUDIO_k] <SPK_A> 그때 내가 <SPK_B> 응 <NEXT_AUDIO>
[AUDIO_k+1] <SPK_A> 갔거든 <SPK_B> 맞아 <NEXT_AUDIO>
```

위 예시는 두 화자의 turn이 이미 OPEN일 때 전사 payload만 보인 것이다. 오디오에는 실제 겹침이 남고 문자열만 직렬화한다. 청크 경계에서도 현재 소유자를 유지하고 최초 lexical 출력과 화자 변경 때 selector를 낸다. v1.2는 모든 start/end marker 앞에 `<SPK_s>`를 명시한다. `<SPK_A><SPK_B>` 같은 행동 없는 selector 반복은 제한하되 **`<SPK_A><end_of_turn>`처럼 텍스트 없는 이벤트 블록은 허용**한다. 태그는 현재 음향 활동이 아니라 뒤 전사·이벤트의 귀속이다.

BPE byte 조각 사이에 다른 화자의 토큰을 끼워 넣으면 화면 텍스트가 깨질 수 있다. 같은 정렬 단위의 토큰 묶음은 원자적으로 유지하고 화자별 decoder buffer로 Unicode/공백을 복원한다. 단어 전체 완료까지 기다리는 변형은 추가 지연을 따로 잰다. TN은 현재 lexical 규약과 지문을 유지하고 화자 태그를 문자열 정규화에 섞지 않는다.

G1의 전사-only payload는 selector가 청크당 최대 두 개지만 v1.2의 event selector는 추가된다. **종료 시각순과 동등한 모델링 문제는 아니다**. 80ms 내부 종료 순서와 자기회귀 조건이 바뀌며 B는 A 생성 뒤에 나온다. Q1 말/Q2 초 동일 데이터·예산의 G0/G1을 비교하고 event 순서·A/B별 지연·삭제·cap hit도 본다. B의 체계적 불이익이 있으면 홀짝 선행 슬롯 교대 G2를 추가한다. 같은 화자의 start→전사→end 의존 순서는 모든 variant에서 지킨다.

두 화자의 전사와 태그 때문에 청크당 생성량이 증가한다. 현재 안전 상한 8 토큰을 그대로 적용하거나 단순히 두 배로 올리지 않고 실제 밀도 p99·강제 NEXT·삭제율·처리 지연을 함께 측정한다. 구조 토큰과 lexical 토큰 예산을 별도로 기록한다. 학습의 무제한 방출과 추론 상한 차이도 QC한다.

### 4.3 출력 API와 세 개의 시각

전사 이벤트는 `session_id, speaker_id, token_ids/text, chunk_index, audio_seen_until, emitted_at, confidence`를 가진다. 활동·turn 이벤트는 `audio_seen_until, emitted_at, activity[2], vap_probs, onset_hazard, event_probs`를 제공한다. 현재 없는 필드는 구현 단계에 맞춰 추가한다.

`audio_seen_until`은 실제 읽은 마지막 오디오 시각, `emitted_at`은 실제 결과 가용 시각이다. `estimated_speech_start/end`를 제공할 경우 별도 시간 추정 head 또는 causal aligner가 필요하며 추정값임을 표시한다. `방출 시각−δ`를 정확한 발화 경계로 간주하지 않는다. 초판은 활동 구간과 전사 방출 시각을 제공하고, tcpWER에 사용할 단어/구간 타임스탬프 생성 규약을 평가기에서 고정한다.

### 4.4 무음·한 화자·turn 종료의 필수 계약

| 상황 | 정답 예시(OPEN 상태·이전 소유자는 문맥에서 유지) |
|---|---|
| 무음, pending 출력 없음 | `[AUDIO_k]<NEXT_AUDIO>` |
| A speech 중이지만 이번 청크 전사 없음 | `[AUDIO_k]<NEXT_AUDIO>`; activity는 `[1,0]` |
| A 시작 | `[AUDIO_k]<SPK_A><start_of_turn>...<NEXT_AUDIO>` |
| A만 전사 | `[AUDIO_k]<SPK_A>텍스트<NEXT_AUDIO>`; 빈 B 블록 없음 |
| A/B 겹침 | `[AUDIO_k]<SPK_A>A 전사<SPK_B>B 전사<NEXT_AUDIO>` |
| A 종료, 새 텍스트 없음 | `[AUDIO_k]<SPK_A><end_of_turn><NEXT_AUDIO>` |
| A 계속, B 맞장구 종료 | `[AUDIO_k]<SPK_A>A 전사<SPK_B><end_of_turn><NEXT_AUDIO>` |

모든 실제 무음은 encoder를 통과한다. NEXT는 ‘이번 라운드 출력 끝’이며 EOT가 아니다. pause 중 A turn은 OPEN일 수 있고, B start/EOT는 A를 자동으로 닫지 않는다. 현재 silence에서도 이전 lexical/EOT를 방출할 수 있다. EOF·crop 끝은 EOT 정답이 아니며 padding을 추가 무음 증거로 사용하지 않는다. 18개 경우와 생성 상태·타깃 시각은 [[output-phase2-block-and-turn-label-spec]] §4–§9를 따른다.

### 4.4 Turn 토큰과 시퀀스 사례 (2026-09-11 추가)

§4.2 의 종료 시각 순 교차 직렬화를 유지한 채, Muse 의 `|speech_onset|`/`|speech_endpoint|` 처럼([[source-muse-voice-transcribe]]) **turn 토큰을 시퀀스 안에** 둔다. 화자 태그가 뒤따르는 토큰의 귀속이므로 turn 토큰도 "직전 태그의 화자" 에 귀속된다. 병렬 헤드(§3.2, §6)는 그보다 이른 연속 확률을 내는 별개 출력이며 둘 다 §8 규약으로 채점한다.

| 토큰 | 뜻 | 시각(청크) | 라벨 출처 | 손실 가중(초기값) |
|---|---|---|---|---|
| `<ONSET>` | 그 화자가 말을 시작했다 | `⌊t_on/80ms⌋ + 2` (160 ms 고정) | 채널 VAD 세그먼트 시작(직전 침묵 ≥ 0.2 s) | 1.0 |
| `<EOT>` | turn 이 끝났다(floor 를 넘김·잃음·침묵으로 종료) | `⌊t_off/80ms⌋ + δ` | `derive_events` SHIFT, terminal overlap, 3 s 무발화, 방해당한 종료 | 2.0 |
| `<HOLD>` | 멈췄지만 turn 을 쥐고 있다 | `⌊t_off/80ms⌋ + δ` | HOLD(같은 화자가 3 s 안에 먼저 재개) | 2.0 |
| `<BC>` | 이 발화는 맞장구였다 | `⌊t_off/80ms⌋ + δ` | BACKCHANNEL(상대 발화 중 시작, ≤ 1 s, 상대 계속) | 2.0 |

- 세그먼트 하나당 `<ONSET>` 1 개와 종료 토큰(`<EOT>`/`<HOLD>`/`<BC>`) 정확히 1 개. INTERRUPT 는 토큰이 아니라 "상대 활동 중 `<ONSET>` + 상대의 `<EOT>`" 로 이벤트 층에서 유도한다.
- `<EOT>`/`<HOLD>`/`<BC>` 의 라벨은 **미래(최대 3 s)** 로 정해진다. 방출 시각(종료 + 160–320 ms)에 모델이 문맥·운율로 판단해야 하며, 그것이 의도다. 입력에 미래 오디오는 들어가지 않는다.
- 같은 시각의 정렬: `<ONSET>` → 텍스트 → 종료 토큰, 화자 간은 A → B. 종료 토큰은 그 화자의 마지막 텍스트 토큰 뒤에 온다(`t_end ≤ t_off`).
- 태그 규칙은 §4.2 그대로: 직전 방출 항목과 화자가 다를 때만 `<SPK_X>`. 내용 없는 태그 반복 금지.

사례(δ=2, 한 줄이 한 청크, `#` 은 설명):

**무음** — 블록 없음. 손실은 `<NEXT_AUDIO>` 에만(EN 0.3 / KO 0.15). activity 헤드는 `00`.

```
[AUDIO_0] <NEXT_AUDIO>
[AUDIO_1] <NEXT_AUDIO>
```

**단일 화자만** — 첫 항목에만 태그. 멈춤은 `<HOLD>`, 재개는 `<ONSET>`(태그 생략), turn 끝은 `<EOT>`. Phase 1 replay 스트림도 같은 문법(§5.3 의 손실 규칙 참조).

```
[AUDIO_5]  <SPK_A> <ONSET> <NEXT_AUDIO>
[AUDIO_8]  안녕 <NEXT_AUDIO>
[AUDIO_9]  하세요 <NEXT_AUDIO>
[AUDIO_12] <HOLD> <NEXT_AUDIO>
# 0.3 s 멈춤 뒤 같은 화자가 이어 말함
[AUDIO_16] <ONSET> <NEXT_AUDIO>
[AUDIO_40] 입니다 <EOT> <NEXT_AUDIO>
[AUDIO_41] <NEXT_AUDIO>
```

**화자 교대** — gap 뒤 B 시작.

```
[AUDIO_40] 입니다 <EOT> <NEXT_AUDIO>
[AUDIO_47] <SPK_B> <ONSET> <NEXT_AUDIO>
[AUDIO_50] 네 <NEXT_AUDIO>
```

**맞장구 overlap** — 교차 직렬화라 한 청크 안에서 화자가 여러 번 바뀔 수 있다.

```
[AUDIO_60] 그래서 <SPK_B> <ONSET> <NEXT_AUDIO>
[AUDIO_62] <SPK_A> 제가 <SPK_B> 응 <BC> <SPK_A> 말씀 <NEXT_AUDIO>
# 종료 시각 순: 제가(A) → 응(B, 종료 → <BC>) → 말씀(A)
[AUDIO_63] 드린 <NEXT_AUDIO>
```

**끼어들기** — A 는 방해당해 floor 를 잃으므로 `<EOT>`.

```
[AUDIO_70] 그런데 <SPK_B> <ONSET> <NEXT_AUDIO>
[AUDIO_72] <SPK_A> 제 <SPK_B> 잠깐만요 <NEXT_AUDIO>
[AUDIO_74] <SPK_A> <EOT> <SPK_B> 그건 <NEXT_AUDIO>
[AUDIO_75] 아니에요 <NEXT_AUDIO>
```

**동시 시작** — A → B 순. A/B 는 관측 prefix 안 첫 식별 화자(§4.1), 동률이면 50 Hz VAD 가 이른 쪽, 그래도 동률이면 라벨 permutation 무작위(일관).

```
[AUDIO_80] <SPK_A> <ONSET> <SPK_B> <ONSET> <NEXT_AUDIO>
```

**스트림 끝(flush)** — δ 때문에 넘긴 텍스트·종료 토큰을 `<EMPTY_AUDIO>` 라운드로 낸다.

```
<EMPTY_AUDIO> 입니다 <EOT> <NEXT_AUDIO>
<EMPTY_AUDIO> <NEXT_AUDIO>
```

**디코드 제약(logit 마스크)**: 상태는 `cur`(직전 화자), `active[A|B]`(자신이 낸 `<ONSET>`/종료 토큰으로 갱신). 비활동 화자에게는 종료 토큰 금지, 활동 화자에게는 `<ONSET>` 금지, 빈 태그 금지, 청크당 상한(§4.2 의 밀도 실측 후 확정) 도달 시 `<NEXT_AUDIO>` 강제·이월. 이벤트 시각은 청크 오디오를 모두 들은 시각 + 디코드 시간(§4.3).

## 5. 데이터 계획

### 5.1 자연 대화와 합성 겹침의 역할

| 데이터 | 첫 사용 | 주의 |
|---|---|---|
| otoSpeech EN | 자연 대화 전사·화자·미래 활동 | 기존 기록 약 104.9 h. 현재 위치·정렬·split·사용 조건 재검증 |
| AI Hub 71631 KO TS_01 실내 | 자연 대화 및 겹침 전사 | 기존 기록 약 196.6 h. Phase 1 발화 crop을 원 대화 시간축으로 복원 |
| AI Hub VS_02 | KO 실외 검증/보고 후보 | 이미 여러 실험에서 사용했으므로 신규 untouched test로 부르지 않음 |
| LibriSpeech/Kspon 등 서로 다른 화자 혼합 | overlap 비율·음량차를 제어한 전사·활동 학습 | 임의 결합은 자연 turn-taking 정답이 아님 |
| Phase 1 EN/KO 데이터 | 단일 화자 ASR replay | E2와 같은 TN·split 유지 |
| Switchboard/CANDOR 등 | 1차 관문 이후 확장 후보 | 원 대화 채널·화자·시간·확보 상태 확인 후 편입 |

규모는 과거 조사 기록이며 이번에 서버 가용량을 실측하지 않았다. 근거: [[source-conversation-corpora]], [[output-vap-target-pipeline]]. 코퍼스별 대화 경과시간, 화자별 speech-hours, 혼합 후 시간, 무음 비율을 따로 집계한다. 두 채널 시간을 더한 값을 대화 시간으로 쓰지 않는다. 라이선스는 실제 사용 목적에 맞는 최신 원문 확인을 데이터 관문에 포함한다.

후속 [[output-phase2-plan]]은 2026-09-11 mxc 보유량을 71631 248h(TS 196.6h + VS 51.7h), otoSpeech 104.9h로 기록했다. 이를 가용량 근거로 사용하되 VS·TurnBench dev를 train 시간에 넣지 않는다. 화자 분할·QC 전 자연 대화 train 후보 상한은 TS+oto 약 301.5h이며 실제 채택 시간은 더 작다.

자연 대화는 원래 gap·중첩·맞장구·발화 순서를 보존한다. 비중첩 단계에서는 긴 대화 중 적합한 창을 선택하며 턴 사이 무음을 삭제하거나 발화를 당겨 붙이지 않는다. 합성 겹침의 VAD는 쓸 수 있지만 `L_VAP/L_hazard/L_semantic-event`는 초기에는 마스킹한다. 단일 화자 replay도 turn 손실을 마스킹한다.

합성 스트림은 20–40초부터 시작하고 다음 조건을 분리한다: 무겹침, 짧은 맞물림, 10–30% overlap, 30–50% stress, 같은 시각 시작, 한쪽 음량이 3/6/12 dB 낮음, 비슷한 음색. overlap 비율 분모는 ‘적어도 한 명이 말하는 시간’으로 고정한다. 모든 합성 원본은 train split에 속해야 한다.

합성 subset 내 **약 70% 자연형 패턴 / 30% stress 패턴**을 최초 sampling 후보로 명시한다. 자연형은 짧은 응답·맞장구 형태, 0.2–0.5초 terminal overlap 등을 train에서 실측한 길이·음량·위치 분포로 만든다. 70:30은 자연 분포의 실측값이 아니며 Q0 검수 후 고정한다. 기존 BC/INT 통계는 휴리스틱 산출물이므로 ‘진짜 BC 비율’로 단정하지 않는다. 임의 ‘응’을 얹었다고 의미상 backchannel 정답이 생기지 않으며 turn 손실 마스크를 유지한다. donor는 해당 세션 B와 같은 화자로 고정해 세 번째 화자가 섞이지 않게 한다. 원래 대화의 자연 overlap와 합성 유형별 지표를 분리 보고한다.

### 5.2 스키마와 로더

기존 `vapasr/data/streams.py::assemble_stream`은 `out[o:o+n] = x[:n]` 대입이라 겹치는 입력을 합산하지 않는다. **Phase 2 전용 대화 mixer**를 만들고 다음을 계약화한다.

- 입력 파형은 자연 mono 원음 우선, 없으면 동기화된 두 채널을 고정 규칙으로 합산한다. clipping·반대 위상·채널 누설·샘플레이트·sample offset을 검사한다. 추론 정규화는 미래 파일 통계에 의존하지 않는다.
- `conversation_id, source_speaker_id, local_speaker_slot, source_channel, src_offset_s, mix_offset_s, duration, gain, split, alignment_quality, task_masks`를 명시한다. 기존 스키마를 묵시적으로 바꾸지 않고 별도 버전과 변환기를 둔다.
- Forced alignment는 각 깨끗한 채널에서 수행한 뒤 혼합 시간축으로 옮긴다. 혼합 파형에 단일 화자 aligner를 그대로 적용하지 않는다. 불확실한 token 경계는 타이밍 손실/평가에서 분리 집계한다.
- 원 대화 길이의 VAD를 먼저 만들고 crop별로 참조한다. crop 뒤의 2.56초까지 라벨 산출에 쓸 수 있으나 그 미래 오디오는 모델 입력에 넣지 않는다. 녹음 끝·주석 누락은 관측 마스크로 처리한다.
- 화자별 채널 매핑을 파일마다 검증한다. 기존 AI Hub 기록에 channel swap이 있었고 발화 EndTime은 정밀 VAD가 아니다. 에너지 VAD도 검수된 정답으로 취급하지 않는다.
- E2까지의 학습 데이터와 화자·대화·오디오 중복을 감사한다. pretraining 노출, Phase 2 train, dev, untouched test를 구분한다. crop을 만든 뒤 무작위 split하지 않는다.

첫 QC는 EN/KO 각 50개 대화 창의 파형·채널·혼합·전사·정렬·활동 라벨을 실제 학습 Dataset 객체를 통해 점검한다. 마지막 1바이트 PCM, EOF, archive channel 선택, 원본 offset, 무음 화자, 전사 없는 speech, 화자 교체, 채널 순서 반전도 회귀 사례에 넣는다.

### 5.3 Turn 토큰·시퀀스 데이터 생성 파이프라인 (2026-09-11 추가)

§4.4 의 시퀀스는 손으로 라벨하지 않는다. 분리 채널 코퍼스에서 아래 순서로 **전부 자동 유도**하고, 사람은 검증만 한다.

```
분리 채널 (A.wav, B.wav) + 발화 라벨(텍스트, 대략 시각)
  ① 채널별 VAD@50Hz ──▶ 세그먼트(≥0.2 s 침묵으로 분리, <0.1 s 조각 제거)      → t_on, t_off (화자별)
  ② 채널별 강제 정렬 ──▶ 토큰 종료 시각 t_end (깨끗한 채널, 발화 창 = 라벨 ±0.5 s)  [기존 s2_prep · align-asr-tn-v1]
  ③ 두 채널 VAD ──▶ derive_events ──▶ 세그먼트 종료마다 EOT | HOLD | BC, 시작마다 ONSET
  ④ 슬롯 배정 ──▶ crop 안 첫 세그먼트 화자 = A (전사·이벤트·activity·VAP·hazard 라벨에 같은 permutation)
  ⑤ 혼합 ──▶ mono = g_A·A + g_B·B (+RIR)   ※ 채널이 동기화돼 있으므로 ①②③ 의 시각은 그대로
  ⑥ 직렬화 ──▶ 항목 (시각, 화자, 종류, payload) 를 §4.2 규칙으로 정렬 → 청크 배정 → 태그 삽입 → flush
  ⑦ QC ──▶ 왕복 복원·통계·청취
```

**① VAD**: 71631 은 채널별 에너지 VAD(라벨 EndTime 이 넉넉하므로 라벨 시각을 쓰지 않음; 실물 검증에서 onset 오차 중앙값 30 ms), otoSpeech·TurnBench 는 라벨 VAD(에너지 VAD 일치 0.84·0.69 라 라벨이 낫다). [[output-vap-target-pipeline]] 의 `vad.py` 그대로. 세그먼트 병합 gap 0.2 s 는 `<HOLD>` 의 최소 pause 와 같다.

**② 정렬**: Phase 1 의 71631 manifest 가 이미 `path#chN` + `src_offset_s` 로 채널별 정렬을 냈다(`experiments/s1_align.py` 출력 `tokens[{id,text,end_time}]`, 스트림 절대 시각). otoSpeech 는 SRT 발화를 같은 형식으로 넣는다. 정렬 실패·빈 레코드(≤1 %)는 그 발화의 텍스트만 빼고 이벤트는 유지한다(`alignment_quality` 필드).

**③ 이벤트 → 토큰** (`vapasr/data/targets.py::derive_events` 를 세그먼트 단위로 확장):

| 세그먼트 종료 상황 | 토큰 |
|---|---|
| 침묵 ≥0.2 s 뒤 3 s 안 첫 발화가 상대 (SHIFT) | `<EOT>` |
| 침묵 ≥0.2 s 뒤 3 s 안 첫 발화가 같은 화자 (HOLD) | `<HOLD>` |
| 3 s 안 아무도 말하지 않음 | `<EOT>` |
| 상대가 내 발화 중 시작했고 나는 0.5 s 안에 끝남 (terminal overlap = SHIFT) | `<EOT>` |
| 상대가 내 발화 중 시작했고 나는 더 말하다 끝났으며 재개하지 않음 (방해당함) | `<EOT>` — 현행 규칙은 "이벤트 없음", 토큰용으로 추가 |
| 내 세그먼트가 상대 발화 중 시작해 ≤1 s 로 끝나고 상대가 계속 (BACKCHANNEL) | `<BC>` |
| 세그먼트 시작 | `<ONSET>` (모든 세그먼트) |

검증 순서: (a) 합성 VAD 단위 테스트(사례 7 종), (b) TurnBench dev gold EOT/INT 와 대조 — 기존 기록은 휴리스틱 SHIFT 992 vs gold EOT 1,904 로 차이가 크므로 **gold 정의(같은 화자 재개도 EOT 인가)를 먼저 맞추고** 규칙을 조정, (c) EN/KO 각 300 세그먼트 종료를 사람이 EOT/HOLD/BC 로 판정해 일치율 보고. 일치율 <0.8 인 유형은 손실 가중 0 으로 두고 학습을 시작한다.

**④ 슬롯**: crop 마다 첫 세그먼트 화자를 A 로 재매핑하고 permutation 을 레코드에 기록. 채널 교환 증강은 permutation 을 뒤집는 것과 같으므로 라벨을 다시 만들 필요가 없다.

**⑥ 직렬화 알고리즘**(`dialogue_interleave.py`):
1. 항목 생성: 텍스트 토큰 `(t_end + δ·80ms, spk, TEXT, id)`, `(t_on + 160ms, spk, ONSET)`, `(t_off + δ·80ms, spk, END, type)`.
2. 정렬 키 `(chunk = ⌊t/80ms⌋, spk(A<B), 종류 순서 ONSET<TEXT<END, 원 순서)`. 같은 청크·같은 화자 안에서는 원 발화 순서 유지, 화자 간은 종료 시각 순(§4.2).
3. 청크마다 `[AUDIO_k]` 뒤에 항목을 순서대로 놓고, 직전 방출 항목과 화자가 다르면 `<SPK_X>` 삽입, 끝에 `<NEXT_AUDIO>`. 스트림 끝을 넘긴 항목은 `<EMPTY_AUDIO>` 라운드로.
4. crop 경계: 창 시작 전에 시작한 세그먼트는 `<ONSET>` 없이 텍스트부터(상태 기계가 "이미 말하는 중" 을 배운다), 창 시작 전 증거(`t_end`)를 가진 텍스트는 버린다(환각 방지), 창 끝 뒤 2.56 s 는 라벨에만 쓴다.
5. 왕복 테스트: 시퀀스 → (화자별 텍스트, 이벤트 목록) 이 입력과 같아야 한다.

**⑦ QC 통계**(코퍼스·언어별): 청크당 토큰 p50/p99, 태그 수/분, `<ONSET>`·`<EOT>`·`<HOLD>`·`<BC>` 시간당 개수를 [[output-vap-target-pipeline]] 의 이벤트 통계(TS_01 실내: SHIFT 121 k / HOLD 207 k / BC 52 k per 196.6 h)와 대조, 렌더한 타임라인 위에서 코퍼스당 20 crop 청취.

**데이터 종류별 손실 규칙**

| 데이터 | `<ONSET>` | `<EOT>`/`<HOLD>`/`<BC>` | 비고 |
|---|---|---|---|
| 자연 대화 mono 혼합(71631·oto) | 손실 | 손실(가중 2.0) | 주 학습 데이터 |
| NIKL 준자연 대화(같은 대화의 발화 오디오를 라벨 시각에 배치해 합산, 겹침 발화 제외) | 손실 | 손실 | **KO turn 토큰의 대규모 원천**(3,800 h, 발화 시각·화자 ID 있음). 겹침이 없다는 점을 명시 |
| Phase 1 단일 화자 replay(LibriSpeech·Kspon 스트림) | 손실 | **가중 0**(낭독 스트림의 pause 는 turn 의미가 없음) | 시퀀스에는 넣어 문법을 유지 |
| 합성 2 화자(무관한 화자) | 손실 | 가중 0(타이밍이 가짜) | 전사·activity 용 |
| Switchboard·CallHome(시간축 복원 시) | 손실 | 손실 | Q0 에서 CSV 시각 확인 |

**규모**: 71631 248 h 만으로 종료 토큰 ≈ 45 만 개(시간당 ≈ 1,900), NIKL 준자연 대화를 더하면 수백만 개. 사람 검증(③-c)은 수백 건이면 충분하다.

## 6. Turn-taking 라벨과 학습 목표

### 6.1 먼저 학습할 목표

`L = L_AR(lexical, speaker, NEXT, start_of_turn, end_of_turn) + λ_activity L_activity + λ_vap L_vap + λ_hazard L_hazard + λ_event L_event`

AR은 종류별 가중 CE, head 손실은 유효 프레임별로 정규화한다. Q1의 완전 주석 자연 창에서는 lexical·speaker·NEXT·start/end·activity를 공동 학습한다. 이후 VAP·semantic-event head·의미 ablation, 필요 시 hazard 순으로 추가한다. unlabeled replay·합성 데이터의 NEXT로 ‘EOT 없음’을 강제 학습하지 않도록 partial-supervision loss 경로를 둔다. event 라벨 생성·초기 가중·교사 강제 shift는 [[output-phase2-block-and-turn-label-spec]] §6–§9에 정의한다.

미래 음성 활동을 학습해 턴 이벤트를 유도하는 근거는 [VAP](https://arxiv.org/abs/2205.09812)다. 정답 현재 VAD를 모델 입력에 제공하는 실험은 oracle로만 구분한다. 배포 조건의 VAD는 모델이 mono에서 예측해야 한다.

### 6.2 80 ms 라벨 정합

원 VAP 구간은 현재 시각 이후 `[0,.2], [.2,.6], [.6,1.2], [1.2,2.0] s`다. 각 화자×4구간의 활동을 이진화해 256개 조합을 만든다. **Phase 2는 50 Hz 기준 VAD에서 이 원래 경계로 라벨을 계산하고, 80 ms마다 그 라벨을 샘플링**한다. 모델 출력률과 target 원천 해상도는 다를 수 있다.

현재 `targets.py`는 12.5 Hz일 때 경계를 `.16/.56/1.2/2.0 s`로 근사하고 any-pool을 쓴다. 기존 라벨을 조용히 재사용하지 말고 새 라벨 버전을 기록한다. 현재 활동의 80 ms 내 짧은 발화 여부(any)와 미래 bin의 활동 점유율(>50%)도 구분한다. [VAP bin 정의](https://aclanthology.org/2022.sigdial-1.51/)

Hazard 초판의 사건은 **현재 비활동인 각 화자의 다음 speech onset**으로 정의한다. 이는 floor transfer와 동일하지 않다. 이미 활동 중인 화자는 해당 onset 위험집합에서 제외하고, floor transfer·backchannel 구분은 event 층에서 한다. horizon은 2.56초이며 완전히 관측한 무사건 구간은 survival 항에 포함하고, 녹음 종료로 관측하지 못한 bin은 마스킹한다. 현행 `time_to_next_onset`의 한 개 `censored` 값만으로 충분한지 검사하고 관측 종료 시각을 추가한다.

### 6.3 의미·담화 라벨

VAD로 얻은 SHIFT/HOLD/INT/BACKCHANNEL은 약한 라벨이다. ‘잠깐 멈췄다’와 ‘생각이 끝났다’, ‘상대가 시작했다’와 ‘말할 기회를 넘겨줬다’는 같지 않다. 기존 [[output-vap-target-pipeline]]도 유도 SHIFT와 TurnBench EOT가 크게 다름을 기록한다.

자연 대화에서 EN/KO 각 약 500개 경계부터 사람이 `complete/incomplete/uncertain`, floor 관계, backchannel/interrupt를 검토하는 파일럿을 만든다. 불확실한 사례는 강제 분류하지 않는다. 20% 이상을 이중 검토하고 agreement·클래스별 confusion을 확인한 뒤 필요하면 언어당 1–2천 건으로 확장한다. 숫자는 초기 작업 예산 제안이다.

‘현재 시점에서 의미적으로 완결됐는가’는 prefix만 들은 판단, ‘실제로 이후 턴이 바뀌었는가’는 미래를 확인한 사건 라벨로 따로 저장한다. 둘을 같은 정답으로 쓰지 않는다. 기존 코퍼스 문장부호를 정답 EOT로 사용하지 않는다. 사람 표본은 학습 seed와 약라벨 검증/최종 평가로 대화 단위 분리한다. SoulX-Duplug의 상태 예측은 참고하되 현재 프로젝트의 자체 ASR 이력을 쓰는 조건에서 검증한다. [[source-soulx-duplug]]

### 6.4 의미 학습 트랙 S — Q0에서 준비, Q1/Q2와 병행

의미 라벨 수천 건이 반드시 부족하다는 근거는 없으나, 기존 계획은 의미 학습 신호의 규모·종류를 검증하는 실험이 부족했다. 다음 두 감독 신호를 분리해 추가한다.

1. **S-behavior**: train 대화의 시간·화자 순서가 확인되는 전사에서 prefix 뒤 실제 상대 화자 시작 여부를 약한 label로 만든다. 양성 경계만 추출하지 말고 발화 내부·같은 화자 재개·침묵·맞장구를 포함한다. 발화만 있는 데이터는 대화 복원 전 이 label에 쓰지 않는다.
2. **S-semantic**: `complete/incomplete/uncertain`을 prefix와 과거 대화만 본 로컬 LLM으로 제안하고 언어·유형별 사람 검증 뒤 학습용으로 확대한다. 미래 문장·다음 화자·후처리 문장부호·정답 종료 표지를 입력에 포함하지 않는다. 운율 없이는 판단할 수 없는 표본은 uncertain으로 남긴다. full-context 사후 사건 label을 따로 만들 수 있으나 prefix 완결성과 합치지 않는다.

Q0에서 언어당 5–20k prefix의 처리량·라벨 품질을 먼저 측정한 뒤 50–200k 규모 확대 여부를 결정한다. 수백만 ‘발화 수’를 유효 학습 경계 수로 간주하거나 ‘GPU 1시간’을 보장하지 않는다. 이미 사용 가능한 로컬 모델만 후보로 두고 대규모 실행·외부 API를 이 문서 작성 중 호출하지 않는다. 사람 라벨은 train seed, 약라벨 QA, 독립 평가 세 용도를 나누어 유지한다.

S-pretrain은 E2 복제본에서 동일 차원의 명시적 readout query와 semantic head를 이용한 text-prefix 학습이다. 먼저 thinker를 동결하고 head를 학습하며 필요할 때만 복제본 thinker를 저율 적응한다. 오디오를 임의로 0으로 채운 표현이 `h_audio`와 같다고 가정하지 않는다. Q3에는 **random head / frozen-thinker text-pretrained head**를 같은 오디오 backbone에서 비교하고, 자연 오디오-prefix 쌍으로 head를 재적응한다. text-adapted thinker 전체 전이는 별도 ASR guardrail 실험으로 제한한다.

S 트랙은 Q0의 데이터 계약이나 Q1/Q2 진행을 막지 않는다. Q3에서는 VAP로 기본 미래 활동을 먼저 확보하고 S-pretrain/약라벨 event 보조 학습으로 의미 기여를 검증한다. clean text 학습 후 ASR 오류·중간 prefix·실제 방출 이력으로 적응해야 한다.

## 7. 단계별 실험과 중단 기준

각 단계는 직전 통과 체크포인트를 기준으로 한다. 실험 비교는 동일 초기값·데이터·유효 노출량·seed를 사용한다. 다음 수치는 **파일럿용 제안 관문**이며 Q0에서 baseline을 측정한 뒤 장기 run 전에 동결한다. test 결과를 보고 관문을 바꾸지 않는다.

| 단계 | 추가하는 능력 / 데이터 | 주요 산출물 | 다음 단계 진입 |
|---|---|---|---|
| Q0 기준선·데이터 계약 | E2 δ=2/4, ASR·자연 대화·합성 평가; 2–5 h QC pack | frozen IDs, mixer/serializer, 18개 블록 fixture·turn 라벨 생성 계약, 저장 provenance | 누락·덮어쓰기·화자/미래 누출 0; 평가 재현 |
| Q1 비중첩 2화자 전사 | EN/KO 약 20–50 h + 완전 주석 자연 turn 창 | G1 전사+start/end+activity, 32창 overfit, recipe·memory 비교 | ASR guardrail·비중첩 DER≤10%·귀속 오류≤5%, pause/start/end 동작 검증 |
| Q2 겹침 전사 | Q1 + 자연 overlap·제어 합성, 자연 turn 주석 | 양 화자 전사·독립 start/end·맞장구와 중단·두 OPEN 상태 | overlap cp 오류 ≥20% 상대 감소, 화자 소실≤5%, 화자별 이벤트 검증 |
| Q3 미래 활동·의미 예측 | 자연 대화 turn supervision + 검증된 S 학습. δ=2 첫 조건/δ=4 비교 | VAP → semantic-event·의미 ablation → 필요 시 hazard | 동일 FPR에서 음향 대조군 대비 검증 가능한 개선, ASR guardrail 유지 |
| Q4 장문·실시간 통합 | 실제 및 합성 10–60분 세션, 자유실행, 잡음/음량차 | 2행 전사+활동+turn UI, 단독 장치 벤치 | 지속 RTF <1, backlog 비발산, ID 지속성·지연 관문 충족 |

32개 창 overfit는 코드 경로 검사다. unseen 화자 일반화의 근거로 쓰지 않는다. 초기 자연 데이터는 화자 다양성·턴 수·겹침 시간을 기준으로 추출하며 단순 폴더 순 20시간을 쓰지 않는다.

### 7.1 학습 recipe 시작점

- 모든 초기화는 E2에서 시작한다. E2 encoder 가중치를 유지하며 새 speaker 행/heads만 초기화한다. ‘새 Qwen’ 재초기화와 비교하지 않는다.
- 동결-only 규약 smoke 후 Q1에서 **R-E2 / R-low 두 조건**을 짧게 비교한다. R-E2는 encoder 해동, thinker `2e-5`, encoder `1e-5`, adapter `1e-3`의 E2 설정을 후보로 둔다. R-low는 같은 해동 범위에서 이 세 LR을 각각 절반으로 낮춘다. 새 heads LR `1e-4`는 두 조건에 동일하게 두고 optimizer group을 명시한다. 두 recipe 묶음의 선택 실험이지 특정 모듈 LR의 인과효과를 분리하는 실험은 아니다. 원안의 더 낮은 adapter LR은 성능 근거가 없어 기본값에서 제외한다.
- 같은 데이터 노출량·batch·scheduler 비율·seed로 warmup 이후의 학습/자유실행 sentinel을 비교하고, ASR·activity·태그·조기 방출을 기준으로 장기 recipe를 선택한다. 100–300 step은 수치·속도 smoke이며 수렴 판정이 아니다. 두 조건 모두 불안정하면 R-low와 같은 LR에서 encoder 동결을 추가해 원인을 분리한다. E2에서는 LR과 encoder 상태가 함께 바뀌었으므로 해동만이 붕괴를 해결했다고 단정하지 않는다.
- E2의 `next_weight EN/KO=0.3/0.15`, delay 분포, TN을 첫 run에 유지한다. 태그 추가가 NEXT 비율을 바꾸므로 삭제·조기 방출·태그율을 보고 별도 sweep한다.
- Q1/Q2 시작 배치 예산은 대화 70% + 단일 화자 replay 30%의 **오디오 초 기준**으로 제안한다. 대화 내 합성 비율은 최대 약 절반부터 시험하며 자연 대화 표본을 유지한다. Q3의 turn 손실은 자연 대화에만 적용한다.
- Q3는 head-only probe 이후 공유 모델 저율 joint를 비교한다. gradient norm·ASR 회귀를 보고 손실 가중을 정한다. 발화 길이와 corpus 구성 때문에 ‘30 epoch’를 이전 run과 같은 노출량으로 가정하지 않는다.
- free-running prefix에서 head를 학습하는 단계를 포함한다. gold prefix 성능은 oracle upper bound로만 보고한다. 같은 checkpoint로 생성한 rollout은 model hash를 기록하고 모델 변경 후 갱신한다.

### 7.2 장문과 speaker memory

20–40초 독립 창 통과 뒤 60–120초 carry 학습, 10–60분 검증으로 늘린다. 과거에 관측한 음성·전사·speaker state만 carry한다. 오래 침묵한 B가 돌아오는 사례와 처음부터 겹치는 사례를 별도 평가한다.

현재 KV를 무제한 유지하면 메모리가 증가한다. bounded context + 세션 speaker memory를 비교하고, memory는 관측 prefix에서만 갱신한다. 단순 KV 앞부분 삭제가 RoPE·화자 정체성을 보존한다고 가정하지 않는다. context 전환 시 logits·전사·ID 연속성, 메모리 상한을 테스트한다.

### 7.3 중간 산출물과 완료 범위

**MVP-ASR/SD** = Q1+Q2+activity+화자별 start/end 생성이며 해당 관문·live smoke를 통과한 버전이다. endpoint token 생성만으로 의미 기반 미래 turn 예측을 달성했다고 부르지 않는다. **MVP-Turn**은 Q3 VAP+semantic 검증을 더한다. hazard·추가 세분화 토큰은 미룰 수 있으나 사용자 요구인 start/end 생성과 의미 ablation은 필수다. 최종 완료에는 ASR 강화·Q4 장문·실시간 결과도 필요하다.

## 8. 평가 및 ‘의미를 이해했다’의 판정

### 8.1 네 가지 성능을 함께 보고

| 축 | 주 지표 | 규약 |
|---|---|---|
| 단일 화자 ASR | EN WER, KO CER; S/D/I | E2와 같은 dev/test·TN·δ. corpus별 상대 회귀 ≤5% 제안 |
| 다화자 전사 | EN cpWER/tcpWER, KO 문자 단위 permutation CER; attribution 오류 | 대화 전체 하나의 A/B permutation. overlap/non-overlap·음량차·언어별 분리 |
| Diarization | overlap 포함 DER: miss/FA/confusion, JER, ID switch | collar=0 주 지표, 250ms 보조. 80ms 해상도를 명시 |
| Overlap | 활동 overlap precision/recall/F1, 양 화자 전사 재현율 | 한쪽 발화를 누락하는 실패를 전체 평균에 숨기지 않음 |
| Turn-taking | EOT/INT recall–FPR–latency, BC F1, calibration | official scorer 고정, 임계값 dev 선택, test 고정 |
| 생성된 turn 이벤트 | 화자별 start/end P/R·false-end/min·누락·중복·귀속 오류·latency | native contribution 종료와 공식 TurnBench 사건 매핑을 구분, pause·맞장구·overlap별 평가 |
| 시간·시스템 | 방출 p50/p90/p99·viol80·matched coverage, TTFT, RTF·tick·backlog | 오디오 가용 시각/계산/네트워크·flush 별도; 단독 장치와 경합 구분 |

cpWER/tcpWER는 [MeetEval](https://github.com/fgnt/meeteval)의 버전·매핑·시간 collar를 고정한다. KO 문자 기준 지표는 EN WER와 이름을 구분한다. 각 화자 전사를 누적한 뒤 채점하며 청크마다 reference와 최적으로 재매칭하지 않는다. 전사 시간 제약의 수 초 collar가 80ms 방출 정확도를 보증하지 않으므로 timing 지표를 따로 둔다.

단일 화자 E2와 다화자 cpWER를 직접 나눠 ‘회귀 5%’를 적용하지 않는다. 동일 입력·채점 조건에서 Q1/Q2의 직전 기준 모델과 비교한다. 깨끗한 분리 채널에 E2를 각각 적용한 결과는 정보가 더 많은 **oracle 참고선**이다. RNN-T 다화자 대조군도 실제 mono diarization/분리 비용을 포함한 완성 파이프라인으로 비교한다.

‘화자 소실’은 각자 참조 단어/문자가 5개 이상 있는 평가 창에서 어느 한 화자의 올바른 matched lexical recall이 10% 미만인 경우로 정의한다. 짧은 맞장구는 이 분모에서 제외하고 별도 recall을 낸다. Overlap token은 참조 token 시간 구간이 양 화자 VAD 동시 활동과 겹치는지로 분류하며 정렬 품질별 결과를 병기한다.

### 8.2 의미 기여를 검증하는 최소 대조군

1. **Audio-only**: 같은 mono encoder, 동일 시간/문맥 예산, 비슷한 head 용량의 causal turn predictor.
2. **Integrated**: 같은 오디오와 자체 생성 이력을 본 thinker audio-state head.
3. **Text ablation**: 입력 오디오는 유지하고 lexical 이력을 마스킹/교란한 조건. 분포 이동 영향 때문에 이에 맞춰 학습한 대조군도 둔다. speaker tags·활동 이력은 유지한다.
4. **Oracle history**: 현재 관측 시간까지 정렬상 사용 가능한 gold 전사만 제공. 완성된 미래 문장을 넣지 않는다. 실제 제품 성능으로 보고하지 않는다.

동일 침묵 길이·비슷한 운율의 완결/미완결 문장, 질문/진술, 한국어 연결어미·종결어미, 긴 망설임, 맞장구 뒤 본 화자 계속 발화로 구성한 사람이 검수한 subset을 둔다. 예: “제가 생각한 건…”과 “제 생각은 여기까지예요.”의 pause를 맞춘 비교. 조작된 음성 실험은 자연 대화 결과와 별도 보고한다.

개선 목표는 같은 FPR≤0.10에서 EOT/INT recall **+3%p 이상** 또는 recall을 유지하며 EOT p50 **80ms 이상 단축**으로 제안한다. 대화 단위 paired bootstrap 95% CI와 두 개 이상 seed로 확인한다. Q0에서 표본 크기·기준선을 보고 이 수치를 사전 동결한다. 마스킹 효과만으로 일반적 ‘의미 이해’를 증명했다고 쓰지 않으며, audio-only 대비 개선과 의미 subset의 일관된 결과가 함께 있어야 가설을 지지한다.

TurnBench는 기존 공식 규약을 따라 비교하되, mono 혼합으로 바꾼 입력임을 명시한다. 기존 stereo VAP 숫자는 별도 입력 조건의 참고선이다. 기존 dev 반복 사용 사실과 hidden test 접근 여부를 기록하고, KO는 검수된 held-out 대화에서 같은 규약을 적용한다. [[turn-taking-evaluation-protocol]], [TurnBench 논문](https://arxiv.org/abs/2608.25218)

### 8.3 최종 채택 관문

- 위 ASR guardrail, Q1 화자 귀속 관문, Q2 overlap 개선, Q3 의미 기여 결과를 동시에 보고한다. 하나의 합산 점수로 약점을 가리지 않는다.
- 타이밍 목표는 신뢰도 높은 정렬 subset의 `viol80 ≤1%`, p99 방출 지연 ≤1초로 제안하고 정렬 불확실/미매칭 비율도 낸다. 기존 ‘위반 0’보다 완화한 **별도 제안**이며, 원래 관문 통과 주장에는 쓸 수 없다. 코드상 미래 정보 누출은 0건이어야 한다.
- 고정 장치에서 encoder+decoder+heads의 tick p99 <80ms를 지향하고 RTF <1을 필수로 둔다. 주기적 초과가 있으면 backlog p99·최대값·지속 추세를 함께 판정한다. 1시간 스트림에서 backlog가 계속 증가하면 채택하지 않는다.
- 10분 단위 ID swap을 기록하고 장문 총 ID switch ≤1회/10분을 초기 목표로 둔다. 사후 전체 파일 relabeling으로 온라인 성능을 대신하지 않는다.
- Turn 결과는 teacher forcing 없이 실제 전사 이력으로 평가한다. 공식 event 판단 시각에 lookahead를 포함하고, 서비스 지연은 wall-clock으로 추가 보고한다. 미래 onset 예측은 ASR 조기 전사 위반과 다른 개념이다.

## 9. 구현 작업 지도와 검증 순서

아래 파일명은 신규 구현 제안이며 아직 생성하지 않았다. 기존 HF Trainer+Liger를 유지하고 프레임워크 교체는 이 실험 축에 섞지 않는다.

| 작업 | 위치 | 필수 확인 |
|---|---|---|
| 대화 스키마·혼합·QC | 신규 `vapasr/data/dialogue.py`, schema, `experiments/p2_build_dialogue.py` | 합산·offset·채널·분할·태스크 마스크 |
| 청크별 화자 전사(G0/G1) | 신규 `vapasr/data/dialogue_interleave.py` | 두 전사로 완전 복원, Unicode·공백, 그룹 순서 편향·flush·cap |
| 모델 출력·손실 | `vapasr/hf/modeling_vapasr.py`, config | SPK/start/end vocabulary·mask, per-speaker state, event CE·partial supervision, E2 호환 |
| turn 라벨·이벤트 상태 | 신규 `dialogue_turn_labels.py`, `p2_build_turn_labels.py`, `dialogue_state.py` | 무음/pause/EOF·두 OPEN·지연 전사/이벤트·사전 주석 coverage, 블록 명세 fixture |
| activity/VAP/hazard | `vapasr/data/targets.py`의 버전 분기 | 50Hz 참조 경계, 80ms sampling, censor/unknown, permutation |
| 학습·replay | Trainer/data, 신규 `experiments/p2_train_hf.py` | 초 기준 mixture·재개·rollout provenance·DDP 불균등 평가 |
| 평가 | 신규 `experiments/p2_eval.py` | speaker-aware 전사·DER·turn·latency, dev/test 불변 |
| live | `vapasr/hf/live.py`, `live_mlx.py`, `experiments/live/` | A/B별 버퍼·80ms heads·타임스탬프·최초 출력·장문 |

특히 E2 encoder를 **동결한 뒤 저장/재로드해도 E2 가중치가 유지되는지** 검사한다. 현재 저장 로직은 `encoder_trainable`에 따라 encoder를 생략하므로 학습 여부와 저장할 가중치 provenance를 분리해야 한다. frozen E2 대신 원본 `.nemo`를 재부착하면 다른 모델이다.

정확한 위험 조건은 config의 `encoder_trainable=False` 또는 저장 ignore 목록 때문에 E2 가중치가 생략되는 경우다. `requires_grad=False`만 설정하고 저장 flag를 유지하면 반드시 소실되는 것은 아니다. Phase 2 checkpoint는 학습된 E2 encoder를 동결 여부와 무관하게 내장하고 hash를 검증한다. 원본 backbone을 외부 참조할 때는 정확한 artifact hash를 필수로 하며, 누락된 학습 encoder를 원본 `.nemo`로 조용히 대체하지 않는다.

현재 deferred `<NEXT_AUDIO>` 최적화는 다음 audio와 묶어 forward한다. 새 head는 묶음의 마지막 audio 위치를 명시적으로 읽어야 한다. MLX가 지금처럼 마지막 logits만 돌려주는 경로에는 hidden-state/head 지원이 필요하다. CPU 왕복 때문에 얻었던 속도 이득을 잃지 않는지도 확인한다.

검증 순서는 serializer round-trip → 실제 Dataset 오디오 spot-check → 32창 overfit → HF save/load logits·heads·encoder parity → prefix 절단/미래 교체 인과성 → 단일/불균등 multi-rank smoke → 동일 held-out 자유실행 → live parity → 장문이다. tiny run에서 optimizer/scheduler/RNG·샘플 위치·split hash 재개도 확인한다.

## 10. 실행 예산과 우선순위

1. **첫 묶음, 약 2–4 작업일 제안:** Q0 데이터 계약·평가 pack·기준선, mono mixer와 serializer, 작은 overfit. 이 단계 종료 전 대규모 job을 제출하지 않는다.
2. **둘째 묶음, 약 1주 제안:** Q1 비중첩 화자 전사와 Q2 overlap pilot. 화자 추적·낮은 음량 삭제·생성량 증가를 먼저 해결한다.
3. **셋째 묶음, 약 1–2주 제안:** Q3 audio-only/thinker heads, semantic annotation pilot, 자유실행 history 학습 및 ablation.
4. **넷째 묶음, 약 1주 이상 제안:** Q4 장문·실시간 통합과 고정 test 보고. 사람 라벨링·데이터 접근·선점 대기는 별도다.

이는 연구 개발 순서와 대략적인 인력 일정이지 GPU 완료 시간 예측이 아니다. Q0/Q1의 100–300 step으로 `audio-seconds/GPU-second`, tokens/second, peak memory, 평가 RTF를 실측한다. 이후 `GPU-hours = 총 노출 audio-seconds / 실측 처리량 / 3600`으로 예산을 계산하며 정렬·rollout·eval 비용을 따로 더한다. 최초 pilot은 1–8 GPU로, 통과 후 필요한 규모로 확장한다.

### 10.1 ASR 강화 트랙 A — Phase 2 필수 산출물

Q0 기준선 이후 Q1–Q3와 병행 가능한 독립 실험 묶음으로 둔다. 인력/GPU 자원이 부족하면 순차 실행하되 개발 목표에서는 빼지 않는다.

| 실험 | 바꾸는 축 | 채택 규칙(장기 run 전 동결할 제안) |
|---|---|---|
| A1 저지연 정확도 | E3 추가 학습 / next_weight / delay 분포를 각각 별도 run | δ=2의 사전 지정 EN dev-other·KO 대화 오류 macro 상대 개선 ≥5%, 각 corpus 회귀 ≤5%, timing 악화 없음 |
| A2 대화·환경 적응 | 대화 corpus 가중 / 잡음·RIR augmentation 각각 비교 | held-out 대화/잡음 개선과 clean ASR guardrail, RIR의 라벨 시간 이동·tail 규약 고정 |
| A3 추론 속도 | static KV·graph·encoder backend를 한 번에 하나씩 검증 | 고정 장치·조건의 logits/출력 parity 및 tick·RTF 개선, p99<80ms 최종 목표 |

‘ASR 강화 완료’는 A1/A2의 인식 개선 하나와 A3의 실시간 개선을 각각 측정하여 판정한다. 미달이면 개선 미달로 보고하고 다화자 추가만으로 대체하지 않는다. Phase 2의 E2 대비 전사 회귀 guardrail도 계속 유지한다.

**승격 규칙**: A의 ASR-only checkpoint를 채택하는 최초 분기는 Q1 정식 학습 시작 전 1회로 한다. 이후 도착한 A 개선을 채택하려면 `A* → Q1 → Q2`를 재실행하거나, 기존 다화자 checkpoint에서 해당 recipe를 다시 적용한 별도 branch로 검증한다. Q2의 thinker/encoder를 A 가중치로 단순 교체하지 않는다. Q3 시작 전에 실제 parent checkpoint를 동결하고 Q0 pack·Q1/Q2 능력을 재검증한다. 수치 동등한 runtime 최적화는 모델 가중치 승격과 별개로 반영 가능하다.

### 10.2 슬롯 조건 오디오 토큰 실험의 진입 조건

Q2가 실패하면 mixer/cap/encoder 적응과 speaker memory를 먼저 확인한다. memory가 안정적이고 음향 정보가 남아 있는데 단일 표현 경로가 병목이라는 진단이 있을 때 **mono-derived 2 audio token**을 multi-output decoder보다 앞선 구조 ablation으로 둔다. 물리적 분리 채널 입력은 아니다.

현행 encoder 출력은 80ms당 1프레임이다. 두 slot query의 key/value는 `현재 1프레임 + causal 좌측 문맥`으로 정의해야 하며, ‘청크 내 8 encoder 프레임’은 존재하지 않는다. 10ms mel/중간 feature를 쓰려면 별도 추출·projection·인과성 검증이 필요하다. 두 query가 같은 내용을 복제할 수 있어 화자별 전사·activity·보조 speaker supervision으로 검증한다. 두 토큰이 분리 능력을 보장하지 않는다.

15분은 `K=900/0.08=11,250`청크다. 2 audio token이면 `22,500 audio + 11,250 NEXT = 33,750` 위치로, **텍스트·태그·prefix 전부터 32,768을 넘는다**. 실제 context 설정과 bounded context를 확인한다. v1.2의 event 수 E까지 포함한 총량은 `2K+T+S+E+P`에서 `3K+T+S+E+P`로 바뀌므로 KV 전체가 정확히 2배인 것도 아니다. selector는 S, start/end는 E에 집계하고 처리량·KV·wall-clock을 다시 측정한다. 두 번째 audio 위치에서 head를 한 번 계산하고 추가 비용을 반영한다.

## 11. 주요 실패 가설과 다음 조치

| 관측 | 먼저 구분할 원인 | 다음 한 가지 실험 |
|---|---|---|
| 작은 화자 전사가 사라짐 | mixer 결함 / encoder 혼합 정보 부족 / decoder cap | clean-channel oracle와 저음량 합성으로 위치 확인 후 encoder 해동 범위 비교 |
| 전사는 맞는데 A/B가 뒤집힘 | local ID 계약 / 긴 문맥 손실 / speaker 표현 부족 | Q1/Q2 조기 memory on/off·carry 검증, 필요 시 t-vector 보조 학습 |
| overlap F1은 높지만 전사는 한 사람뿐 | 활동 감지와 음성 내용 분리 능력 차이 | 화자별 D/S/I·cap·memory 점검 후 슬롯 조건 2 audio token, 이후 multi-output decoder 비교 |
| turn head는 gold에서만 좋음 | ASR 지연·오류 이력의 노출 편향 | self-generated history 학습, δ별 turn curve |
| thinker가 audio-only보다 못함 | semantic 표현/라벨 부족 또는 80ms 운율 손실 | 라벨 검수 후 같은 mono의 고해상도 causal acoustic branch ablation |
| 평균 속도는 좋고 장문은 밀림 | burst·KV 증가·speaker token 비용 | context 상한·cap·runtime profiling, 같은 정확도에서 개선 검증 |

고해상도 branch, 분리 보조 학습, multi-output decoder는 최초 주 경로에 넣지 않는다. 관측된 실패 원인을 겨냥한 후속 비교이며 입력은 계속 mono다. Q3에서 의미 기여가 확인되지 않으면 ‘공유 모델로 화자 전사와 활동을 처리했다’까지 주장하고 의미 기반 턴 예측 달성은 보류한다.

## 12. 근거와 남은 불확실성

프로젝트 근거는 [[output-stage2-e2-final-eval]], [[decision-mono-input]], [[output-vapasr-model-and-sequence]], [[source-hf-trainer-migration]], [[output-vap-target-pipeline]]와 이번에 읽은 `vapasr/hf/`, `vapasr/data/streams.py`, `vapasr/data/targets.py`다. 코드 관찰은 2026-09-11 작업 트리 기준이다.

외부 1차 자료 확인일: 2026-09-11. [t-SOT](https://arxiv.org/abs/2202.00842)와 [t-vector](https://arxiv.org/abs/2203.16685)는 직렬화·화자 귀속의 근거이며 E2+LLM의 성능을 보증하지 않는다. [VAP 공식 구현](https://github.com/ErikEkstedt/VoiceActivityProjection)은 현재·미래 활동 공동 학습과 입력 조건을 확인하는 참고다. [MeetEval](https://github.com/fgnt/meeteval)은 다화자 전사 평가, [TurnBench](https://arxiv.org/abs/2608.25218)는 턴 이벤트 평가 근거다.

후속 보유량 기록은 [[output-phase2-plan]]에 연결했다. 확인 전 항목은 실제 train 가능량·대화/화자 누출, 자연 overlap 정렬 신뢰도, E2 speaker 표현, mono turn 기준선, 그룹 순서 편향, 2화자 처리량, 약라벨 품질·의미 head 전이 효과다. 이 항목들은 Q0–Q3의 측정 대상이다. P1–P9 판단과 새 제안의 한계는 [[output-phase2-critique-response]]에 남긴다.
