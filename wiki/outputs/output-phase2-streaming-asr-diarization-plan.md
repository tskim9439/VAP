---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: E2 기반 최대 2화자 mono 스트리밍 ASR·겹침 전사·화자 추적·의미 기반 turn-taking 개발 계획과 단계별 검증 관문
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
---

# Phase 2 개발 계획: Streaming ASR & Speaker Diarization / Turn-Taking

작성 기준: 2026-09-11. 사용자가 제시한 최대 2화자 목표를 구체화한 **실행 제안서**다. 아래 새 구조·수치 관문·예산은 제안이며, 구현 완료나 성능 보장이 아니다.

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

E2 체크포인트·tokenizer·TN 지문·평가 split을 고정한다. δ=4를 Phase 2 최초 주 평가점, δ=2를 저지연 평가점으로 삼는다. δ=4에서 보이는 품질을 δ=2 지연과 한 숫자로 합쳐 보고하지 않는다. E2의 ASR 방출 타이밍 학습은 turn-taking의 의미 이해를 입증하지 않는다.

## 3. 권장 모델 구조

```text
16 kHz mono 혼합 음성
        ↓
E2 Nemotron streaming encoder [56,0] → adapter
        ↓
Qwen3-ASR thinker: [AUDIO_k] → 화자 태그·전사 → <NEXT_AUDIO>
        ├─ 생성 경로: <SPK_A/B> + lexical text → 화자별 전사 버퍼
        └─ h_audio[k]: 현재 오디오 위치의 causal hidden state
              ├─ 현재 활동: [P(A active), P(B active)]
              ├─ 미래 활동: VAP 256-class
              └─ 후속 단계: 화자별 onset hazard / semantic-event heads
```

분리 채널은 라벨·정렬·훈련용 혼합 생성에만 사용한다. 추론에는 mono만 제공하고, 정답 VAD·정답 화자·깨끗한 채널·완성 전사·전체 파일 통계를 넣지 않는다. 이는 [[decision-mono-input]]을 유지한다. 화자별 활동 헤드는 별도 외부 diarization 시스템이 아닌 공유 모델의 보조 출력이다.

### 3.1 왜 이 구성을 먼저 시험하는가

기존 thinker와 80 ms 시간축, `<NEXT_AUDIO>` 규약을 재사용한다. 겹침 전사는 한 decoder에서 시간순으로 직렬화한다. 다화자 토큰을 단일 출력 경로에 직렬화하는 선행 근거는 [t-SOT](https://arxiv.org/abs/2202.00842)다. 다만 원 논문의 virtual channel과 세션 내 고정 화자 ID는 같지 않으므로 본 계획의 A/B 추적 능력은 별도로 검증한다.

오디오마다 예측하는 활동 헤드가 있어야 텍스트가 없는 무음·호흡·겹침에서도 상태를 낼 수 있다. 기존 `<SPK_A/B>`는 현재 HF 디코드에서 차단되어 있고 활동·turn 헤드도 없다. 예약 토큰이 있다는 이유로 기능 구현이 완료된 것은 아니다.

### 3.2 Audio-clock 헤드와 의미 정보

`h_audio[k]`는 `[AUDIO_k]`를 처리한 직후, **같은 청크의 전사를 생성하기 전** 상태로 고정한다. 현재까지의 오디오와 이전 청크까지 방출한 텍스트를 볼 수 있다. 현재 청크의 gold 토큰이나 미래 발화를 attention으로 볼 수 없게 한다. 활동·turn 출력은 텍스트 생성 여부와 무관하게 80 ms마다 계산한다.

이 선택은 turn 출력이 ASR flush를 기다리는 문제를 줄이지만, 최근 확정 전사는 δ만큼 늦게 들어온다. δ=2/4별 turn 성능도 비교해야 한다. 같은 청크 전사를 본 뒤의 헤드는 후속 ablation으로만 추가하고 추가 생성 시간이 포함된 가용 시각으로 채점한다.

초기 헤드는 작은 MLP로 두고 E2 표현의 유효성을 확인한다. 활동은 2개 sigmoid로 두어 `00/10/01/11`을 모두 허용한다. 중첩 확률은 초기에는 `VAP`의 현재 상태 대체가 아니라 현재 활동 출력으로 평가하며, 두 sigmoid의 곱은 독립 가정이므로 필요 시 4상태 공동 분포 헤드와 비교한다.

장기 ID 유지가 약하면 동일 모델 내부에 causal speaker memory 또는 token-level speaker embedding 보조 손실을 추가한다. [t-vector 연구](https://arxiv.org/abs/2203.16685)는 겹침 전사 토큰에 화자 표현을 대응시키는 선행 근거다. 이 확장은 최초 run에 한꺼번에 넣지 않는다.

## 4. 화자·전사·시간 계약

### 4.1 A/B 정체성

- 최초로 식별 가능한 화자를 A, 다음 새 화자를 B로 배정하고 세션 중 유지한다. A/B를 남녀·채널 번호·큰 목소리와 연결하지 않는다.
- 시작부터 겹쳐 식별이 모호하면 임시 슬롯으로 시작하고 confidence를 기록한다. 주 지표는 최초 출력 기준이다. 나중에 ID를 고쳤다면 수정 횟수·확정 지연도 보고한다.
- 학습의 channel→speaker 매핑, 전사 태그, 활동·VAP·hazard 라벨은 **같은 permutation**을 사용한다. 무작위 channel 순서와 gain을 바꾸어 지름길 학습을 검사한다.
- 독립 crop은 관측 prefix 내 최초 화자 기준으로 일관되게 재매핑한다. carry 학습·장문 평가는 세션 ID를 유지한다. 미래 단독 구간이나 전체 파일 clustering 결과로 온라인 A/B를 정하지 않는다.
- PIT를 쓰는 실험도 permutation은 crop/세션 단위로 고정하며 청크마다 최적 permutation을 바꾸지 않는다. 전사 손실과 활동 손실을 따로 permutation하면 ID 의미가 충돌한다.

### 4.2 겹침 전사의 직렬화

각 화자의 lexical 전사를 독립적으로 tokenize·정렬한 뒤 토큰 종료 시각으로 합친다. 동일 시각은 고정된 A→B 및 원래 토큰 순서로 결정하고, 화자 내부 순서를 보존한다. 기본 청크 배정은 E2의 `k=floor(t_end/0.08)+δ`를 유지한다.

```text
[AUDIO_k] <SPK_A> 그때 <SPK_B> 응 <SPK_A> 내가 <NEXT_AUDIO>
[AUDIO_k+1] 갔거든 <SPK_B> 맞아 <NEXT_AUDIO>
```

예시는 직렬화 형식을 설명한다. 오디오에는 실제 겹침이 그대로 남고, 출력 문자열만 번갈아 기록한다. 청크 경계에서도 현재 전사 화자를 유지한다. 최초 lexical 출력과 화자 변경 때 태그를 내며, `<SPK_A><SPK_B>`처럼 내용 없는 태그 반복은 문법으로 제한한다. 태그는 다음 오디오의 현재 화자를 뜻하는 것이 아니라 **뒤따르는 전사 토큰의 귀속**이다.

BPE byte 조각 사이에 다른 화자의 토큰을 끼워 넣으면 화면 텍스트가 깨질 수 있다. 같은 정렬 단위의 토큰 묶음은 원자적으로 유지하고 화자별 decoder buffer로 Unicode/공백을 복원한다. 단어 전체 완료까지 기다리는 변형은 추가 지연을 따로 잰다. TN은 현재 lexical 규약과 지문을 유지하고 화자 태그를 문자열 정규화에 섞지 않는다.

두 화자의 전사와 태그 때문에 청크당 생성량이 증가한다. 현재 안전 상한 8 토큰을 그대로 적용하거나 단순히 두 배로 올리지 않고 실제 밀도 p99·강제 NEXT·삭제율·처리 지연을 함께 측정한다. 구조 토큰과 lexical 토큰 예산을 별도로 기록한다. 학습의 무제한 방출과 추론 상한 차이도 QC한다.

### 4.3 출력 API와 세 개의 시각

전사 이벤트는 `session_id, speaker_id, token_ids/text, chunk_index, audio_seen_until, emitted_at, confidence`를 가진다. 활동·turn 이벤트는 `audio_seen_until, emitted_at, activity[2], vap_probs, onset_hazard, event_probs`를 제공한다. 현재 없는 필드는 구현 단계에 맞춰 추가한다.

`audio_seen_until`은 실제 읽은 마지막 오디오 시각, `emitted_at`은 실제 결과 가용 시각이다. `estimated_speech_start/end`를 제공할 경우 별도 시간 추정 head 또는 causal aligner가 필요하며 추정값임을 표시한다. `방출 시각−δ`를 정확한 발화 경계로 간주하지 않는다. 초판은 활동 구간과 전사 방출 시각을 제공하고, tcpWER에 사용할 단어/구간 타임스탬프 생성 규약을 평가기에서 고정한다.

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

자연 대화는 원래 gap·중첩·맞장구·발화 순서를 보존한다. 비중첩 단계에서는 긴 대화 중 적합한 창을 선택하며 턴 사이 무음을 삭제하거나 발화를 당겨 붙이지 않는다. 합성 겹침의 VAD는 쓸 수 있지만 `L_VAP/L_hazard/L_semantic-event`는 초기에는 마스킹한다. 단일 화자 replay도 turn 손실을 마스킹한다.

합성 스트림은 20–40초부터 시작하고 다음 조건을 분리한다: 무겹침, 짧은 맞물림, 10–30% overlap, 30–50% stress, 같은 시각 시작, 한쪽 음량이 3/6/12 dB 낮음, 비슷한 음색. overlap 비율 분모는 ‘적어도 한 명이 말하는 시간’으로 고정한다. 모든 합성 원본은 train split에 속해야 한다.

### 5.2 스키마와 로더

기존 `vapasr/data/streams.py::assemble_stream`은 `out[o:o+n] = x[:n]` 대입이라 겹치는 입력을 합산하지 않는다. **Phase 2 전용 대화 mixer**를 만들고 다음을 계약화한다.

- 입력 파형은 자연 mono 원음 우선, 없으면 동기화된 두 채널을 고정 규칙으로 합산한다. clipping·반대 위상·채널 누설·샘플레이트·sample offset을 검사한다. 추론 정규화는 미래 파일 통계에 의존하지 않는다.
- `conversation_id, source_speaker_id, local_speaker_slot, source_channel, src_offset_s, mix_offset_s, duration, gain, split, alignment_quality, task_masks`를 명시한다. 기존 스키마를 묵시적으로 바꾸지 않고 별도 버전과 변환기를 둔다.
- Forced alignment는 각 깨끗한 채널에서 수행한 뒤 혼합 시간축으로 옮긴다. 혼합 파형에 단일 화자 aligner를 그대로 적용하지 않는다. 불확실한 token 경계는 타이밍 손실/평가에서 분리 집계한다.
- 원 대화 길이의 VAD를 먼저 만들고 crop별로 참조한다. crop 뒤의 2.56초까지 라벨 산출에 쓸 수 있으나 그 미래 오디오는 모델 입력에 넣지 않는다. 녹음 끝·주석 누락은 관측 마스크로 처리한다.
- 화자별 채널 매핑을 파일마다 검증한다. 기존 AI Hub 기록에 channel swap이 있었고 발화 EndTime은 정밀 VAD가 아니다. 에너지 VAD도 검수된 정답으로 취급하지 않는다.
- E2까지의 학습 데이터와 화자·대화·오디오 중복을 감사한다. pretraining 노출, Phase 2 train, dev, untouched test를 구분한다. crop을 만든 뒤 무작위 split하지 않는다.

첫 QC는 EN/KO 각 50개 대화 창의 파형·채널·혼합·전사·정렬·활동 라벨을 실제 학습 Dataset 객체를 통해 점검한다. 마지막 1바이트 PCM, EOF, archive channel 선택, 원본 offset, 무음 화자, 전사 없는 speech, 화자 교체, 채널 순서 반전도 회귀 사례에 넣는다.

## 6. Turn-taking 라벨과 학습 목표

### 6.1 먼저 학습할 목표

`L = L_lexical + λ_spk L_spk + λ_next L_next + λ_activity L_activity + λ_vap L_vap + λ_hazard L_hazard + λ_event L_event`

각 손실은 유효 토큰/프레임 수로 각각 정규화하고 언어별로 기록한다. 기존 next_weight의 정규화와 새 표현의 수치 동등성을 확인한다. 초기에는 lexical·speaker·NEXT·activity만, 이후 VAP, 마지막 hazard와 event 순으로 추가한다. 처음부터 여러 손실의 가중치를 동시에 탐색하지 않는다.

미래 음성 활동을 학습해 턴 이벤트를 유도하는 근거는 [VAP](https://arxiv.org/abs/2205.09812)다. 정답 현재 VAD를 모델 입력에 제공하는 실험은 oracle로만 구분한다. 배포 조건의 VAD는 모델이 mono에서 예측해야 한다.

### 6.2 80 ms 라벨 정합

원 VAP 구간은 현재 시각 이후 `[0,.2], [.2,.6], [.6,1.2], [1.2,2.0] s`다. 각 화자×4구간의 활동을 이진화해 256개 조합을 만든다. **Phase 2는 50 Hz 기준 VAD에서 이 원래 경계로 라벨을 계산하고, 80 ms마다 그 라벨을 샘플링**한다. 모델 출력률과 target 원천 해상도는 다를 수 있다.

현재 `targets.py`는 12.5 Hz일 때 경계를 `.16/.56/1.2/2.0 s`로 근사하고 any-pool을 쓴다. 기존 라벨을 조용히 재사용하지 말고 새 라벨 버전을 기록한다. 현재 활동의 80 ms 내 짧은 발화 여부(any)와 미래 bin의 활동 점유율(>50%)도 구분한다. [VAP bin 정의](https://aclanthology.org/2022.sigdial-1.51/)

Hazard 초판의 사건은 **현재 비활동인 각 화자의 다음 speech onset**으로 정의한다. 이는 floor transfer와 동일하지 않다. 이미 활동 중인 화자는 해당 onset 위험집합에서 제외하고, floor transfer·backchannel 구분은 event 층에서 한다. horizon은 2.56초이며 완전히 관측한 무사건 구간은 survival 항에 포함하고, 녹음 종료로 관측하지 못한 bin은 마스킹한다. 현행 `time_to_next_onset`의 한 개 `censored` 값만으로 충분한지 검사하고 관측 종료 시각을 추가한다.

### 6.3 의미·담화 라벨

VAD로 얻은 SHIFT/HOLD/INT/BACKCHANNEL은 약한 라벨이다. ‘잠깐 멈췄다’와 ‘생각이 끝났다’, ‘상대가 시작했다’와 ‘말할 기회를 넘겨줬다’는 같지 않다. 기존 [[output-vap-target-pipeline]]도 유도 SHIFT와 TurnBench EOT가 크게 다름을 기록한다.

자연 대화에서 EN/KO 각 약 500개 경계부터 사람이 `complete/incomplete/uncertain`, floor 관계, backchannel/interrupt를 검토하는 파일럿을 만든다. 불확실한 사례는 강제 분류하지 않는다. 20% 이상을 이중 검토하고 agreement·클래스별 confusion을 확인한 뒤 필요하면 언어당 1–2천 건으로 확장한다. 숫자는 초기 작업 예산 제안이다.

‘현재 시점에서 의미적으로 완결됐는가’는 prefix만 들은 판단, ‘실제로 이후 턴이 바뀌었는가’는 미래를 확인한 사건 라벨로 따로 저장한다. 둘을 같은 정답으로 쓰지 않는다. 기존 코퍼스 문장부호를 정답 EOT로 사용하지 않는다. LLM 자동 라벨은 후속 약한 라벨 후보이며, 사람 검증을 대체하지 않는다. SoulX-Duplug의 상태 예측은 참고하되 현재 프로젝트의 자체 ASR 이력을 쓰는 조건에서 검증한다. [[source-soulx-duplug]]

## 7. 단계별 실험과 중단 기준

각 단계는 직전 통과 체크포인트를 기준으로 한다. 실험 비교는 동일 초기값·데이터·유효 노출량·seed를 사용한다. 다음 수치는 **파일럿용 제안 관문**이며 Q0에서 baseline을 측정한 뒤 장기 run 전에 동결한다. test 결과를 보고 관문을 바꾸지 않는다.

| 단계 | 추가하는 능력 / 데이터 | 주요 산출물 | 다음 단계 진입 |
|---|---|---|---|
| Q0 기준선·데이터 계약 | E2 δ=2/4, 기존 ASR·자연 대화·합성 겹침 평가; 2–5 h QC pack | frozen eval IDs, mixer/serializer round-trip, no-future 검사 | 누락·덮어쓰기·화자 누출 0; 평가 재현 |
| Q1 비중첩 2화자 전사 | EN/KO 균형 약 20–50 h, 실제 시간 보존. 전사만 warm-up → 태그·activity 추가 | A/B 전사·활동, 32개 창 overfit | 아래 ASR guardrail, 비중첩 DER ≤10%·화자 귀속 오류 ≤5% |
| Q2 겹침 전사 | Q1 모델 + 자연 overlap와 제어 합성. 실측 가용 자연 대화 약 100–300 h까지 | 양 화자 전사·고정 ID·밀도/삭제 진단 | 같은 모델의 겹침 학습 전 대비 overlap cp 오류 ≥20% 상대 감소, 한 화자 소실 ≤5% |
| Q3 미래 활동·의미 예측 | 자연 대화만 turn supervision; backbone freeze head probe → 저율 joint | VAP·선택적 hazard/event, semantic ablations | 동일 FPR에서 음향 대조군 대비 검증 가능한 개선, ASR guardrail 유지 |
| Q4 장문·실시간 통합 | 실제 및 합성 10–60분 세션, 자유실행, 잡음/음량차 | 2행 전사+활동+turn UI, 단독 장치 벤치 | 지속 RTF <1, backlog 비발산, ID 지속성·지연 관문 충족 |

32개 창 overfit는 코드 경로 검사다. unseen 화자 일반화의 근거로 쓰지 않는다. 초기 자연 데이터는 화자 다양성·턴 수·겹침 시간을 기준으로 추출하며 단순 폴더 순 20시간을 쓰지 않는다.

### 7.1 학습 recipe 시작점

- 모든 초기화는 E2에서 시작한다. E2 encoder 가중치를 유지하며 새 speaker 행/heads만 초기화한다. ‘새 Qwen’ 재초기화와 비교하지 않는다.
- Q1에서 encoder를 잠시 동결하고 새 출력 규약을 배운 뒤, 정체 시 상위층부터 해동한다. 후보 LR은 thinker `5e-6–1e-5`, adapter `1e-5–5e-5`, 새 heads `1e-4`, encoder 해동 시 `1e-6–5e-6`. 이는 측정 전 탐색 범위다.
- E2의 `next_weight EN/KO=0.3/0.15`, delay 분포, TN을 첫 run에 유지한다. 태그 추가가 NEXT 비율을 바꾸므로 삭제·조기 방출·태그율을 보고 별도 sweep한다.
- Q1/Q2 시작 배치 예산은 대화 70% + 단일 화자 replay 30%의 **오디오 초 기준**으로 제안한다. 대화 내 합성 비율은 최대 약 절반부터 시험하며 자연 대화 표본을 유지한다. Q3의 turn 손실은 자연 대화에만 적용한다.
- Q3는 head-only probe 이후 공유 모델 저율 joint를 비교한다. gradient norm·ASR 회귀를 보고 손실 가중을 정한다. 발화 길이와 corpus 구성 때문에 ‘30 epoch’를 이전 run과 같은 노출량으로 가정하지 않는다.
- free-running prefix에서 head를 학습하는 단계를 포함한다. gold prefix 성능은 oracle upper bound로만 보고한다. 같은 checkpoint로 생성한 rollout은 model hash를 기록하고 모델 변경 후 갱신한다.

### 7.2 장문과 speaker memory

20–40초 독립 창 통과 뒤 60–120초 carry 학습, 10–60분 검증으로 늘린다. 과거에 관측한 음성·전사·speaker state만 carry한다. 오래 침묵한 B가 돌아오는 사례와 처음부터 겹치는 사례를 별도 평가한다.

현재 KV를 무제한 유지하면 메모리가 증가한다. bounded context + 세션 speaker memory를 비교하고, memory는 관측 prefix에서만 갱신한다. 단순 KV 앞부분 삭제가 RoPE·화자 정체성을 보존한다고 가정하지 않는다. context 전환 시 logits·전사·ID 연속성, 메모리 상한을 테스트한다.

## 8. 평가 및 ‘의미를 이해했다’의 판정

### 8.1 네 가지 성능을 함께 보고

| 축 | 주 지표 | 규약 |
|---|---|---|
| 단일 화자 ASR | EN WER, KO CER; S/D/I | E2와 같은 dev/test·TN·δ. corpus별 상대 회귀 ≤5% 제안 |
| 다화자 전사 | EN cpWER/tcpWER, KO 문자 단위 permutation CER; attribution 오류 | 대화 전체 하나의 A/B permutation. overlap/non-overlap·음량차·언어별 분리 |
| Diarization | overlap 포함 DER: miss/FA/confusion, JER, ID switch | collar=0 주 지표, 250ms 보조. 80ms 해상도를 명시 |
| Overlap | 활동 overlap precision/recall/F1, 양 화자 전사 재현율 | 한쪽 발화를 누락하는 실패를 전체 평균에 숨기지 않음 |
| Turn-taking | EOT/INT recall–FPR–latency, BC F1, calibration | official scorer 고정, 임계값 dev 선택, test 고정 |
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
| 시간순 화자 전사 | 신규 `vapasr/data/dialogue_interleave.py` | 두 전사로 완전 복원, Unicode·공백, tie·flush·cap |
| 모델 출력·손실 | `vapasr/hf/modeling_vapasr.py`, config | SPK 차단 조건, audio-position 수집, head logits·mask, E2 호환 |
| activity/VAP/hazard | `vapasr/data/targets.py`의 버전 분기 | 50Hz 참조 경계, 80ms sampling, censor/unknown, permutation |
| 학습·replay | Trainer/data, 신규 `experiments/p2_train_hf.py` | 초 기준 mixture·재개·rollout provenance·DDP 불균등 평가 |
| 평가 | 신규 `experiments/p2_eval.py` | speaker-aware 전사·DER·turn·latency, dev/test 불변 |
| live | `vapasr/hf/live.py`, `live_mlx.py`, `experiments/live/` | A/B별 버퍼·80ms heads·타임스탬프·최초 출력·장문 |

특히 E2 encoder를 **동결한 뒤 저장/재로드해도 E2 가중치가 유지되는지** 검사한다. 현재 저장 로직은 `encoder_trainable`에 따라 encoder를 생략하므로 학습 여부와 저장할 가중치 provenance를 분리해야 한다. frozen E2 대신 원본 `.nemo`를 재부착하면 다른 모델이다.

현재 deferred `<NEXT_AUDIO>` 최적화는 다음 audio와 묶어 forward한다. 새 head는 묶음의 마지막 audio 위치를 명시적으로 읽어야 한다. MLX가 지금처럼 마지막 logits만 돌려주는 경로에는 hidden-state/head 지원이 필요하다. CPU 왕복 때문에 얻었던 속도 이득을 잃지 않는지도 확인한다.

검증 순서는 serializer round-trip → 실제 Dataset 오디오 spot-check → 32창 overfit → HF save/load logits·heads·encoder parity → prefix 절단/미래 교체 인과성 → 단일/불균등 multi-rank smoke → 동일 held-out 자유실행 → live parity → 장문이다. tiny run에서 optimizer/scheduler/RNG·샘플 위치·split hash 재개도 확인한다.

## 10. 실행 예산과 우선순위

1. **첫 묶음, 약 2–4 작업일 제안:** Q0 데이터 계약·평가 pack·기준선, mono mixer와 serializer, 작은 overfit. 이 단계 종료 전 대규모 job을 제출하지 않는다.
2. **둘째 묶음, 약 1주 제안:** Q1 비중첩 화자 전사와 Q2 overlap pilot. 화자 추적·낮은 음량 삭제·생성량 증가를 먼저 해결한다.
3. **셋째 묶음, 약 1–2주 제안:** Q3 audio-only/thinker heads, semantic annotation pilot, 자유실행 history 학습 및 ablation.
4. **넷째 묶음, 약 1주 이상 제안:** Q4 장문·실시간 통합과 고정 test 보고. 사람 라벨링·데이터 접근·선점 대기는 별도다.

이는 연구 개발 순서와 대략적인 인력 일정이지 GPU 완료 시간 예측이 아니다. Q0/Q1의 100–300 step으로 `audio-seconds/GPU-second`, tokens/second, peak memory, 평가 RTF를 실측한다. 이후 `GPU-hours = 총 노출 audio-seconds / 실측 처리량 / 3600`으로 예산을 계산하며 정렬·rollout·eval 비용을 따로 더한다. 최초 pilot은 1–8 GPU로, 통과 후 필요한 규모로 확장한다.

ASR 자체 개선(E3 추가 학습, next_weight, δ 목표 변경)은 E2 기준 별도 run으로 검증한다. 개선 체크포인트를 Phase 2에 도입할 때 Q0 pack을 다시 평가하고 기준 변경을 기록한다. 다화자 학습·인식 recipe·turn loss를 동시에 바꿔 원인을 놓치지 않는다.

## 11. 주요 실패 가설과 다음 조치

| 관측 | 먼저 구분할 원인 | 다음 한 가지 실험 |
|---|---|---|
| 작은 화자 전사가 사라짐 | mixer 결함 / encoder 혼합 정보 부족 / decoder cap | clean-channel oracle와 저음량 합성으로 위치 확인 후 encoder 해동 범위 비교 |
| 전사는 맞는데 A/B가 뒤집힘 | local ID 계약 / 긴 문맥 손실 / speaker 표현 부족 | permutation·carry 검증 후 speaker memory/t-vector 보조 학습 |
| overlap F1은 높지만 전사는 한 사람뿐 | 활동 감지와 음성 내용 분리 능력 차이 | 화자별 D/S/I·전사 recall, multi-output decoder를 제한적 구조 ablation으로 검토 |
| turn head는 gold에서만 좋음 | ASR 지연·오류 이력의 노출 편향 | self-generated history 학습, δ별 turn curve |
| thinker가 audio-only보다 못함 | semantic 표현/라벨 부족 또는 80ms 운율 손실 | 라벨 검수 후 같은 mono의 고해상도 causal acoustic branch ablation |
| 평균 속도는 좋고 장문은 밀림 | burst·KV 증가·speaker token 비용 | context 상한·cap·runtime profiling, 같은 정확도에서 개선 검증 |

고해상도 branch, 분리 보조 학습, multi-output decoder는 최초 주 경로에 넣지 않는다. 관측된 실패 원인을 겨냥한 후속 비교이며 입력은 계속 mono다. Q3에서 의미 기여가 확인되지 않으면 ‘공유 모델로 화자 전사와 활동을 처리했다’까지 주장하고 의미 기반 턴 예측 달성은 보류한다.

## 12. 근거와 남은 불확실성

프로젝트 근거는 [[output-stage2-e2-final-eval]], [[decision-mono-input]], [[output-vapasr-model-and-sequence]], [[source-hf-trainer-migration]], [[output-vap-target-pipeline]]와 이번에 읽은 `vapasr/hf/`, `vapasr/data/streams.py`, `vapasr/data/targets.py`다. 코드 관찰은 2026-09-11 작업 트리 기준이다.

외부 1차 자료 확인일: 2026-09-11. [t-SOT](https://arxiv.org/abs/2202.00842)와 [t-vector](https://arxiv.org/abs/2203.16685)는 직렬화·화자 귀속의 근거이며 E2+LLM의 성능을 보증하지 않는다. [VAP 공식 구현](https://github.com/ErikEkstedt/VoiceActivityProjection)은 현재·미래 활동 공동 학습과 입력 조건을 확인하는 참고다. [MeetEval](https://github.com/fgnt/meeteval)은 다화자 전사 평가, [TurnBench](https://arxiv.org/abs/2608.25218)는 턴 이벤트 평가 근거다.

확인 전 항목: 서버의 현재 대화 데이터 가용량, 대화/화자 누출 감사, 자연 overlap의 정렬 신뢰도, E2 표현의 speaker 보존 정도, mono 조건의 turn 기준선, 실제 2화자 디코드 처리량, 한국어 의미 라벨의 annotator agreement. 이 항목들은 Q0–Q3의 측정 대상이다.
