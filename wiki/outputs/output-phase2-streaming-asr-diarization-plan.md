---
type: output
status: superseded
created: 2026-09-11
updated: 2026-09-15
summary: (2026-09-15 이후 정본은 [[output-phase2-lane-plan]]) 이전 Phase 2 정본 — Q1 C-mode EOT·K슬롯 mono ASR, Q3 미래 예측; 직렬화·스키마·80 ms 정합·평가 원칙·구현 지도·QC 절은 참조로 유효
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
  - '[[decision-multi-speaker-scope]]'
  - '[[question-turn-token-label-reliability]]'
  - '[[output-phase2-data-inventory]]'
  - '[[output-phase2-db-survey]]'
  - '[[source-muse-voice-transcribe]]'
---

# Phase 2 개발 계획: Streaming ASR & Speaker Diarization / Turn-Taking

> **2026-09-15 이후 정본은 [[output-phase2-lane-plan]] 이다**([[decision-phase2-canonical-lane-plan]]). 이 문서는 이전 정본으로, 새 정본이 참조로 지정한 절(§3.2 헤드, §4.2 직렬화 순서, §4.4 무음·flush 규칙, §5.2 스키마, §5.3 파이프라인, §6.2 80 ms 정합, §8 평가 원칙, §9 구현 지도·QC, §10 예산)만 계속 유효하다. K 슬롯 정체성(§4.1)·C-mode EOT(§6.3)·16 사례의 EOT 시각은 새 정본이 대체한다.

개정 기준: 2026-09-14. **Q1의 EOT 토큰은 C 모드, P 모드는 Q3 ablation으로 이월**한다. 최대 2화자 제한 해제·ONSET/EOT 우선·의미 라벨 Stage 3 이월은 유지한다. 이번 변경은 실행 설계이며 모델·라벨러 구현 완료가 아니다. 사용자 8개 비판의 검증과 처리 결과는 [[output-phase2-sequence-critique-response]]에 기록한다.

이 문서가 이전 실행 요약·최종안·turn-token 제안서 및 [[output-phase2-block-and-turn-label-spec]]의 상충 규약보다 우선한다. 특히 contribution당 start/end, EOT에 의한 전사 강제 확정, A/B 고정 블록은 이번 기본 규약이 아니다. 변경 근거는 [[decision-multi-speaker-scope]], [[question-turn-token-label-reliability]], 데이터 실측은 [[output-phase2-data-inventory]]·[[output-phase2-db-survey]]다.

**규약은 이 정본에만 둔다.** probe/replay 문서와 코드는 당시 P-mode 실험 기록이며 현재 기본 규칙을 정의하지 않는다. 검수 문서는 정본을 실행하는 작업 안내다. 공용 serializer 구현 후 정본 fixture로 일치성을 검사한다.

## 1. 목표와 범위

하나의 마이크에서 들어오는 16 kHz mono 스트림을 받아, **여러 화자의 말을 실시간으로 전사·귀속하고, 화자별 발화 시작과 종료 가능성 및 미래 활동을 예측**한다. 겹치는 각 화자의 전사가 남아야 한다. Phase 2는 음향·행동 기반 라벨로 학습 가능성을 검증하고, 의미적 완결성의 직접 감독은 Stage 3으로 미룬다.

| 요구 | 모델이 제공할 결과 | 검증 질문 |
|---|---|---|
| 실시간 ASR 강화 | 누적 lexical 전사, 방출 시각, 안정적인 처리 지연 | E2 정확도를 유지하면서 실제 입력 속도를 따라가는가? |
| Speaker diarization | 세션 내 `<SPK_1..K>`, 슬롯별 활동 | 새 화자와 재등장 화자를 구별하는가? |
| Overlap 인식 | K개 활동 확률과 2명 이상 동시 활동 | 동시 발화 수가 늘어도 놓치지 않는가? |
| Overlap 전사 | 각 슬롯의 전사와 시간 정보 | 작은 목소리를 누락하거나 큰 목소리를 복제하지 않는가? |
| Turn-taking | 화자별 `<ONSET>`·`<EOT>`, 미래 활동·onset hazard | 자체 전사 이력으로 음향 대조군보다 행동 예측이 개선되는가? |

**Diarization과 turn-taking은 서로 다른 목표**다. 현재 화자 변경이나 자동 유도 EOT를 잘 맞힌 것만으로 의미 이해를 입증하지 않는다. 맞장구도 전사·활동에는 포함하지만 `<BC>` 분류나 의미적 floor 판정은 이번 필수 출력에서 제외한다.

범위는 EN/KO, **0–K화자**다. K는 세션 내 누적 식별 화자 슬롯 상한이며 동시 발화 수와 다르다. 초기 K=4/8 비교는 제안이고, Q0에서 실제 포함 코퍼스의 세션 화자 수·비용을 조사해 동결한다. ICSI처럼 8명을 넘는 자료를 포함하면 그 세션을 수용하도록 K를 늘리거나 별도 확장 실험으로 명시한다. ‘2화자 제한 해제’를 ‘무제한 화자 지원 완료’로 표현하지 않는다.

회의를 2명만 활동하는 창으로 골라 나머지 화자를 버리지 않는다. 메모리용 chunk/crop은 허용하되 전체 시간축·화자·carry를 보존한다. 실명 인식·세션 간 인증, 파형 분리 출력, 번역·응답/TTS, 대소문자·구두점 복원은 필수 산출물이 아니다. `<HOLD>`·`<BC>`·의미 완결성 라벨과 의미 전용 head는 Stage 3 범위다.

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

추가된 표본 test(s300/u1000)의 E2 δ=4는 test-clean/other WER 5.4/10.7%, Kspon eval_clean/other CER 12.5/13.6%다. C2는 각각 5.8/12.3%, 18.7/19.6%. 전체 test 완료나 다화자 성능의 근거로 확대하지 않는다. [[output-phase1-report]]

## 3. 권장 모델 구조

```text
16 kHz mono 혼합 음성
        ↓
E2 Nemotron streaming encoder [56,0] → adapter
        ↓
Qwen3-ASR thinker: [AUDIO_k] → 화자 태그·ONSET/전사/EOT → <NEXT_AUDIO>
        ├─ 생성 경로: <SPK_1..K> + lexical / <ONSET> / <EOT>
        │             → K개 전사 버퍼 + 화자별 예측 이벤트
        └─ h_audio[k]: 현재 오디오 위치의 causal hidden state
              ├─ 현재 활동: K sigmoid
              ├─ 미래 활동: K × 4 sigmoid (factorized 제안)
              └─ 후속 단계: 슬롯별 onset hazard
```

분리 채널은 라벨·정렬·훈련용 혼합 생성에만 사용한다. 추론에는 mono만 제공하고, 정답 VAD·정답 화자·깨끗한 채널·완성 전사·전체 파일 통계를 넣지 않는다. 이는 [[decision-mono-input]]을 유지한다. 화자별 활동 헤드는 별도 외부 diarization 시스템이 아닌 공유 모델의 보조 출력이다.

### 3.1 왜 이 구성을 먼저 시험하는가

기존 thinker와 80 ms 시간축, `<NEXT_AUDIO>` 규약을 재사용한다. 겹침 전사는 한 decoder에서 시간순으로 직렬화한다. 다화자 토큰을 단일 출력 경로에 직렬화하는 선행 근거는 [t-SOT](https://arxiv.org/abs/2202.00842)다. 원 논문의 virtual channel과 세션 내 고정 화자 ID는 다르므로 K화자 추적 능력을 별도로 검증한다.

오디오마다 예측하는 활동 헤드가 있어야 텍스트가 없는 무음·호흡·겹침에서도 상태를 낼 수 있다. 기존 `<SPK_A/B>`는 현재 HF 디코드에서 차단되어 있고 활동·turn 헤드도 없다. 예약 토큰이 있다는 이유로 기능 구현이 완료된 것은 아니다.

Muse에서 참고하는 것은 전사·화자·endpoint 이벤트의 공동 생성이다. `<ONSET>`·`<EOT>`라는 이름과 아래 행동 라벨·K슬롯 규약은 우리 설계이며 Muse의 공개 규약과 동일하다고 주장하지 않는다. [[source-muse-voice-transcribe]]

### 3.2 Audio-clock 헤드와 의미 정보

`h_audio[k]`는 `[AUDIO_k]`를 처리한 직후, **같은 청크의 전사를 생성하기 전** 상태로 고정한다. 현재까지의 오디오와 이전 청크까지 방출한 텍스트를 볼 수 있다. 현재 청크의 gold 토큰이나 미래 발화를 attention으로 볼 수 없게 한다. 활동·turn 출력은 텍스트 생성 여부와 무관하게 80 ms마다 계산한다.

이 선택은 turn 출력이 ASR flush를 기다리는 문제를 줄이지만, 최근 확정 전사는 δ만큼 늦게 들어온다. δ=2/4별 turn 성능도 비교해야 한다. 같은 청크 전사를 본 뒤의 헤드는 후속 ablation으로만 추가하고 추가 생성 시간이 포함된 가용 시각으로 채점한다.

초기 헤드는 작은 MLP로 두고 E2 표현의 유효성을 확인한다. 활동은 K개 sigmoid로 동시 0–K명 발화를 허용한다. overlap은 현재 활동 기준으로 평가한다. 독립 marginal로 만든 동시 활동 확률을 정확한 공동 분포로 부르지 않고 필요하면 별도 overlap head를 비교한다.

미래 활동은 슬롯별 4-bin BCE를 주 후보로 한다. 이는 원 VAP 256-class의 공동 분포가 아니며 동일 모델명·동일 calibration으로 취급하지 않는다. ‘나/나머지’ 집계는 다자 경쟁 정보를 잃는 보조 비교다. 원 256-class 대조는 dyadic 평가에 한정한다.

장기 ID 유지가 약하면 동일 모델 내부에 causal speaker memory 또는 token-level speaker embedding 보조 손실을 추가한다. [t-vector 연구](https://arxiv.org/abs/2203.16685)는 겹침 전사 토큰에 화자 표현을 대응시키는 선행 근거다. 이 확장은 최초 run에 한꺼번에 넣지 않는다.

## 4. 화자·전사·시간 계약

### 4.1 K슬롯 정체성과 신규 화자

- 최초로 식별 가능한 화자부터 빈 슬롯 1, 2, …를 배정하고 세션 중 유지한다. 슬롯을 성별·채널 번호·큰 목소리와 연결하지 않는다. 모델이 기존 슬롯 또는 다음 빈 슬롯을 생성하며 정답 화자 수를 추론 입력으로 주지 않는다.
- 시작부터 겹쳐 식별이 모호하면 임시 슬롯으로 시작하고 confidence를 기록한다. 주 지표는 최초 출력 기준이다. 나중에 ID를 고쳤다면 수정 횟수·확정 지연도 보고한다.
- 학습의 channel→speaker 매핑, 전사 태그, 활동·VAP·hazard 라벨은 **같은 permutation**을 사용한다. 무작위 channel 순서와 gain을 바꾸어 지름길 학습을 검사한다.
- 독립 crop은 관측 prefix 내 최초 화자 기준으로 일관되게 재매핑한다. carry 학습·장문 평가는 세션 ID를 유지한다. 미래 단독 구간이나 전체 파일 clustering 결과로 온라인 슬롯을 정하지 않는다.
- PIT를 쓰는 실험도 permutation은 crop/세션 단위로 고정하며 청크마다 최적 permutation을 바꾸지 않는다. 전사 손실과 활동 손실을 따로 permutation하면 ID 의미가 충돌한다.
- 미등장 슬롯은 ‘나중에 나타날 특정 gold 화자’가 아니다. 그 화자가 식별되기 전의 identity-specific 미래 활동/hazard 손실은 마스크한다. 신규 화자의 onset은 빈 슬롯 생성 경로로 검출하고, 그 이전의 신규 화자 미래 예측은 별도 `any-new-speaker` head 후속 후보로 둔다. 식별 전 현재 활동은 임시 슬롯/식별 지연으로 계수하며 gold-slot teacher 결과와 자유실행을 분리한다.
- K개 슬롯이 모두 할당되면 API에 `capacity_exhausted`를 알리고 오래 쉰 슬롯을 자동 재활용하지 않는다. 이는 ‘K+1번째 화자를 정확히 검출했다’는 뜻이 아니다. 실제 초과는 gold 화자 수로 별도 평가하고, 온라인 신규 화자 거부/unknown 처리는 novelty 검출을 추가 검증해야 하는 미완 항목으로 둔다. 초과 세션의 강제 오귀속을 평가 분모에서 빼지 않는다. 학습 corpus의 모든 화자를 담을 K 선택이 Q0 진입 조건이다.

### 4.2 겹침 전사의 직렬화

**Q0 token registry 결정:** 슬롯 1/2는 기존 `<SPK_A>`/`<SPK_B>` ID를 재사용한다. 이 문서의 `<SPK_1>`/`<SPK_2>`는 논리 슬롯 표기이며 새 tokenizer 문자열이 아니다. 슬롯 3..K만 `<SPK_3..K>`, 이벤트는 `<ONSET>`·`<EOT>`를 추가한다. raw 텍스트 별칭을 중복 등록하지 않고 `slot_to_token_id`로 매핑한다. 기존 probe의 `<SPK_1..4>` 임시 ID와 혼용하지 않는다.

2026-09-14 E2 MLX export 실측은 `<SPK_A>=151707`, `<SPK_B>=151708`, tokenizer ID 0..151716, embedding `[151936,1024]`다. 코드상 출력 head는 tied이며 MLX 파일에는 독립 lm_head 텐서가 없다. **219개 미등록 행은 용량 여유일 뿐 미사용/안전 초기화 보증이 아니다.** HF 원 체크포인트의 config·input/output 행 수·weight tying을 재확인한 뒤 새 ID 행만 고정 seed로 초기화한다. 기존 A/B 행은 재초기화하지 않는다. padding·비등록 ID는 학습 softmax/생성에서 마스킹하고, A/B의 기존 decode 차단은 Phase 2에서만 해제한다. `len(tokenizer)`로 embedding을 151936행 미만으로 축소하지 않는다. save/load·HF/MLX ID/logit parity가 resize 생략의 관문이다. 실제 migration은 아직 미구현이다.

각 화자의 lexical 전사를 독립적으로 tokenize·정렬한 뒤 토큰 종료 시각으로 합친다. 동일 시각은 슬롯 번호 및 원래 토큰 순서로 결정하고, 화자 내부 순서를 보존한다. 기본 청크 배정은 E2의 `k=floor(t_end/0.08)+δ_text`를 유지한다. ONSET/EOT를 합치는 우선순위와 시각은 §4.4·§6.3을 따른다.

```text
[AUDIO_k] <SPK_1> 그때 <SPK_2> 응 <SPK_1> 내가 <NEXT_AUDIO>
[AUDIO_k+1] 갔거든 <SPK_3> 맞아 <NEXT_AUDIO>
```

예시는 각 화자가 이미 등장한 뒤의 lexical 직렬화 형식이다. 오디오에는 실제 겹침이 남고 출력 문자열만 번갈아 기록한다. 청크 경계에서도 현재 전사 화자를 유지한다. 최초 lexical 출력과 화자 변경 때 태그를 낸다. 태그는 다음 오디오의 현재 화자가 아니라 **뒤따르는 lexical/event payload의 귀속**이다. 빈 태그 반복은 금지하지만 `<SPK_2><EOT>` 같은 이벤트만 있는 블록은 허용한다.

BPE byte 조각 사이에 다른 화자의 토큰을 끼워 넣으면 화면 텍스트가 깨질 수 있다. 같은 정렬 단위의 토큰 묶음은 원자적으로 유지하고 화자별 decoder buffer로 Unicode/공백을 복원한다. 단어 전체 완료까지 기다리는 변형은 추가 지연을 따로 잰다. TN은 현재 lexical 규약과 지문을 유지하고 화자 태그를 문자열 정규화에 섞지 않는다.

학습 경로는 `asr-tn-v1.3.0`의 `target(text, lang, corpus)` → `target_flags` 격리 → tokenization → alignment 순서를 사용한다. KO는 `corpus=aihub71631`, EN은 실제 corpus key를 전달한다. `target_en`만 호출하고 숫자 quarantine을 생략하면 불완전하다. AMI는 기존 숫자 허용 corpus가 아니므로 digit 잔존 행을 격리한다. TN 코드·숫자 backend·tokenizer/registry 지문을 저장하며 fallback `norm_word`는 진단 재현에만 허용한다.

K화자 전사·태그·이벤트로 청크당 생성량이 증가한다. 안전 상한을 단순히 K배로 올리지 않고 실제 밀도 p99·강제 NEXT·삭제율·처리 지연을 측정한다. lexical·selector·event 및 전체 예산을 따로 기록한다. selector를 내기 전 최소 payload까지 2자리 여유를 확보해 `<SPK_s><NEXT_AUDIO>`를 만들지 않는다. cap 초과는 bounded backlog로 이월하며 청크를 버리거나 EOT로 대체하지 않는다. 학습·추론의 cap 차이도 QC한다.

기본은 **시각순 교차 직렬화**다. 청크 안 슬롯별 묶음 출력은 태그 감소 ablation일 뿐 기본이 아니다. 비교 시 뒤 슬롯의 계산 지연·삭제·cap 편향을 함께 측정한다.

공용 `dialogue_interleave.py`의 순서 계약은 먼저 화자 내부 lexical 순서·원자 payload·ONSET/lexical/EOT 의존성을 만족시킨 뒤, ready 항목을 `(target_chunk, reference_sample, slot, kind_priority, source_ordinal)`로 안정 정렬하는 것이다. 완전 동률의 `kind_priority`는 ONSET < lexical < EOT다. 이 tie-break만으로 서로 다른 reference 시각의 의존성이 해결된다고 가정하지 않는다. pending 이월 뒤에도 의존성을 재검사한다. probe의 정렬 코드를 그대로 정본 구현으로 승격하지 않는다.

cap은 아직 미동결이다. Q0의 KO 실제 데이터와 EN 회의에서 전체/출력있는 청크 각각의 lexical·selector·event·총량 p50/p95/p99/max를 측정한다. 동시 0/1/2/3+화자별, EN/KO별 분리하고 자연 3+ 표본이 없으면 미측정으로 남긴다. 합성 stress 결과는 자연 회의 결과와 합치지 않는다. cap 후보마다 강제 NEXT·이월·화자별 삭제·tick을 함께 비교하며 AMI 38.4초 최대 3토큰은 cap=8의 근거가 아니다.

### 4.3 출력 API와 세 개의 시각

전사 이벤트는 `session_id, speaker_id, token_ids/text, chunk_index, audio_seen_until, emitted_at, confidence`를 가진다. 활동·turn 이벤트는 `activity[K], future_activity[K,4], onset_hazard[K,...], event_type, speaker_id, emission_source`를 제공한다. `emission_source=model/timeout_policy/hazard_policy`를 구분하고 모델-only 결과를 기본 보고한다. 현재 없는 필드는 구현 단계에 맞춰 추가한다.

`audio_seen_until`은 실제 읽은 마지막 오디오 시각, `emitted_at`은 실제 결과 가용 시각이다. `estimated_speech_start/end`를 제공할 경우 별도 시간 추정 head 또는 causal aligner가 필요하며 추정값임을 표시한다. `방출 시각−δ`를 정확한 발화 경계로 간주하지 않는다. 초판은 활동 구간과 전사 방출 시각을 제공하고, tcpWER에 사용할 단어/구간 타임스탬프 생성 규약을 평가기에서 고정한다.

### 4.4 무음·단독·겹침의 블록 계약

매 80ms 실제 PCM은 무음이어도 `[AUDIO_k]` 하나를 만든다. `[AUDIO_k] payload* <NEXT_AUDIO>`가 기본이며 payload가 없는 슬롯의 빈 블록은 만들지 않는다. `<NEXT_AUDIO>`는 청크 라운드 종료이지 발화/턴 종료가 아니다. `<EMPTY_AUDIO>`는 실제 EOF 이후 flush용이지 무음용이 아니다.

`<ONSET>`은 음향 발화 구간의 시작, Q1의 `<EOT>`는 **관측한 행동 조건에 근거한 종료 판단(C 모드)**이다. **ONSET 한 번당 EOT 한 번인 괄호 문법이 아니다.** 같은 화자가 pause 뒤 재개하면 EOT 없이 ONSET이 반복될 수 있고, 맞장구에도 ONSET·전사는 있지만 EOT는 생략될 수 있다. 모든 ONSET/EOT 앞에 `<SPK_s>`를 명시한다. 아래 1/2/3은 논리 슬롯 표기이며 C 모드의 ready 시각은 §6.3을 따른다.

| 입력/상황 | 학습 출력 예시 | 활동·상태 해석 |
|---|---|---|
| 시작 전 무음, pending 없음 | `[AUDIO_k] <NEXT_AUDIO>` | 관측 가능한 activity 전부 0; 빈 화자 태그 없음 |
| 1만 시작, 아직 단어 없음 | `[AUDIO_k] <SPK_1><ONSET> <NEXT_AUDIO>` | speech onset도 생성 타깃 |
| 1만 계속 말함 | `[AUDIO_k] <SPK_1>안녕하세요 <NEXT_AUDIO>` | 현재 selector가 1이면 lexical 앞 태그 생략 가능 |
| 1은 말하지만 완료된 토큰 없음 | `[AUDIO_k] <NEXT_AUDIO>` | activity 1과 NEXT는 모순 아님 |
| 1의 자연 pause | `[AUDIO_k] <NEXT_AUDIO>` | activity 0; EOT를 무조건 붙이지 않음 |
| pause 뒤 1 재개 | `[AUDIO_k] <SPK_1><ONSET> 그리고 <NEXT_AUDIO>` | gap ≥0.25s일 때 새 음향 구간; 이전 EOT 필수 아님 |
| 무음 중 지연 전사 도착 | `[AUDIO_k] <SPK_1>감사합니다 <NEXT_AUDIO>` | 현재 activity와 전사 귀속을 분리 |
| 1 종료 판단 ready, 전사 없음 | `[AUDIO_k] <SPK_1><EOT> <NEXT_AUDIO>` | C 증거 가용 이후 이벤트-only 블록 허용 |
| 1 마지막 전사와 종료 판단 ready | `[AUDIO_k] <SPK_1>이상입니다 <SPK_1><EOT> <NEXT_AUDIO>` | C 증거 가용 이후 해당 경계의 마지막 lexical 뒤 EOT |
| 1 계속 + 2 맞장구 | `[AUDIO_k] <SPK_1>설명하면 <SPK_2><ONSET> 응 <NEXT_AUDIO>` | 두 활동이 1; 1의 EOT를 강제하지 않음; BC 토큰 없음 |
| 2의 짧은 맞장구 종료, 1 계속 | `[AUDIO_k] <SPK_1>이렇게 <NEXT_AUDIO>` | 2의 activity만 0; 휴리스틱상 EOT 없음 |
| 2 발화 중 1의 지연 EOT | `[AUDIO_k] <SPK_1><EOT> <SPK_2>네 <NEXT_AUDIO>` | 1의 이벤트가 2를 닫지 않음 |
| 1·2·3 동시 발화 | `[AUDIO_k] <SPK_1>저는 <SPK_3><ONSET> 잠깐 <SPK_2>동의해요 <NEXT_AUDIO>` | 시간순으로 직렬화; 슬롯 수만큼 audio를 복제하지 않음 |
| 겹치지만 pending 없음 | `[AUDIO_k] <NEXT_AUDIO>` | activity는 복수 1이어도 됨 |
| 두 종료 판단이 함께 due | `[AUDIO_k] <SPK_1><EOT> <SPK_2><EOT> <NEXT_AUDIO>` | 각각 C 증거 확인·독립 귀속; global EOT 없음 |
| crop/전송 종료 | bounded flush 후 stream-closed API | crop/EOF 자체를 EOT 정답으로 만들지 않음 |

테스트에서 위 16행을 표 순서대로 `case_01`…`case_16`으로 고정한다. C의 실제 ready 시각, 논리 슬롯→실제 ID, next-token shift와 loss mask까지 assert한다. 문자열 스냅샷만 맞는 검사를 통과로 보지 않는다. 실제 EOF와 중간 crop의 차이는 아래 규약을 따른다.

표는 pending 순서가 성립하는 예시이지 VAD→문자열의 고정 변환기가 아니다. 모델은 같은 청크에서 NEXT/lexical/ONSET/EOT 중 무엇을 낼지 CE로 학습한다. activity 헤드는 별도의 조밀한 감독을 받는다.

parser 상태는 `current_selector`, 슬롯별 `last_onset_seq`, `last_eot_seq`, 누적 전사·confidence다. activity는 별도 추정이다. EOT는 슬롯 삭제나 전사 hard-close를 유발하지 않는다. 새 ONSET 없이 지연 lexical이 EOT 뒤에 나와도 보존하고 프로토콜 지연 오류로 계수한다. 추론 grammar는 자체 생성 상태만 쓰며 gold VAD·남은 정답 단어·미래 경계를 참조하지 않는다. 한 onset episode의 중복 EOT는 억제하되 이전 EOT 없이 새 ONSET을 허용한다.

실제 EOF에서만 bounded `<EMPTY_AUDIO>` flush를 수행한다. 이미 관측한 음성의 잔여 전사/이벤트를 처리하되 zero padding을 ‘3초 실제 무음’ 증거로 쓰지 않는다. 미완료는 API `truncated=true`로 남긴다. mid-speech crop은 과거 warm prefix/carry를 제공하거나 event 학습에서 제외하고, 시작점에 가짜 ONSET을 만들지 않는다.

## 5. 데이터 계획

### 5.1 자연 대화와 합성 겹침의 역할

| 데이터 | 첫 사용 | 주의 |
|---|---|---|
| otoSpeech EN | 자연 대화 전사·화자·미래 활동 | 420대화 104.9 h 보유, 학습 사용 결정. actor 단위 split 후 train 약 90–95 h 예상; 라이선스 원문 확인 |
| AI Hub 71631 TS_01 실내 | KO 자연 대화·겹침 | 757대화 196.6 h, E2 학습 노출. Phase 2 train 전용 |
| AI Hub VS_02 실외 | KO dev/보고 | 186대화 51.7 h, E2 dev 재사용; untouched test 아님 |
| TurnBench dev | dyadic EN 이벤트 평가 전용 | 38대화 7.3 h. 학습 제외, 공식 scorer와 입력 조건 고정 |
| NIA24 134-1 성인 실외 | KO 추가 복원 우선 후보 | 1,492대화 완전 사본, 약 360 h는 추정. 채널 기원 조각·시간축·중복·split QC 후 편입 |
| 134-1 실내 부분 / 134-2 청소년 | 부분 감독 / 확장 후보 | 실내 결손 많음. 청소년 채널 기원·완전성 미검증; 모두 완전한 대화로 취급하지 않음 |
| NIA23 002_Meeting | KO 다화자 회의 후보 | 혼합 mono 조각·화자·시각 있음. 총 시간·복원 완전성·중첩 전사 누락 확인 필요 |
| AMI, ICSI, CHiME-6, NOTSOFAR-1, DiPCo | EN 다화자 회의 후보 | 2화자 창 제한 없이 사용 대상. AMI도 현재는 발화 사본이므로 원본 시간축 필요; 나머지 확보·라이선스 검증 전 |
| Switchboard / CANDOR | EN 확장 후보 | Switchboard 약 230 h 복원 후보; 원 gap 음향 없음. CANDOR는 미확보, 확보 시 사용 |
| LibriSpeech/Kspon/NIKL train 혼합 | 합성 2–4화자 전사·활동 | 임의 결합은 자연 turn-taking 정답이 아님 |
| Phase 1 EN/KO 데이터 | 단일 화자 ASR replay | 약 5,555 h 풀. E2와 같은 TN·split; 신규 대화 시간과 중복 합산 금지 |

규모·확보 상태는 2026-09-11 저장된 서버 조사 [[output-phase2-data-inventory]]·[[output-phase2-db-survey]] 기준이다. 이번 정본 개정에서 서버를 다시 측정한 것은 아니다. 현재 기본 학습 대화는 약 290 h(split 전 추정), 134-1 실외 약 360 h는 **추가 후보**이며 전처리 완료 시간이 아니다. 채널 기원 상호상관 검증도 성인 실내 2대화·12발화 표본이다. 전체 성인·청소년 검증으로 확대 해석하지 않는다.

코퍼스별 대화 경과시간, 화자별 speech-hours, 혼합 후 시간, gap 재구성 비율, 유효 감독 시간을 따로 집계한다. K개 채널 시간을 더해 대화 시간으로 부르지 않는다. 71631 원본·134-1 사본·E2 crop은 원 대화 ID로 중복 제거한다. 기존 51.7 h dev는 학습 총량에서 제외한다. 새 사본도 자동으로 untouched가 아니며 E2 exposure 감사가 선행한다. 라이선스는 실제 사용 목적의 원문을 Q0에서 확인한다.

자연 대화는 원래 gap·중첩·맞장구·발화 순서를 보존한다. Q1 비중첩 warm-up은 curriculum일 뿐 회의 데이터 전체를 dyadic 창으로 제한하는 정책이 아니다. 음성이 잘린 조각을 복원한 gap은 시각만 보존한 인공 무음이며 자연 무음 음향과 구분한다. 합성 overlap/replay에는 lexical·신뢰 가능한 activity를 학습하고, 초기에는 ONSET/EOT·미래 활동·hazard를 마스킹한다(negative 억제 방지는 §6.1).

합성 스트림은 20–40초부터 시작하고 다음 조건을 분리한다: 무겹침, 짧은 맞물림, 10–30% overlap, 30–50% stress, 같은 시각 시작, 한쪽 음량이 3/6/12 dB 낮음, 비슷한 음색. overlap 비율 분모는 ‘적어도 한 명이 말하는 시간’으로 고정한다. 모든 합성 원본은 train split에 속해야 한다.

### 5.2 스키마와 로더

기존 `vapasr/data/streams.py::assemble_stream`은 `out[o:o+n] = x[:n]` 대입이라 겹치는 입력을 합산하지 않는다. **Phase 2 전용 대화 mixer**를 만들고 다음을 계약화한다.

- 입력 파형은 자연 mono 원음 우선, 없으면 동기화된 화자 채널을 고정 규칙으로 합산한다. 동일 혼합 신호의 잘린 사본을 여러 화자 오디오인 것처럼 더하지 않는다. clipping·반대 위상·누설·샘플레이트·offset을 검사한다. 원거리 mic와 headset 라벨 간 시간 정합도 검증한다. 추론 정규화는 미래 파일 통계에 의존하지 않는다.
- `conversation_id, source_speaker_id, local_speaker_slot, source_channel, src_offset_s, mix_offset_s, duration, gain, split, alignment_quality, task_masks`를 명시한다. 기존 스키마를 묵시적으로 바꾸지 않고 별도 버전과 변환기를 둔다.
- A등급(독립 채널/채널 기원 조각)은 채널별 forced alignment 뒤 혼합 시간축으로 옮긴다. B등급(혼합 mono)은 비중첩·신뢰 가능한 발화부터 정렬하며 겹침 정렬을 gold timing으로 쓰지 않는다. 해당 구간은 재검수하거나 timing 손실/평가에서 제외한다. 발화 시각 라벨만으로 모든 겹침 화자의 VAD·전사가 완비됐다고 가정하지 않는다. C등급(대화 시각 없음)은 복원 대상에서 제외한다.
- 원 대화 길이의 VAD를 먼저 만들고 crop별로 참조한다. 미래 활동 2.0초·hazard 2.56초·EOT 휴리스틱 최대 3.0초 등 **목표별 horizon**과 관측 종료 시각을 저장한다. 미래는 정답 산출에만 쓰고 모델 입력·슬롯 배정에는 넣지 않는다. 녹음 끝·주석 누락은 censor/mask한다.
- 화자별 채널 매핑을 파일마다 검증한다. 기존 AI Hub 기록에 channel swap이 있었고 발화 EndTime은 정밀 VAD가 아니다. 에너지 VAD도 검수된 정답으로 취급하지 않는다.
- E2까지의 학습 데이터와 화자·대화·오디오 중복을 감사한다. pretraining 노출, Phase 2 train, dev, untouched test를 구분한다. crop을 만든 뒤 무작위 split하지 않는다.

첫 QC는 EN/KO 각 50개 대화 창의 파형·채널·혼합·전사·정렬·활동 라벨을 실제 학습 Dataset 객체를 통해 점검한다. 마지막 1바이트 PCM, EOF, archive channel 선택, 원본 offset, 무음 화자, 전사 없는 speech, 화자 교체, 채널 순서 반전도 회귀 사례에 넣는다.

부분 사본에는 `audio_observed_mask, annotation_complete_mask, gap_reconstructed_mask, label_horizon_mask`를 추가한다. **오디오 없는 정답 발화를 zero padding 위에서 학습시키지 않는다.** 초기에는 완전 관측 창만 joint에 쓰고, 결손 인접 문맥·미래 horizon까지 turn 감독에서 제외한다. 한 화자 조각 누락으로 혼합 자체가 바뀌므로 해당 시간 전체를 불완전으로 표시한다. 부분 창은 보수적인 ASR-only 별도 실험이며 라벨 하나만 마스크했다고 정상 대화가 되지 않는다.

### 5.3 라벨·블록 생성 파이프라인

`원 대화 split·중복 감사 → 복원/mono mix + 관측 mask → 화자별 VAD·정렬 → 슬롯 매핑 → ONSET/EOT 후보·horizon mask → 시간순 serializer → 실제 Dataset QC` 순으로 수행한다.

중간 레코드는 `conversation_id, source_speaker_id, slot, segment_id, event_type, boundary_ref_s, target_chunk, label_observed_until_s, label_source, confidence, valid_mask`와 token alignment를 갖는다. `label_observed_until_s`는 **정답을 만들 때 본 미래의 끝**이고 모델의 `audio_seen_until`과 다르다. TN/tokenizer·VAD·gap·event-rule·K·split·audio hash를 manifest 버전에 고정한다.

**바로 다음 실물 Q0는 71631 성인 실외 대화 1개**다. 데이터 식별자는 분명히 한다: 71631 `VS_02` 186개는 E2가 사용한 dev stereo 원본, 134-1 `TS_02` 1,492개는 같은 성인 자원의 조각 후보 풀이다. 우선 원본과 대응 조각을 모두 확인할 수 있는 실외 대화를 원 ID로 join해 2채널 복원을 대조한다. 대응 조각이 없으면 그 사실을 보고하고 원본 경로와 다른 조각 대화 검사를 분리하며 대응을 지어내지 않는다. [[output-phase2-training-db]]

순서는 `raw Text + 원 대화 offset → target_ko(aihub71631)·flags → 화자 채널 기원 조각 ForcedAligner → 2채널 시간축 복원 → 관측 mask·에너지 VAD 50Hz → mono mix → registry·공용 serializer → 실제 Dataset`이다. 내부 문맥과 미래 3초 이상을 확보한 **38.4초(480블록)** 창을 선택한다. 숫자·Latin·겹침·재개·무음 사례를 기록하고 모든 사례가 한 창에 있다고 가정하지 않는다. 조각 사이 인공 zero는 자연 무음 gold가 아니며 원본과의 비교가 없으면 그 구간과 영향을 받는 horizon을 제외한다.

필수 산출물은 PCM·전사/정렬·VAD와 `activity[480,K]`, `future_activity[480,K,4]`, hazard bin/risk/censor 및 각 mask다. target 생성과 head 학습 구현은 별개이며 Q0에서 target만 검사할 수 있다. 의미 라벨 대신 원 오디오·offset·VAD/정렬 오차를 검수한다. 이 대화는 진단 노출로 기록하고 학습/untouched test에 넣지 않는다. 실제 Dataset 경로가 없으면 실물 JSON 성공만 보고하며 Q0 완료를 선언하지 않는다.

처음에는 완전한 자연 대화의 event-positive/negative 창으로 만든다. noisy 자동 라벨의 일치도·coverage·EOT/min·ONSET/min·빈 슬롯 오류·이월 토큰을 dyadic/회의·EN/KO별로 보고하고 32창 overfit 전에 §4.4 사례를 round-trip한다. 새로운 학습 라벨을 gold로 부르지 않는다.

## 6. Turn-taking 라벨과 학습 목표

### 6.1 먼저 학습할 목표

`L = L_AR(lexical, SPK, NEXT, ONSET, EOT) + λ_activity L_activity + λ_future L_future_activity + λ_hazard L_hazard`

AR은 기존 weighted CE의 분모를 보존하고 token 종류별 합·개수·gradient를 기록한다. 초기 가중 제안은 lexical/SPK/ONSET=1, EOT=2, NEXT EN/KO=0.3/0.15다. head 손실은 유효 슬롯·프레임/bin으로 별도 정규화한다. Q1에서 ONSET/EOT 생성까지 검증하고, Q3에서 미래 활동·hazard를 추가한다. 의미 전용 손실은 Phase 2에 없다.

학습은 causal next-token shift다. `[AUDIO_k]<SPK_1><EOT><NEXT_AUDIO>`에서 audio hidden state가 SPK, SPK가 EOT, EOT가 NEXT를 맞힌다. `[AUDIO_k]<NEXT_AUDIO>`는 audio 위치의 NEXT CE와 activity BCE를 동시에 받는다. audio/EMPTY 자체를 예측하지 않되 **audio 위치가 다음 출력 토큰을 예측하는 손실까지 지우지 않는다.** 완전 감독된 pause의 NEXT는 조기 EOT에 대한 negative다.

replay/합성의 ‘EOT 라벨 없음’을 ‘EOT가 절대 없어야 함’으로 학습하지 않는다. 초기 구현 제안은 ASR-only batch에서 ONSET/EOT logits를 AR softmax 분모에서 제외하고, event-complete 자연 batch에서는 전체 유효 vocabulary CE를 사용하는 보조 목적이다. 단순 event label `-100`만으로는 NEXT 위치의 억제가 사라지지 않는다. 이 방식은 event 삽입을 주변화한 완전 likelihood와 다르므로 혼합률·자유실행 calibration을 검사한다. 배포 decoder에는 gold 데이터 종류별 event mask를 주지 않는다.

미래 음성 활동을 학습해 턴 이벤트를 유도하는 근거는 [VAP](https://arxiv.org/abs/2205.09812)다. 정답 현재 VAD를 모델 입력에 제공하는 실험은 oracle로만 구분한다. 배포 조건의 VAD는 모델이 mono에서 예측해야 한다.

### 6.2 80 ms 라벨 정합

원 VAP 구간은 현재 시각 이후 `[0,.2], [.2,.6], [.6,1.2], [1.2,2.0] s`다. **Phase 2는 50 Hz 기준 VAD에서 경계를 계산하고 80 ms마다 샘플링해 K×4 이진 target을 만든다.** 2화자 원 VAP 공동 분포는 256개 조합이지만 K화자에 `2^(4K)` 분류를 사용하지 않는다. 모델 출력률과 target 원천 해상도는 다를 수 있다.

현재 `targets.py`는 12.5 Hz일 때 경계를 `.16/.56/1.2/2.0 s`로 근사하고 any-pool을 쓴다. 기존 라벨을 조용히 재사용하지 말고 새 라벨 버전을 기록한다. 현재 활동의 80 ms 내 짧은 발화 여부(any)와 미래 bin의 활동 점유율(>50%)도 구분한다. [VAP bin 정의](https://aclanthology.org/2022.sigdial-1.51/)

Hazard 초판 사건은 **현재 식별됐고 비활동인 각 슬롯의 다음 speech onset**이다. floor transfer·의미적 맞장구와 동일하지 않다. 활동 중이거나 아직 정체성을 모르는 슬롯은 위험집합에서 제외한다. horizon 2.56초를 완전히 관측한 무사건 구간은 survival 항에 포함하고 관측하지 못한 bin은 마스킹한다. 현행 `time_to_next_onset`은 2슬롯 전용이므로 K 일반화·위험집합·bin별 censor를 새 버전으로 검사한다.

### 6.3 ONSET/EOT 타이밍 라벨과 방출 의미

**Q1 기본은 C 모드다.** 종료 판정에 실제로 필요한 관측을 끝낸 뒤 EOT를 출력한다. 미래 예측은 Q3의 VAP/hazard 헤드와 별도 P-mode 토큰 ablation에 맡긴다. 헤드는 현재 미구현이며, 역할 분리가 성능 우위를 보장하지는 않는다. C도 행동 휴리스틱의 오라벨을 없애지는 못한다.

초기 후보 horizon은 offset 뒤 **3초**로 유지한다. “그 안에 본인 재개 없음”을 조건으로 사용하는 이상 **상대가 시작한 즉시 C 토큰을 내서는 안 된다.** 상대 시작 직후 내는 빠른 C 규칙을 시험하려면 판정 자체를 짧은 관측 구간으로 다시 정의·검증해야 한다. 미래 3초 조건으로 라벨을 걸러 놓고 그 전에 출력시키는 것은 선택 편향이 남은 P 타깃이다.

| 항목 | Q1 규칙 | 목표 위치 / 감독 |
|---|---|---|
| ONSET | 화자별 VAD, gap <0.25s 병합. onset detector의 실제 가용 시각도 기록 | 기존 δ_on∈{0,1,2}와 실제 가용 시각 중 늦은 시점 |
| EOT: shift 후보 | 다른 화자 시작이 horizon 안에 있고, 원 화자 재개는 horizon 전체에서 없음 | C: 최소 horizon 끝과 필요한 검출 지연을 관측한 뒤 |
| EOT: terminal overlap 후보 | 다른 화자가 offset을 지나 계속 말하고, 원 화자는 horizon 안 재개하지 않음 | 같은 C 기준. 짧은 겹침/귀속 불명확은 uncertain |
| **상대 시작 + 원 화자 재개** | 둘 다 같은 horizon 안에 있으면 순서와 무관하게 uncertain | 초기에는 해당 창 event 감독 제외; non-EOT로도 확정하지 않음 |
| 본인 재개·타 화자 없음 | 신뢰 가능한 pause negative 후보 | NEXT negative는 관측/라벨 완전성이 확인된 창에서만 |
| 모두 무음 / silence_timeout | 3초 실제 무음은 정책 증거이지 자동 floor gold 아님 | Q1 EOT positive에서 제외, timeout-policy 기록만 |
| 미래 부족·결손·짧은 겹침·다자 충돌 | 재개/상대/시각/귀속을 확정할 수 없음 | unknown mask; zero padding이나 파일 끝으로 보충하지 않음 |

이 표는 **보수적인 weak-label 선정 규칙**이다. 본인 재개와 상대 시작이 공존해도 실제 floor transfer였을 수 있으므로 uncertain은 오류 확정이 아니다. “marketing … expert”는 새 규칙에서 자동 positive가 아니라 uncertain으로 분류할 사례다. 기존 probe 산출물은 이전 P-rule 결과로 보존하며 새 승인 라벨로 재사용하지 않는다.

시간은 sample 정수로 계산한다. crop 원점 `a`, 청크 길이 `C=1280`, 필요한 관측 끝 `t_evidence`에 대해 `k_seen=ceil((t_evidence-a)/C)-1`이다. `k_eot=max(k_seen,k_last_text)`로 배정하고, encoder가 실제로 그 관측을 사용할 수 있는 시각·연산·cap 지연은 별도 더한다. `t_evidence`는 최소 offset+3초이며 규칙에 필요한 VAD 안정화 지연도 포함한다. `label_observed_until`을 단순 다른 화자 onset으로 축소하지 않는다.

예: offset=84.576초면 3초 비재개 조건의 관측 끝은 최소 **87.576초**다. 예전 P 타깃 84.96초를 C 타깃으로 이름만 바꿀 수 없다. 실제 AMI 사례에는 86.048초 본인 재개가 있으므로 C positive를 만들지 않는다. clean shift라면 관측 끝 이후 첫 청크와 마지막 lexical 중 늦은 위치에 배치한다. 같은 화자의 다음 ONSET을 EOT가 넘어가면 자동 재귀속하지 않고 창 event 감독을 보류한다.

P ablation은 동일한 유효 후보 집합에서 `floor((t_off-a)/C)+δ_text`로 조기 배정한다. 결과를 C와 섞지 않고 사건 precision/coverage, offset 기준 지연, 관측 완료 기준 지연을 함께 낸다. **3초 C는 저지연 제품 해법이 아니라 라벨/학습 가능성 대조군**이다. lexical 지연 관문과 EOT 지연을 분리하고, Q3의 빠른 예측 경로를 통과하기 전 저지연 turn-taking 달성을 주장하지 않는다.

원 `derive_events`는 2화자 함수다. K 일반화는 종료한 화자에게 귀속하고 같은 offset 중복을 제거하며 pairwise 충돌을 uncertain으로 남긴다. ONSET은 contribution 시작이 아니므로 EOT와 괄호로 짝짓지 않는다. 짧은 겹침을 semantic BC로 자동 확정하지 않는다.

API는 `endpoint_prediction`과 `event_mode=C/P`, `emission_source=model/timeout_policy/hazard_policy`를 함께 낸다. 이름에 관계없이 **전사 hard-final이 아니다**. 정책 이벤트를 모델 생성 토큰처럼 이력에 삽입하지 않는다. C의 “확인”도 지정한 관측 조건의 확인이지 의미적 종료 gold 보증이 아니다.

자동 라벨은 weak target이다. Q0 검수는 §9의 경량 범위로 제한한다. 모호한 라벨을 마스킹하더라도 NEXT의 softmax가 false negative를 학습하지 않도록 §6.1의 ASR-only/event-complete 분리를 유지한다.

### 6.4 Stage 3 이월

`<HOLD>`·`<BC>`, 의미 완결/미완결·floor 관계·의미 전용 head, LLM 시드→분류기 증류와 대규모 의미 라벨링은 제외한다. [[question-turn-token-label-reliability]]의 사람 간/모델-사람 agreement 파일럿은 Stage 3 착수 조건으로 남긴다. 미래 행동 자동 라벨의 성능으로 이 의미 과제를 통과했다고 주장하지 않는다.

## 7. 단계별 실험과 중단 기준

각 단계는 직전 통과 체크포인트를 기준으로 한다. 실험 비교는 동일 초기값·데이터·유효 노출량·seed를 사용한다. 다음 수치는 **파일럿용 제안 관문**이며 Q0에서 baseline을 측정한 뒤 장기 run 전에 동결한다. test 결과를 보고 관문을 바꾸지 않는다.

| 단계 | 추가하는 능력 / 데이터 | 주요 산출물 | 다음 단계 진입 |
|---|---|---|---|
| Q0 기준선·데이터 계약 | E2 δ=2/4, dyadic/회의·합성 QC pack 2–5 h | K·split·event rule·P/C 의미 동결, 복원·mask·serializer·no-future 검사 | 누락·덮어쓰기·누출 0; training 세션 화자 수 수용; 평가 재현 |
| Q1 비중첩 중심 K화자 전사·이벤트 | 약 20–50 h, EN/KO·화자 수별 층화; lexical warm-up → SPK/activity/ONSET/EOT(C) | K슬롯·32창 overfit·δ_on sweep·pause/무음 negative | ASR guardrail, dyadic DER ≤10%·귀속 오류 ≤5% 제안; C 지연·coverage 별도 |
| Q2 겹침 전사 | Q1 + 자연 dyadic/회의 overlap·2–4화자 합성; QC 통과 데이터만 확장 | 각 화자 전사·고정 ID·다자 event QC·밀도/삭제 진단 | overlap cp 오류 ≥20% 상대 감소·화자 소실 ≤5%를 dyadic 제안 관문으로; 회의는 동시 발화 수별 별도 판정 |
| Q3 미래 활동·행동 예측 | 완전 관측 자연 대화; frozen head probe → 저율 joint | K×4 미래 활동·선택적 hazard·P-token ablation·자체 이력 대조 | 동일 FPR에서 음향 대조군 대비 검증 가능한 개선, ASR guardrail 유지; 의미 라벨 필수 아님 |
| Q4 장문·실시간 통합 | 실제 및 합성 10–60분 세션, 자유실행·잡음·음량차 | K행 전사+활동+예측/정책 분리 API, 단독 장치 벤치 | RTF <1, backlog 비발산, ID·슬롯 초과·지연 관문 충족 |

32개 창 overfit는 코드 경로 검사다. unseen 화자 일반화의 근거로 쓰지 않는다. 초기 자연 데이터는 화자 다양성·턴 수·겹침 시간을 기준으로 추출하며 단순 폴더 순 20시간을 쓰지 않는다.

회의 관문은 동등 mono baseline 대비 전사·DER 변화와 절대 오류를 함께 기록하고 장기 run 전에 Q0에서 수치화한다. dyadic 평균이 좋아졌다는 이유로 회의 실패를 통과시키지 않는다. ONSET/EOT도 label recall·false-EOT/min·귀속 오류·지연의 baseline을 고정한다. 추가 데이터 준비와 별개로 Q0/Q1 검증 전 대규모 학습을 시작하지 않는다.

### 7.1 학습 recipe 시작점

- 모든 초기화는 E2에서 시작한다. §4.2 registry에 따라 슬롯 1/2는 `<SPK_A/B>` 기존 ID·행을 재사용하고 신규 슬롯/event 행과 heads만 초기화한다. 기존 Phase 1의 blocked 설정은 해당 모델에 유지하며 Phase 2 설정에서만 허용한다. ‘새 Qwen’ 재초기화와 비교하지 않는다.
- Q1에서 encoder를 잠시 동결하고 새 출력 규약을 배운 뒤, 정체 시 상위층부터 해동한다. 후보 LR은 thinker `5e-6–1e-5`, adapter `1e-5–5e-5`, 새 heads `1e-4`, encoder 해동 시 `1e-6–5e-6`. 이는 측정 전 탐색 범위다.
- E2의 `next_weight EN/KO=0.3/0.15`, delay 분포, TN을 첫 run에 유지한다. 태그 추가가 NEXT 비율을 바꾸므로 삭제·조기 방출·태그율을 보고 별도 sweep한다.
- Q1/Q2 시작 배치 예산은 대화 70% + 단일 화자 replay 30%의 **오디오 초 기준**으로 제안한다. 대화 내 합성 비율은 최대 약 절반부터 시험하며 자연 표본을 유지한다. ONSET/EOT·미래 활동·hazard 손실은 해당 목표가 완전 감독되는 자연 창에만 적용한다. dyadic/회의·EN/KO·동시 발화 수별 노출량을 따로 기록해 KO 추가 데이터가 EN을 압도하지 않게 한다.
- Q3는 head-only probe 이후 공유 모델 저율 joint를 비교한다. gradient norm·ASR 회귀를 보고 손실 가중을 정한다. 발화 길이와 corpus 구성 때문에 ‘30 epoch’를 이전 run과 같은 노출량으로 가정하지 않는다.
- free-running 텍스트·speaker/event prefix에서 head를 학습하는 단계를 포함한다. 정답 슬롯/이력 성능은 oracle 조건으로만 보고한다. rollout의 예측 슬롯과 참조 라벨 대응도 관측 prefix에서만 정하고, 대응 불명확 슬롯은 mask·coverage로 보고한다. 같은 checkpoint의 rollout은 model hash를 기록하고 모델 변경 후 갱신한다.

### 7.2 장문과 speaker memory

20–40초 독립 창 통과 뒤 60–120초 carry 학습, 10–60분 검증으로 늘린다. 과거에 관측한 음성·전사·speaker state만 carry한다. 오래 침묵한 화자 재등장, 중간의 신규 화자, 시작부터 겹침, 세션 누적 화자 수가 동시 화자 수보다 큰 사례를 별도 평가한다.

현재 KV를 무제한 유지하면 메모리가 증가한다. bounded context + 세션 speaker memory를 비교하고, memory는 관측 prefix에서만 갱신한다. 단순 KV 앞부분 삭제가 RoPE·화자 정체성을 보존한다고 가정하지 않는다. context 전환 시 logits·전사·ID 연속성, 메모리 상한을 테스트한다.

speaker memory는 Q1/Q2의 on/off 비교 대상으로 두며 필수 효과를 전제하지 않는다. 겹침·오귀속 예측으로 잘못 갱신되는 오염률과 회복을 측정한다. 총 context budget은 audio+NEXT+lexical+SPK+event를 모두 포함한다. 80ms당 audio 1개는 유지되지만 K 증가에 따른 출력 토큰·decode 횟수 증가는 따로 예산화한다.

## 8. 평가 및 주장 범위

### 8.1 인식·화자·이벤트·시스템 성능을 함께 보고

| 축 | 주 지표 | 규약 |
|---|---|---|
| 단일 화자 ASR | EN WER, KO CER; S/D/I | E2와 같은 dev/test·TN·δ. corpus별 상대 회귀 ≤5% 제안 |
| 다화자 전사 | EN cpWER/tcpWER, KO 문자 단위 permutation CER; attribution 오류 | 대화 전체 하나의 K화자 매핑; 참조/예측 수 불일치 처리 고정. overlap·음량차·언어·N별 분리 |
| Diarization | overlap 포함 DER: miss/FA/confusion, JER, ID switch | collar=0 주 지표, 250ms 보조. 80ms 해상도를 명시 |
| Overlap | 활동 overlap P/R/F1, 화자별 전사 recall | 동시 2/3/4+명 분리, 한 화자 소실을 평균에 숨기지 않음 |
| 화자 수 | 과소/과대 추정·신규/재등장 혼동·슬롯 초과율 | 세션 누적 화자 수와 동시 발화 수를 별도 집계 |
| Turn-taking | ONSET/EOT P/R·오귀속·중복·누락·false-EOT/min·지연, 미래 활동 calibration | 모델-only와 정책 포함 분리, P/C 분리; TurnBench EOT/INT는 dyadic 외부 평가 |
| 시간·시스템 | 방출 p50/p90/p99·viol80·matched coverage, TTFT, RTF·tick·backlog | 오디오 가용 시각/계산/네트워크·flush 별도; 단독 장치와 경합 구분 |

cpWER/tcpWER는 [MeetEval](https://github.com/fgnt/meeteval)의 버전·매핑·시간 collar를 고정한다. KO 문자 기준 지표는 EN WER와 이름을 구분한다. 각 화자 전사를 누적한 뒤 채점하며 청크마다 reference와 최적으로 재매칭하지 않는다. 전사 시간 제약의 수 초 collar가 80ms 방출 정확도를 보증하지 않으므로 timing 지표를 따로 둔다.

단일 화자 E2와 다화자 cpWER를 직접 나눠 ‘회귀 5%’를 적용하지 않는다. 동일 입력·채점 조건에서 Q1/Q2의 직전 기준 모델과 비교한다. 깨끗한 분리 채널에 E2를 각각 적용한 결과는 정보가 더 많은 **oracle 참고선**이다. RNN-T 다화자 대조군도 실제 mono diarization/분리 비용을 포함한 완성 파이프라인으로 비교한다.

‘화자 소실’은 참조 단어/문자가 5개 이상인 화자의 matched lexical recall이 10% 미만인 경우다. 화자-창 단위 비율과 ‘한 명이라도 소실한 창’ 비율을 모두 내고 N별로 분리한다. 짧은 맞장구는 별도 recall을 낸다. Overlap token은 참조 구간과 2명 이상 VAD 활동이 겹치는지로 분류하며 정렬 품질별 결과를 병기한다.

### 8.2 자체 전사 이력의 행동 예측 기여

1. **Audio-only**: 같은 mono encoder, 동일 시간/문맥 예산, 비슷한 head 용량의 causal turn predictor.
2. **Integrated**: 같은 오디오와 자체 생성 이력을 본 thinker audio-state head.
3. **Text ablation**: 입력 오디오는 유지하고 lexical 이력을 마스킹/교란한 조건. 분포 이동 영향 때문에 이에 맞춰 학습한 대조군도 둔다. speaker tags·활동 이력은 유지한다.
4. **Oracle history**: 현재 관측 시간까지 정렬상 사용 가능한 gold 전사만 제공. 완성된 미래 문장을 넣지 않는다. 실제 제품 성능으로 보고하지 않는다.

Phase 2는 pause 길이·중첩·화자 수별 subset에서 이력 사용 효과를 본다. 완결/미완결 의미 라벨을 새로 구축하는 실험은 Stage 3으로 이월한다. 자동 EOT target 향상이나 lexical masking 효과만으로 ‘의미 이해’를 증명했다고 쓰지 않는다.

dyadic 개선 목표는 같은 공식 FPR≤0.10에서 EOT/INT recall **+3%p 이상** 또는 recall을 유지하며 EOT p50 **80ms 이상 단축**으로 제안한다. 대화 단위 paired bootstrap 95% CI와 두 개 이상 seed로 확인한다. Q0에서 표본 크기·기준선을 보고 사전 동결한다. native ONSET/EOT의 FP/min과 공식 FPR은 분모가 다르므로 혼용하지 않는다. 회의에는 원 TurnBench dyadic 수치를 그대로 적용하지 않고 별도 사건 매핑·임계값을 고정한다.

TurnBench는 기존 공식 규약을 따라 비교하되, mono 혼합으로 바꾼 입력임을 명시한다. 기존 stereo VAP 숫자는 별도 입력 조건의 참고선이다. 기존 dev 반복 사용 사실과 hidden test 접근 여부를 기록하고, KO는 검수된 held-out 대화에서 같은 규약을 적용한다. [[turn-taking-evaluation-protocol]], [TurnBench 논문](https://arxiv.org/abs/2608.25218)

native EOT는 Q1에서 C-mode 행동 판단, Q3에서 P ablation이며 공식 EOT와 1:1이라고 가정하지 않는다. dyadic에서만 공식 EOT/INT adapter를 검증한다. 회의는 §6.3의 슬롯별 onset·offset/교대 후보라는 **명시한 로컬 사건 정의**로 보고하고, 인간 gold 없는 자동 라벨 점수는 weak-target agreement라고 표시한다. Stage 3 의미 평가와 별개다.

### 8.3 최종 채택 관문

- 위 ASR guardrail, Q1 K슬롯·ONSET/EOT, Q2 overlap, Q3 행동 예측 결과를 dyadic/회의별로 보고한다. 하나의 합산 점수로 약점을 가리지 않는다. 의미 완결성은 Phase 2 완료 조건이 아니다.
- 타이밍 목표는 신뢰도 높은 정렬 subset의 `viol80 ≤1%`, p99 방출 지연 ≤1초로 제안하고 정렬 불확실/미매칭 비율도 낸다. 기존 ‘위반 0’보다 완화한 **별도 제안**이며, 원래 관문 통과 주장에는 쓸 수 없다. 코드상 미래 정보 누출은 0건이어야 한다.
- 고정 장치에서 encoder+decoder+heads의 tick p99 <80ms를 지향하고 RTF <1을 필수로 둔다. 주기적 초과가 있으면 backlog p99·최대값·지속 추세를 함께 판정한다. 1시간 스트림에서 backlog가 계속 증가하면 채택하지 않는다.
- 10분 단위 ID swap을 기록하고 장문 총 ID switch ≤1회/10분을 초기 목표로 둔다. 사후 전체 파일 relabeling으로 온라인 성능을 대신하지 않는다.
- Turn 결과는 teacher forcing 없이 실제 전사 이력으로 평가한다. 공식 event 판단 시각에 lookahead를 포함하고, 서비스 지연은 wall-clock으로 추가 보고한다. 미래 onset 예측은 ASR 조기 전사 위반과 다른 개념이다.

## 9. 구현 작업 지도와 검증 순서

아래 파일명은 신규 구현 제안이며 아직 생성하지 않았다. 기존 HF Trainer+Liger를 유지하고 프레임워크 교체는 이 실험 축에 섞지 않는다.

| 작업 | 위치 | 필수 확인 |
|---|---|---|
| 대화 스키마·혼합·QC | 신규 `vapasr/data/dialogue.py`, schema, `experiments/p2_build_dialogue.py` | 합산·offset·채널·분할·태스크 마스크 |
| 시간순 화자 전사 | 신규 `vapasr/data/dialogue_interleave.py` | K 전사·event 복원, Unicode·공백, tie·flush·cap·원자 payload |
| turn 라벨·parser | 신규 `dialogue_turn_labels.py`, `vapasr/hf/dialogue_state.py`, `experiments/p2_build_turn_labels.py` | ONSET/EOT·P/C·K슬롯·미래 horizon·정책 분리, §4.4 fixtures |
| 모델 출력·손실 | `vapasr/hf/modeling_vapasr.py`, config/token registry | K·event vocabulary·legacy ID·tied embedding·mask·audio-position·partial supervision |
| activity/미래 활동/hazard | `vapasr/data/targets.py`의 버전 분기 | K×4 BCE, 50Hz 경계·80ms sampling, 미등장 슬롯·censor·permutation |
| 학습·replay | Trainer/data, 신규 `experiments/p2_train_hf.py` | 초 기준 mixture·재개·rollout provenance·DDP 불균등 평가 |
| 평가 | 신규 `experiments/p2_eval.py` | speaker-aware 전사·DER·turn·latency, dev/test 불변 |
| live | `vapasr/hf/live.py`, `live_mlx.py`, `experiments/live/` | K 버퍼·80ms heads·event source·최초 출력·장문·capacity overflow |

특히 E2 encoder를 **동결한 뒤 저장/재로드해도 E2 가중치가 유지되는지** 검사한다. 현재 저장 로직은 `encoder_trainable`에 따라 encoder를 생략하므로 학습 여부와 저장할 가중치 provenance를 분리해야 한다. frozen E2 대신 원본 `.nemo`를 재부착하면 다른 모델이다.

현재 deferred `<NEXT_AUDIO>` 최적화는 다음 audio와 묶어 forward한다. 새 head는 묶음의 마지막 audio 위치를 명시적으로 읽어야 한다. MLX가 지금처럼 마지막 logits만 돌려주는 경로에는 hidden-state/head 지원이 필요하다. CPU 왕복 때문에 얻었던 속도 이득을 잃지 않는지도 확인한다.

검증 순서는 serializer round-trip → 실제 Dataset 오디오 spot-check → 32창 overfit → HF save/load logits·heads·encoder parity → prefix 절단/미래 교체 인과성 → 단일/불균등 multi-rank smoke → 동일 held-out 자유실행 → live parity → 장문이다. tiny run에서 optimizer/scheduler/RNG·샘플 위치·split hash 재개도 확인한다.

`tests/test_dialogue_turn_sequence.py`에 §4.4 전 사례와 K+1번째 화자·미등장 슬롯 future mask·결손 화자 조각·EOT 후 지연 lexical·ONSET 반복·중복 EOT·timeout 분리·crop EOF를 fixture로 둔다. causal 입력을 고정하고 미래 suffix만 바꿨을 때 **추론 출력은 같아야 하지만 미래 예측 정답은 달라질 수 있다**. 두 검사를 혼동하지 않는다.

실데이터 사전 점검: [[output-phase2-real-sequence-probe]](2026-09-13)는 AMI 4화자 38.4초로 480개 블록·BPE ID·next-token label을 생성했다. encoder/모델 추론은 미실행이며, 자동 EOT가 `marketing … expert` 사이에 놓이는 검수 후보가 확인됐다. 청취 전 확정 오답으로 취급하지 않는다. Q0 라벨 검수 사례로 사용하고 이 직렬화 결과만으로 학습 관문을 통과시키지 않는다.

**검수 범위를 경량화한다.** Q0/F2는 `TurnBench dev gold 대조 + AMI 200경계`로 제한한다. AMI는 무작위 EOT 후보 100·no_eot 50·위험 50을 구분하고, 기존 재생 도구와 JSON/TSV 응답으로 처리한다. 정적 HTML·전 코퍼스 200개씩·필수 2인 전수 검수·독립 릴리스 플랫폼은 Q1 선행 조건에서 뺀다. 애매한 사례만 추가 검토하고 안 풀리면 uncertain이다. KO 1대화는 §5.3의 음향·정렬·시퀀스 QA로 유지하되 별도 의미 라벨링 프로젝트로 확대하지 않는다.

이 축소 검수로 AMI 전체 EOT recall·KO floor 라벨 품질·검수자 일치도 통과를 주장하지 않는다. 후보 조건부 precision·오류 유형·unknown/유효 감독 coverage를 먼저 보고한다. TurnBench dev를 보고 규칙을 수정하면 해당 점수는 **튜닝된 dev 결과**이며 독립 검증이 아니다. 최초 고정 규칙 결과를 보존하고, 새 held-out 평가 전 일반화 주장을 보류한다. 작업 상세만 [[output-phase2-eot-review-framework]]에 두고 사건/직렬화 규칙은 이 정본만 따른다.

구현 우선순위는 `(1) registry·TN fail-fast → (2) 공용 serializer·state와 §4.4 16 fixtures → (3) 71631 실외 480블록 실제 Dataset·head targets → (4) 경량 QC·밀도 → (5) Trainer 32창 overfit`다. HF Trainer가 실제 batch를 읽고 forward/backward·save/load까지 통과해야 연결 완료다. probe/replay 독립 JSON은 이 관문을 충족하지 않는다.

## 10. 실행 예산과 우선순위

1. **첫 묶음, 약 2–4 작업일 제안:** Q0 데이터 계약·평가 pack·기준선, mono mixer와 serializer, 작은 overfit. 이 단계 종료 전 대규모 job을 제출하지 않는다.
2. **둘째 묶음, 약 1주 제안:** Q1 비중첩 화자 전사와 Q2 overlap pilot. 화자 추적·낮은 음량 삭제·생성량 증가를 먼저 해결한다.
3. **셋째 묶음, 약 1–2주 제안:** Q3 audio-only/thinker heads, 행동 라벨 QC, 자유실행 history 학습 및 ablation. 대규모 의미 라벨링 제외.
4. **넷째 묶음, 약 1주 이상 제안:** Q4 장문·실시간 통합과 고정 test 보고. 사람 라벨링·데이터 접근·선점 대기는 별도다.

이는 연구 개발 순서와 대략적인 인력 일정이지 GPU 완료 시간 예측이 아니다. Q0/Q1의 100–300 step으로 `audio-seconds/GPU-second`, tokens/second, peak memory, 평가 RTF를 실측한다. 이후 `GPU-hours = 총 노출 audio-seconds / 실측 처리량 / 3600`으로 예산을 계산하며 정렬·rollout·eval 비용을 따로 더한다. 최초 pilot은 1–8 GPU로, 통과 후 필요한 규모로 확장한다.

데이터 목록의 약 840 h/epoch는 자연 290 h+합성 300 h와 replay 30%를 가정한 예시일 뿐 고정 schedule이 아니다. 134-1·회의 편입 뒤 재계산한다. E2의 1.02s/step을 그대로 사용해 ‘다화자도 1시간/epoch’라고 예측하지 않는다. 길이·K·event 밀도·활성화·평가 비용이 달라진다.

ASR 자체 개선(E3 추가 학습, next_weight, δ 목표 변경)은 E2 기준 별도 run으로 검증한다. 개선 체크포인트를 Phase 2에 도입할 때 Q0 pack을 다시 평가하고 기준 변경을 기록한다. 다화자 학습·인식 recipe·turn loss를 동시에 바꿔 원인을 놓치지 않는다.

## 11. 주요 실패 가설과 다음 조치

| 관측 | 먼저 구분할 원인 | 다음 한 가지 실험 |
|---|---|---|
| 작은 화자 전사가 사라짐 | mixer 결함 / encoder 혼합 정보 부족 / decoder cap | clean-channel oracle와 저음량 합성으로 위치 확인 후 encoder 해동 범위 비교 |
| 전사는 맞는데 슬롯이 뒤집힘 | local ID 계약 / 긴 문맥 손실 / speaker 표현 부족 | permutation·carry 검증 후 speaker memory/t-vector 보조 학습 |
| overlap F1은 높지만 전사는 한 사람뿐 | 활동 감지와 음성 내용 분리 능력 차이 | 화자별 D/S/I·전사 recall, multi-output decoder를 제한적 구조 ablation으로 검토 |
| turn head는 gold에서만 좋음 | ASR 지연·오류 이력의 노출 편향 | self-generated history 학습, δ별 turn curve |
| thinker가 audio-only보다 못함 | 행동 라벨 잡음 / 자체 이력 오류 / 80ms 운율 손실 | 라벨·슬롯 대응 검수 후 고해상도 causal acoustic branch ablation |
| ONSET 반복이 막히거나 EOT가 전사를 끊음 | contribution 문법이 음향 onset에 잘못 적용됨 | §4.4 parser fixture, predicted endpoint와 transcript-final 분리 |
| 새 화자를 기존 슬롯으로 합침 | 슬롯 용량 / 새 화자 검출 / memory 오염 | N별 오귀속·capacity 초과 분리 후 K/identity 실험 |
| 평균 속도는 좋고 장문은 밀림 | burst·KV 증가·speaker token 비용 | context 상한·cap·runtime profiling, 같은 정확도에서 개선 검증 |

고해상도 branch, 분리 보조 학습, multi-output decoder는 최초 주 경로에 넣지 않는다. 관측된 실패 원인을 겨냥한 후속 비교이며 입력은 계속 mono다. Q3에서 행동 예측 개선이 없으면 ‘공유 모델로 화자 전사와 활동을 처리했다’까지 주장한다. 의미 기반 턴 이해는 별도 Stage 3 검증 전까지 미입증이다.

## 12. 근거와 남은 불확실성

프로젝트 근거는 [[output-stage2-e2-final-eval]], [[decision-mono-input]], [[output-vapasr-model-and-sequence]], [[source-hf-trainer-migration]], [[output-vap-target-pipeline]]와 이번에 읽은 `vapasr/hf/`, `vapasr/data/streams.py`, `vapasr/data/targets.py`다. 코드 관찰은 2026-09-11 작업 트리 기준이다.

외부 1차 자료 확인일: 2026-09-11. [t-SOT](https://arxiv.org/abs/2202.00842)와 [t-vector](https://arxiv.org/abs/2203.16685)는 직렬화·화자 귀속의 근거이며 E2+LLM의 성능을 보증하지 않는다. [VAP 공식 구현](https://github.com/ErikEkstedt/VoiceActivityProjection)은 현재·미래 활동 공동 학습과 입력 조건을 확인하는 참고다. [MeetEval](https://github.com/fgnt/meeteval)은 다화자 전사 평가, [TurnBench](https://arxiv.org/abs/2608.25218)는 턴 이벤트 평가 근거다.

데이터 가용량은 [[output-phase2-data-inventory]]·[[output-phase2-db-survey]]로 보완했다. 남은 Q0–Q3 항목은 실제 K·다자 event 규칙과 P/C 모드 고정, 대화/화자 노출 감사, 조각 복원 완전성, 겹침 정렬·다자 주석 coverage, speaker 표현, mono turn 기준선, 실제 K화자 처리량이다. 한국어 의미 라벨 agreement는 [[question-turn-token-label-reliability]]에 따라 Stage 3 전 검증으로 이월한다.
