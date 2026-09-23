---
type: output
status: active
created: 2026-09-14
updated: 2026-09-15
summary: 대안 상세안 — R 전사 채널·발화 episode·동적 화자 메모리 분리, SEG_END와 pointer EOT, 학습·HF 연결·단계별 관문
sources:
  - '[[output-phase2-speaker-representation-comparison]]'
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[output-phase2-eot-review-framework]]'
  - '[[output-phase2-training-db]]'
---

# 대안 설계: 재사용 전사 채널 + 동적 화자 메모리

## 0. 결론과 문서의 권한

**전사 채널은 짧게 빌려 쓰고, 발화 기록과 사람 ID는 별도로 유지한다.** 모델은 희소한 `<SPK_17>`을 분류하는 대신 공유 speaker encoder·matcher로 관측된 메모리 중 같은 사람 또는 새 사람을 판단한다. 회의 전체 인원 증가가 vocabulary 증가로 이어지지 않게 한다.

이 문서는 사용자가 요청한 **세 번째 후보의 상세 실험 계획**이다. 고정 K 정본을 대체하지 않으며 구현·실측·채택은 아직 없다. 모델 입력 mono, 80ms audio clock, E2·TN, Q1 C-mode/후속 P ablation은 비교 기준과 맞춘다. 단, 아래 `SEG_END`, dynamic reference와 head는 별도 프로토콜 변경이다. 정본과 혼합한 데이터셋을 만들지 않는다.

**조건부 후속 검토(2026-09-15):** 동일 인물의 재등장 매핑을 외부 화자 모듈에 맡겨도 된다면, [[output-phase2-external-speaker-mapping-plan]]의 local lane/episode 중심 분리형을 먼저 검증한다. 본 상세안은 내부 정체성 관리가 필요한 경우의 대안으로 보존하며, 외부화가 채택됐다고 간주하지 않는다.

가장 중요한 결론은 **채널 → 발화 episode → 세션 화자**의 3층 분리다. 채널과 사람 ID만 나누면 지연 EOT가 재사용된 채널의 새 사람에게 잘못 붙는 문제가 남는다.

## 1. 고정할 최소 설계

아래 수치는 최초 비교를 위한 제안값이며 데이터 분포·처리량으로 재검토한다.

| 항목 | 시작값/정책 | 의미 |
|---|---|---|
| 전사 채널 R | 4, R=2 대조 | 동시에 미완료인 전사 구간을 담는 용량; 세션 인원 아님 |
| 세션 메모리 N | 가변, 최초 운영 상한 64 | 관측된 화자 prototype 목록. “무제한 지원” 아님 |
| 화자 표현 | 256차원 정규화, 공유 extractor | ID마다 다른 분류 행을 만들지 않음 |
| 화자당 prototype | 최대 4개, 신뢰 구간으로 갱신 | 잡음/음색 변화 대응; 표본 부족 시 1개부터 |
| 미해결 event episode | 최대 64개부터 측정 | R 채널과 별도 저장. 초과는 명시적 오류/coverage |
| 입력 | E2 mono encoder [56,0], 80ms | clean channel·미래 오디오는 입력 금지 |
| lexical | δ=4 주 평가, δ=2 보조 | 기존 TN·정렬 계약 유지 |
| EOT | C-mode + episode reference | 수초 후 판단해도 이전 발화에 귀속 |
| 메모리 수명 | 한 세션 안, 재시작 시 초기화 | 세션 간 사람 인증·영구 voiceprint 저장은 범위 밖 |

메모리는 모델 런타임의 RAM/GPU 상태다. 위키 검색용 임베딩 DB나 별도 서버를 도입하는 것이 아니다. 64명×4개×256차원×FP32는 prototype만 약 256KiB지만, 실제 비용은 speaker feature history·KV·batch state·query 연산까지 실측한다.

## 2. 세 종류의 ID

| ID | 생성/수명 | 예시 | 사용 |
|---|---|---|---|
| `lane_id` | R개 고정, 재사용 | lane_2 | lexical 출력 라우팅 |
| `episode_id` | 새 전사 구간마다 증가, 재사용 금지 | ep_0041 | pending lexical·ONSET/EOT의 소유자 |
| `speaker_id` | 같은 사람으로 판정된 세션 기록 | speaker_007 | 누적 전사·화자별 활동·미래 예측 |

lane은 `(lane_id, generation)`으로 식별해 이전 소유자와 새 소유자가 충돌하지 않게 한다. 모든 출력에 episode_id를 붙인다. speaker_id는 늦게 확정될 수 있지만 episode_id는 생성 즉시 안정적이다.

episode는 semantic turn이 아니라 **음향·전사 처리 단위**다. 같은 사람이 pause 뒤 재개하면 새 episode를 만들 수 있으며 기존 episode에 EOT가 반드시 있어야 하는 것은 아니다. `<SEG_END>`는 episode의 lexical 처리 종료일 뿐 turn 종료가 아니다.

추가로 `event candidate`의 key에는 episode 요약뿐 아니라 생성/lexical-close 시각과 현재 audio clock 대비 나이를 제공한다. 모델이 C 지연을 판단할 시간 정보도 필요하다. 이 clock은 실제 처리한 오디오로 계산하며 참조 미래 시각으로 업데이트하지 않는다.

## 3. 모델 구성과 한 tick의 순서

```text
mono PCM → E2 encoder → adapter → thinker
                                  ├─ lexical + LANE/ONSET/SEG_END/EVENT_REF/EOT/NEXT
                                  ├─ source-conditioned speaker embedding
                                  │        → 공유 matcher → 세션 speaker memory
                                  └─ h_audio + 과거 speaker memory
                                           → 현재 activity / 미래 4-bin / hazard

runtime: lane table ↔ episode table ↔ speaker memory
```

**처리 순서:** (1) 직전 tick까지의 memory snapshot 고정 → (2) 새 audio embedding 처리 → (3) h_audio에서 memory-conditioned heads 계산 → (4) lexical/구조 토큰 autoregressive 생성, speaker embedding·pointer 계산 → (5) episode·prototype 갱신 → 다음 tick. 현재 tick에서 뒤늦게 생성된 gold/예측 텍스트를 앞선 h_audio 헤드에 소급 제공하지 않는다.

처음부터 R개의 LLM decoder를 만들거나 R개의 audio soft token을 넣지 않는다. decoder는 하나, audio token도 하나다. speaker extractor와 matcher는 작은 부가 경로로 시작한다. 추가 토큰·참조 선택·head 계산 비용은 공짜가 아니므로 전체 tick에 포함한다.

### 3.1 화자 표현을 어디서 얻는가

최초 구현 제안은 `z = normalize(g(h_payload, q_episode, causal_audio_context))`다. g는 episode/query에 조건화한 작은 cross-attention + projection이며 모든 lane/person이 공유한다. h_payload는 해당 lexical 또는 ONSET을 처리한 decoder state다. 오디오 문맥은 실제 가용한 encoder prefix까지만 사용한다.

혼합 h_audio 하나를 모든 화자에게 복사해 speaker embedding으로 쓰지 않는다. 세 명이 같은 시각에 말하면 **선택된 source에 따라 다른 표현**을 내야 한다. 이를 overlap 별 same/different-speaker 검사로 검증한다. lexical 이후 정보가 필요한 z는 그만큼 늦게 사용하고 ONSET 당시 이미 알고 있었다고 기록하지 않는다.

ONSET만 있고 음성이 너무 짧으면 episode를 provisional로 유지한다. 80ms 하나로 사람을 확정하도록 강요하지 않는다. 현재 활동에서 미식별 발화의 coverage를 별도로 보고하며 신규 화자 검출을 기존 memory head에만 맡기지 않는다.

## 4. 출력 시퀀스와 새로운 구조 토큰

실제 token ID는 migration 시 새 registry로 동결한다. **기존 `<SPK_A/B>`의 의미를 조용히 바꾸지 않는다.** 대안 checkpoint에 `<LANE_1..R>`, `<SEG_END>`, `<EVENT_REF>`를 별도로 등록하고, ONSET/EOT/NEXT는 registry를 확인해 재사용한다. 총 vocabulary는 세션 N과 무관하다.

기본 예시:

```text
[AUDIO_k] <LANE_2><ONSET> 안녕하세요 <NEXT_AUDIO>
[AUDIO_k+1] <LANE_2> 저는 … <LANE_3><ONSET> 네 <NEXT_AUDIO>
…
[AUDIO_j] <LANE_2><SEG_END><NEXT_AUDIO>
…
[AUDIO_m] <EVENT_REF>{episode pointer}<EOT><NEXT_AUDIO>
```

빈 lane 뒤의 ONSET은 새 generation/episode를 연다. 활성 lane의 lexical은 같은 episode에 붙는다. `SEG_END` 뒤 lexical을 이전 episode로 조용히 해석하지 않는다. 잘못 닫은 모델의 후속 lexical은 새 episode/protocol error로 보존·계수하고 전사를 삭제하지 않는다.

`{episode pointer}`는 **사람이 읽기 위한 표기이며 tokenizer 문자열/정수 ID 토큰이 아니다.** vocab의 EVENT_REF를 낼지 결정한 decoder state에서 pointer head가 현재 보유 episode 중 하나를 선택한다. 선택된 episode key를 EVENT_REF의 입력 embedding에 더한 뒤 다음 EOT를 예측한다. 학습에는 categorical pointer target, 추론에는 pointer argmax/신뢰도가 별도로 저장된다.

확률은 `p(EVENT_REF | prefix) × p(ep | prefix, live_episode_table) × p(EOT | prefix, ep)`로 분해한다. 일반 텍스트 tokenizer만으로 round-trip할 수 없으므로 결과를 token IDs와 pointer sidecar가 포함된 structured sequence로 저장한다. 해당 표를 잃으면 replay를 거부한다. episode key는 과거에 생성된 정보만 포함한다.

이 경로를 쓰는 이유는 이미 재사용된 lane에 과거 EOT를 다시 선택시키지 않기 위해서다. EOT는 계속 모델이 생성하는 이벤트이며 pointer 선택은 모델 출력의 일부다. 매칭 실패 때문에 종료를 정책으로 강제 생성해 모델 성능으로 보고하지 않는다.

초기 pointer 후보는 **관측 이력에서 생성되고 lexical-close된 미해결 episode**다. training에서는 당시 teacher-forced state, inference에서는 예측 state만 사용한다. 정답 EOT 유무로 후보를 미리 걸러내지 않는다. C 시각은 우선 학습 타깃으로 주며 모델의 조기 EOT를 오류로 측정한다. 엄격한 관측시간 guard를 추가한다면 실제 모델 추정 경계/clock만 사용하고, guard 전후 성능과 정책 효과를 분리한다. guard가 오류를 막았다는 이유로 모델이 타이밍을 학습했다고 주장하지 않는다.

## 5. lane 상태와 해제: EOT를 기다리지 않는다

lane 상태는 `FREE → OPEN → FREE`, episode 상태는 별도로 `LEXICAL_OPEN → LEXICAL_CLOSED → EVENT_RESOLVED/UNKNOWN`을 가진다.

1. 새 source가 필요하면 출력된 `LANE_r ONSET`으로 FREE lane을 할당한다. 동시에 여러 source가 시작한 경우 초기 판별 불확실성도 기록한다.
2. 학습 reference의 `SEG_END` 목표는 해당 음향 segment의 마지막 lexical 목표 시각과 offset 후 짧은 관측 gap 중 늦은 시각이다. 초기 gap 제안은 240ms이며 Q0에서 동결한다. gap 안 재개가 있으면 같은 segment로 병합한다.
3. 추론은 정답 마지막 단어·참조 offset을 모른다. **SEG_END를 모델이 예측**하며 학습 시각은 감독일 뿐 런타임 oracle가 아니다. 시작은 teacher forcing으로 검사하되 실제 자유실행 종료 오류가 채택 지표다.
4. SEG_END가 출력되면 해당 episode의 소유자·요약·마지막 시각·event 상태를 episode table에 남기고 lane만 해제한다. 모델이 내놓은 lexical의 처리 완료 선언이지 물리적으로 정답을 다 냈다는 보증이 아니다.
5. C-mode EOT는 정본의 3초 관측 조건을 만족한 사건만 별도 pointer 경로로 학습한다. 본인 재개와 상대 시작이 공존하는 uncertain 구간은 정본처럼 event 감독에서 제외한다.

**R은 순간 겹침 수 S와 같다고 가정하지 않는다.** lexical 지연·SEG_END 오류·빠른 교대 때문에 S≤2라도 R개 lane이 모두 사용 중일 수 있다. `busy-lane occupancy`와 실제 S를 따로 측정해 R/해제 규약을 고른다. 해제만 빨리 강제해 삭제율을 높이지 않는다.

lane이 모두 사용 중이면 `lane_capacity_exhausted`를 보고한다. 기존 lane을 덮어쓰거나 가장 작은 화자를 무음으로 처리하지 않는다. 미귀속 전사 fallback을 두는 후속 실험도 main 성능과 분리한다.

lexical-close episode의 event 후보는 EOT가 나오면 resolved로, 초기 제안 TTL 8초를 지나도록 안 나오면 `event_unresolved_timeout`으로 큐에서 내린다. TTL은 의미 종료 라벨이 아니며 EOT 토큰을 만들지 않는다. 참조 사건이 있었다면 누락으로 평가한다. 소유자 매핑/출력 이력은 결과 로그에 남기고 live pointer 큐만 bounded로 유지한다. TTL/64개 상한의 잘림률을 계수해 C 타깃이 큐 밖으로 밀리지 않는지 확인한다.

### 5.1 채널 재사용 뒤 늦은 EOT 예시

아래는 **설계용 가상 시나리오**, 실측 데이터가 아니다.

| 시점 | lane_1 | episode table | 출력의 소유자 |
|---|---|---|---|
| 민수 시작 | generation 1, ep_01 | ep_01 → speaker_001 | lexical/ONSET → ep_01 |
| 민수 전사 종료 | SEG_END 뒤 FREE | ep_01은 C 판단 대기 | 이미 출력된 전사는 speaker_001 유지 |
| 철수 시작 | generation 2, ep_03 | ep_03 → speaker_003 | 새 lexical → ep_03 |
| 민수의 C 판단 시점 | 여전히 철수 사용 중 | ep_01을 pointer로 선택 | EOT → speaker_001, 철수와 무관 |
| 민수 재등장 | 다른 빈 lane, ep_05 | matcher가 speaker_001에 연결 | 새 episode지만 동일 speaker |

민수의 재개가 기존 C horizon 안이었다면 ep_01 EOT는 자동 positive가 아니다. 예시의 EOT는 horizon 이후 재등장 조건으로만 성립한다. lane 해제가 EOT를 확정하지 않는다.

## 6. 메모리 매칭·갱신

각 speaker entry는 `speaker_id, prototypes[P,d], quality, observed_duration, last_seen, status`를 갖는다. raw corpus의 실제 인물 ID는 정답/평가에만 사용하고 모델 state에 넣지 않는다.

### 6.1 공유 matcher

episode의 누적 z와 각 prototype 사이 cosine/작은 공유 network score를 계산한다. 첫 구현은 모든 N≤64 entry를 비교한다. 후보 검색을 top-L로 줄이는 최적화는 recall을 측정한 뒤 한다.

판정은 세 가지다.

- **KNOWN:** top-1 score와 top-1/2 margin, 음성 증거 품질이 dev에서 동결한 조건을 충족하면 기존 ID 연결.
- **NEW:** 충분한 증거가 있는데 기존 entry와 일치하지 않으면 새 ID 발급. 후보 수 N별 false-new/false-known calibration을 검사한다.
- **UNRESOLVED:** 증거가 짧거나 겹침·음향 품질이 나쁘면 provisional episode 유지. unknown을 모든 다른 사람과 같은 하나의 speaker로 합치지 않는다.

임계값·필요 음성 길이는 데이터 없이 숫자로 확정하지 않는다. 초기 평가에서 증거 길이별 0.25/0.5/1/2초 구간을 비교하되 그 시간을 의무 대기로 고정하지 않는다. ASR는 provisional episode로 바로 전달하고 화자 확정만 늦출 수 있다. 최초 출력·ID 확정·수정 지연을 따로 보고한다.

### 6.2 보호 규칙

- prototype 갱신은 높은 신뢰도의 source-conditioned 표현으로만 한다. 초기에는 비중첩·충분한 발화 위주로 제한하고 overlap 갱신을 ablation으로 추가한다.
- 신규 episode를 가장 가까운 entry에 무조건 연결하지 않는다. 비슷한 음색·짧은 맞장구를 hard negative로 검사한다.
- 잘못된 초기 배정을 수정하면 `speaker_assignment_revision`을 추가한다. 과거 event 소유 episode는 바뀌지 않고 episode→speaker 연결 revision만 바뀐다. 최초 출력 오류를 지우지 않는다.
- 같은 사람으로 보이는 두 동시 episode가 있으면 자동으로 중복 채널을 합치지 않는다. lane duplicate 또는 matcher 충돌로 기록하고 low-confidence 갱신을 막는다.
- 메모리 상한에 닿아도 기존 ID를 새 사람에게 재사용하지 않는다. `speaker_memory_full`과 provisional을 보고한다. 64를 넘는 장기 운영은 별도 상한/eviction 설계가 필요하다.
- training/export에 세션별 생체 표현을 영구 저장하는 방식은 기본에서 제외한다. 학습 재개용 state 저장이 필요하면 접근·보존 범위를 따로 정한다.

## 7. activity·VAP·hazard도 공유 head로 바꾼다

고정 K행을 그대로 두면 ID 토큰만 없애도 전체 모델의 인원 상한은 남는다. 각 알려진 speaker j의 **직전 tick까지** prototype에 대해 공유 함수 `f(h_audio[k], m_j)`를 적용한다.

- 현재 활동: `[N]` sigmoid.
- 미래 활동: `[N,4]` sigmoid, 정본의 4-bin 경계와 점유율 정의 유지.
- 다음 onset hazard: `[N,B]`, 현재 비활동·정체성 알려진 entry만 위험집합.
- 미식별 새 발화: lane ONSET/provisional 경로로 검출. NEW speaker에게 과거부터 특정 ID의 future target을 주지 않는다. `any-new-speaker` 미래 head는 후속 옵션이다.

한 번에 한 명을 고르는 softmax로 activity를 만들지 않는다. 여러 entry가 동시에 활성화될 수 있다. 반대로 N이 커지면 다수 음성 negative가 loss를 압도하므로 active·inactive 항의 분모/가중을 따로 기록한다. 아무것도 모르는 slot을 negative로 채우지 않는다.

speaker가 아직 unresolved라면 identity-specific head supervision을 mask하고 그 coverage를 보고한다. 정답 화자 수/미래 roster로 head query를 채우지 않는다. 모든 memory를 매 tick 조회하는 비용부터 실측하고 잠든 화자를 무조건 빼 재개 예측을 잃는 최적화는 피한다.

## 8. 학습 라벨과 인과성 계약

### 8.1 라벨 파이프라인

`원 대화 split → TN target/flags → 화자별 FA·VAD → mono/관측 mask → 음향 segment → lane allocation/episode → lexical·SEG_END → C-event/episode reference → head targets → Dataset` 순서다. 71631 실외 원본–조각 대응 검증과 AMI 원 시각 QA는 정본 경로를 재사용한다.

lane allocator는 episode 시작 시 사용 가능한 lane을 고른다. 초기 기본은 알려진 free-lane 순서다. 학습 정답의 SEG_END 시각으로 해제한 teacher allocator와 모델 SEG_END로 해제한 runtime allocator의 차이는 rollout에서 반드시 평가한다. t-SOT의 정답 utterance-end 기반 채널 재사용을 그대로 online 증거라고 부르지 않는다.

### 8.2 뒤쪽 lane의 학습 부족을 줄이는 방법

세션/crop마다 FREE lane의 우선순위를 바꾸되, 그 순서를 **짧은 prefix의 lane token 목록**으로 모델에도 제공한다. 예: `[LANE_3, LANE_1, LANE_4, LANE_2]`. 같은 음성에 다른 정답을 줄 때 어떤 순서인지 입력에 드러나게 한다. 추론은 고정 순서 또는 명시된 순서를 사용한다. 메모리의 사람 ID와는 무관하다.

lane token·episode state·activity/projection labels는 같은 permutation을 따른다. 2화자 자료도 모든 lane token을 positive로 학습시킬 수 있지만, **3중 겹침과 여러 사람 재식별의 훈련을 대체하지 않는다.** 후자는 별도 자연/합성 데이터로 확보한다. 도착순 세션 ID를 쓰는 정본 baseline에는 이 augmentation을 몰래 적용하지 않는다.

### 8.3 미래를 넣지 않는 teacher forcing

참조 speaker ID는 loss의 같은/다른 사람 판정에만 쓴다. 모델이 읽는 prototype은 이전 chunk의 source-conditioned z에서 생성한다. 미래 발화·깨끗한 채널 전체 pooling·미래 full transcript를 prototype으로 넣지 않는다. 동일 인물의 미래 표본을 contrastive **loss target**으로 쓰는 실험도 입력과 분리해 provenance를 기록한다.

teacher-forced lane/episode 이력은 초기 plumbing 검사에 쓰되, 이후 예측 lane·SEG_END·speaker assignment에서 생성한 state로 학습한다. 상태가 틀렸을 때 gold state로 조용히 복구하면 free-running 문제를 숨기므로 복구 빈도/coverage와 oracle 결과를 분리한다. memory 연결이 불확실하면 해당 ID loss를 mask하고 lexical를 보존한다.

### 8.4 손실

`L = L_AR + λ_spk L_metric + λ_match L_match + λ_ref L_episode_pointer + λ_act L_activity + λ_future L_future + λ_haz L_hazard`.

- `L_AR`: lexical·lane·ONSET·SEG_END·EVENT_REF·EOT·NEXT의 weighted CE. 새 구조 토큰을 기존 text 정확도에 섞지 않고 별도 보고한다.
- `L_metric`: same/different speaker contrastive 목표. 다른 대화에서 우연히 같은 speaker index를 positive로 묶지 않는다. 언어·성별만으로 구분하는 shortcut을 검사한다.
- `L_match`: 당시 관측된 memory entry 중 정답 또는 NEW의 분류. unknown/너무 짧은 구간은 명시적 유효 mask.
- `L_episode_pointer`: reference가 당시 episode table에 존재하는 event에만 CE. 미래 episode가 후보에 들어가면 검사 실패.
- activity/future/hazard는 정본의 관측·위험집합·censor 조건을 따르고, 항마다 유효 분모를 기록한다.

처음부터 모든 loss를 켜지 않는다. lexical/lane/SEG_END → speaker embedding/matcher → EOT reference → future heads 순서다. event 라벨 없는 replay/합성에서는 정본의 event-logit 제외 보조 CE를 새 이벤트 문법까지 확장해 false negative를 막는다. 단순 EOT label `-100`만으로 해결했다고 하지 않는다. 합성에는 신뢰 가능한 음향/전사·segment 목표만 사용하고 자연 floor EOT를 붙이지 않는다.

## 9. HF Trainer에 연결하는 방법

현재 `VapAsrForStreamingASR.build()`는 audio 위치를 embedding으로 바꾸고, `forward()`는 전체 teacher-forced 시퀀스를 한 번에 처리한다. **동적 prototype을 이전 decoder 출력으로 갱신하는 이 설계를 그대로 한 번의 병렬 forward에 넣을 수 있다고 가정하지 않는다.** 모델 출력 의존 state에는 순차 처리가 필요하다.

1. 최초 speaker 실험은 고정 K/lane 모델의 causal hidden rollout을 저장해 공유 extractor/matcher의 가능성을 확인한다. oracle 귀속을 사용한 결과와 예측 귀속 결과를 분리한다.
2. 실제 동적 모델의 첫 구현은 **chunk 순차 teacher forcing + 짧은 TBPTT**다. memory는 chunk 경계에서 detach하고 동일한 decode/state transition 함수를 사용한다. 느리더라도 causal 정확성을 먼저 검증한다.
3. 긴 학습 전에 teacher rollouts에서 만든 과거 memory snapshot을 재사용하는 2-pass 근사를 비교할 수 있다. snapshot model hash·생성 이력·staleness를 기록하고 최신 자유실행 평가를 필수로 둔다. 근사와 joint end-to-end를 같은 이름으로 보고하지 않는다.
4. `EVENT_REF`의 선택 episode embedding은 pointer 선택 직전 state에만 의존한다. forward labels와 입력 embedding의 episode ID가 어긋나면 fail-fast한다.
5. 긴 세션 carry batch는 시간순으로 같은 rank에서 처리한다. 세션 shuffle·padding·DDP uneven batch·resume 시 memory/KV/lane/episode/RNG를 함께 검증한다. 세션 사이 memory가 섞이면 실패다.

Liger는 기존 lexical CE 경로를 최대한 유지하되 pointer·memory head CE는 별도 계산한다. custom embedding/state 때문에 실제 적용 가능한 kernel 경로·loss normalization을 확인한다. custom cache를 사용한 TBPTT와 gradient checkpointing 호환성도 작은 실험으로 확인하고 미검증 속도를 약속하지 않는다.

HF config에는 `speaker_representation=local_lane_memory`, R, embedding dim, prototype cap, memory cap, episode cap, registry version, event mode, SEG_END/allocator version을 저장한다. 기본 E2를 불러온 뒤 새 모듈/토큰만 초기화하며 frozen E2 encoder가 save/load 후 원 Nemotron으로 바뀌지 않는지 검사한다. 대안 모델을 기존 `stream_decode`로 조용히 실행하지 않도록 config 호환 검사를 둔다.

### 9.1 신규 모듈 제안

| 위치 | 책임 |
|---|---|
| `vapasr/data/lane_interleave.py` | 정답 lane/episode·구조 토큰·pointer sidecar |
| `vapasr/hf/lane_state.py` | lane generation·SEG_END·episode lifecycle |
| `vapasr/hf/speaker_memory.py` | 공유 matcher·NEW/unknown·prototype·revision |
| `vapasr/hf/episode_pointer.py` | 가변 후보 reference scoring·입력 embedding |
| `vapasr/hf/memory_heads.py` | 공유 activity/future/hazard |
| `vapasr/hf/dialogue_model.py` | chunk 순차 forward·loss·runtime parity |
| `experiments/p2_memory_probe.py`, `p2_train_memory.py` | 작은 실험·Trainer 실행 |
| `tests/test_lane_memory_protocol.py` | 아래 필수 fixture |

이 파일들은 **구현 예정**이다. 기존 mono 경로는 대조군으로 보존하며 Dataset의
`episode_of_token, pointer_target, memory_valid_mask, task_masks, state_snapshot_version`을 명시적으로 추가한다. 수치 `speaker_17`을 고정 vocabulary ID로 되돌려 넣지 않는다.

## 10. 단계별 실행과 통과 기준

고정 K baseline과 대안이 같은 corpus·split·노출 시간·TN·δ·가능한 동일 초기 backbone을 사용하도록 한다. 실제 단계 완료는 코드/결과가 있을 때만 기록한다.

| 단계 | 작업 | 완료 증거 / 중단 조건 |
|---|---|---|
| D0 프로토콜 | oracle lane·episode·pointer serializer/parser | 무음·겹침·재사용·지연 EOT round-trip, 소유자 오류 0 |
| D1 local 전사 | R=4 lexical/ONSET/SEG_END, 일단 memory 매칭 제외 | 실제 71631·AMI Dataset, 32창 overfit, R 초과/조기 SEG_END/삭제 계수 |
| D2 speaker probe | 고정 backbone의 z·prototype·shared matcher | 예측 lexical/lane 기준 새/기존·재등장 성능. oracle만 좋으면 joint 확대 보류 |
| D3 동적 통합 | causal memory carry + episode-pointer EOT | lane 재사용 후 event 오귀속, unknown/수정 지연, C 조건 검사 |
| D4 미래 예측 | memory-conditioned activity/VAP/hazard | 고정 K와 같은 FPR/지연 평가, N 증가에 따른 비용·calibration |
| D5 장문 | N 확장·세션 재시작·고정 장치 runtime | 30–60분 자유실행, memory 오염·ID switch·backlog·resume |

D0의 가상 fixture만 끝났다고 D1 실제 데이터 검증을 건너뛰지 않는다. 현재 정본의 71631 실물 경로가 없다면 그것을 먼저 공유 기반으로 만든다. D1/D2가 실패하면 EOT/UI/미래 head 인프라를 더 쌓지 말고 local ASR 또는 speaker embedding 병목을 해결한다.

### 10.1 최소 실험 행렬

- **고정 K 기준선:** 해당 평가 세션을 수용하는 K. K 초과 실패도 별도 보고.
- **local R + oracle speaker mapping:** 표현/ASR 가능성 상한. 제품 결과 아님.
- **local R + learned memory:** 실제 후보. oracle count 금지.
- **local R + memory + EOT pointer:** event 회귀와 비용 분리.

데이터 축은 `N=2/4/8(학습), N=12/16(일반화 시험 제안)`과 `S=1/2/3/4`를 분리한다. N이 큰 자연 자료가 부족하면 합성임을 명시한다. 실제 3+ 겹침이 없는 자료로 그 성능을 주장하지 않는다. 짧은 맞장구·비슷한 음색·음량차·60초 이상 침묵 후 재등장·새 사람이 여러 번 연속 오는 조건을 포함한다.

첫 person prototype을 충분히 만들기 전에 모든 사람이 겹쳐 시작하는 hard case도 평가한다. 긴 깨끗한 enrollment가 있는 조건은 별도 oracle/easy subset이다. 라이선스·동일 원 대화 중복·train/test 화자/녹음 노출은 기존 DB 계약을 따른다.

## 11. 측정·채택 관문

아래는 실험 전 동결할 **초기 제안 관문**이다. 구현 후 결과를 보고 유리하게 바꾸지 않는다.

- **프로토콜/인과성:** 잘못된 episode 참조·미래 memory·세션 간 state 누출·stale generation 귀속 0. runtime 오류를 gold로 고쳐 숨기지 않는다.
- **인식 보존:** 고정 K와 동일 수용 범위에서 EN WER/KO CER 및 speaker-attributed 오류 상대 회귀 ≤5%를 제안한다. lane oracle와 실제 memory 결과를 함께 낸다.
- **확장 이득:** N>K 구간에서 제외 없는 화자 recall·오귀속 개선, N≤K에서 성능 보존. 가변 모델이 표현 가능한 것만으로 승리라고 하지 않는다.
- **메모리 품질:** false-new, false-known merge, 재등장 ID 오류, unknown, 최초 ID 확정 지연, revision 횟수를 N/S/증거 길이별 보고한다. provisional 비율을 높여 오귀속만 낮추는 해법은 채택하지 않는다.
- **이벤트:** C-offset 지연과 증거 가용 후 추가 지연, wrong-speaker EOT, 중복/누락, episode reference 오류를 분리한다. prototype 오류와 pointer 오류를 같은 항목으로 합치지 않는다.
- **실시간:** end-to-end RTF<1, backlog 비발산 필수; tick p99<80ms 지향. N=4/16/64에서 matcher+heads 비용을 포함한다. C의 3초 기다림을 compute 지연으로 빼거나 과거 시각으로 채점하지 않는다.

구조화 출력은 `session_id, lane_id, generation, episode_id, speaker_id|null, assignment_status, token_ids/text, audio_seen_until, emitted_at, event_mode, emission_source, revision`을 포함한다. 최초 출력과 수정 후 평가를 따로 낸다. cpWER/DER는 speaker별 regrouping 뒤 비교하며 virtual lane를 사람 ID로 채점하지 않는다.

실행 시간은 D1/D3의 100–300 step으로 측정한다. 단일 forward의 기존 throughput을 재사용하지 않는다. chunk 순차 루프가 과도하게 느리면 먼저 state·loss 정확성을 유지하는 batching/rollout 근사를 비교한다. GPU job 제출·수일 학습 예산은 이 실측 이후 결정한다.

## 12. 필수 회귀 fixture

1. 시작 무음은 AUDIO/NEXT만, 가짜 episode 없음.
2. 한 사람 발화→SEG_END→재개가 같은 global ID로 연결됨.
3. 두/세/네 명 겹침에 서로 다른 lane, 같은 audio token 하나.
4. 첫 단어 이전 ONSET은 provisional이며 미래 speaker prototype 없음.
5. lane 재사용 후 이전 episode EOT가 옛 소유자에게 붙음.
6. 본인 재개와 상대 시작이 C horizon 안에 있으면 event unknown.
7. SEG_END 조기 출력 뒤 잔여 lexical을 삭제하지 않고 오류 기록.
8. C EOT 없음/unknown이어도 lexical 종료 lane은 재사용 가능.
9. R 초과·episode cap·memory cap을 덮어쓰기 없이 계수.
10. 같은 lane generation을 재사용한 stale reference를 거부.
11. 메모리 permutation과 sidecar를 함께 바꾸면 동등 출력/손실.
12. 동일 prefix·다른 미래 suffix에서 현재 입력/state는 동일.
13. 무음인 기존 화자의 future/hazard query를 유지, 미등장 ID는 mask.
14. padding·미라벨 event·pointer 없는 위치에서 손실 오염 없음.
15. checkpoint 재개·불균등 DDP·세션 종료/중간 crop/EOF carry parity.
16. 새 protocol을 legacy mono decoder로 잘못 로드하면 명시적 실패.

## 13. 근거와 남은 가장 큰 위험

- [t-SOT §2·Appendix A](https://arxiv.org/html/2202.00842v5): virtual channels와 세션 사람 ID는 다르며 M 채널 일반화를 제안한다. 여기의 SEG_END·episode pointer는 그 논문 규약이 아니라 우리 확장안이다.
- [t-vector §3·4](https://arxiv.org/pdf/2203.16685): token-conditioned speaker 표현의 선행 근거다. 논문 일부 SD 평가의 oracle speaker count를 실제 unknown-N 성능 보증으로 쓰지 않는다.
- [Streaming Sortformer](https://arxiv.org/html/2507.18446v1): speaker cache·공유 시간축 참고. 해당 논문의 4명 상한을 제거하는 구현을 제공한다는 뜻은 아니다. 모두 확인일 2026-09-14.
- [[output-phase2-streaming-asr-diarization-plan]]: 유지할 mono·TN·80ms·C-mode·관측 mask·ASR guardrail.

가장 큰 위험은 **(a) 겹침에서도 source-conditioned z가 충분히 분리되는가, (b) 예측 SEG_END의 오류가 lane/memory 전체를 오염시키는가, (c) 동적 state 때문에 학습/추론 비용이 지나치게 커지는가**다. 따라서 첫 투자는 새 head 전체가 아니라 **D0 프로토콜 round-trip과 D1/D2의 작은 실제 데이터 실험**이다.
