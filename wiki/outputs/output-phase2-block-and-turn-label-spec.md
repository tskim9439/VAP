---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 블록·라벨 규약 — 무음·단독·겹침·지연 전사·화자별 start/end_of_turn, 상태 전이·정답 생성·학습·평가
sources:
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[source-muse-voice-transcribe]]'
  - '[[output-vap-target-pipeline]]'
  - '[[turn-taking-objectives]]'
  - '[[output-phase2-critique-response]]'
---

# Phase 2 블록 구성과 화자별 Turn 토큰 라벨 규약

2026-09-11 사용자 추가 요구 반영. 정본 계획 v1.2의 상세 계약이다. **설계 명세이며 실제 코퍼스 라벨 생성·모델 구현·학습 완료를 뜻하지 않는다.** 아래 예시가 라벨 생성기·serializer·decoder 회귀 테스트의 기준이 된다.

## 1. 변경 사항과 Muse에서 참고한 범위

80ms마다 audio token 하나를 넣고 모델이 전사·화자·턴 이벤트·`<NEXT_AUDIO>`를 같은 autoregressive vocabulary에서 예측한다. activity/VAP head도 유지한다. 기존 ‘상태 토큰은 선택적 후속 실험’에서 **화자별 `<start_of_turn>`·`<end_of_turn>` 생성은 필수 출력**으로 변경한다.

Muse 공식 설명은 diarization용 `<|start_of_turn|>`·speaker tag와 endpointing용 `<|speech_onset|>`·`<|speech_endpoint|>`를 소개한다. 같은 화자도 여러 start segment를 가질 수 있다. 공개 설명에 정확히 `<end_of_turn>`이라는 토큰이나 본 문서의 2화자 결합 문법·라벨 생성 세부는 제시되지 않았다. 따라서 **전사와 이벤트의 공동 생성 원칙을 참고한 우리 규약**으로 명시한다. [Meta 공식 설명, 2026-09-11 확인](https://research.meta.ai/blog/introducing-muse-voice-transcribe)

Muse는 empty audio 이후 NEXT 없이 잔여 텍스트를 낸다고 설명하지만, 우리는 E2의 bounded flush + NEXT 규약을 유지한다. 무음과 EOF도 구분한다.

## 2. 두 개의 축: 현재 소리와 아직 닫히지 않은 발화

`activity[k]=[A,B]`는 **현재 80ms 안에 실제 speech가 있는가**다. `turn_state[s]`는 화자 s의 생성 스트림에서 현재 contribution이 열려 있는가다. 둘은 같지 않다.

| 현재 activity | 가능한 turn 상태 | 해석 |
|---|---|---|
| `[0,0]` | 둘 다 CLOSED | 대화 시작 전 또는 두 사람 모두 발화를 마침 |
| `[0,0]` | A OPEN, B CLOSED | A가 생각 중 잠시 쉼; A 종료 토큰을 아직 내지 않음 |
| `[1,0]` | A OPEN, B OPEN도 가능 | A가 말하고, B는 잠시 쉬거나 B의 지연 전사/종료 처리가 남음 |
| `[0,1]` | A OPEN, B OPEN도 가능 | B가 말하지만 A가 맞장구 뒤 계속할 수 있음 |
| `[1,1]` | 둘 다 OPEN이 일반적 | 겹침. 한 decoder여도 두 화자의 contribution은 동시에 열림 |

여기서 turn은 **화자별로 시작과 종료를 가진 기여 단위(contribution)**다. 짧은 맞장구도 B의 기여이며 A의 floor를 가져온다는 뜻은 아니다. A가 끊겨 미완결로 종료해도 EOT는 가능하다. `종료 사유=완결/중단/불확실`과 `floor holder`는 별도 라벨/헤드에 둔다. ‘문장이 완결됐다’와 ‘이 화자의 기여를 지금 닫아도 된다’를 같은 정답으로 쓰지 않는다.

## 3. Vocabulary와 기본 문법

| 토큰/출력 | 역할 | 생성 여부 |
|---|---|---|
| `[AUDIO_k]` | 현재 mono 80ms를 인코딩한 soft token | 시스템 입력, 예측 타깃 아님 |
| `<SPK_A>`, `<SPK_B>` | 뒤 payload가 누구의 것인지 선택 | 모델 생성 |
| `<start_of_turn>` | 선택 화자의 새 contribution 시작 | 모델 생성, 해당 기여당 한 번 |
| lexical token | 선택 화자의 전사 | 모델 생성 |
| `<end_of_turn>` | 선택 화자의 현재 contribution 종료 확정 | 모델 생성, 해당 기여당 최대 한 번 |
| `<NEXT_AUDIO>` | 이번 청크의 출력 라운드를 마치고 다음 오디오를 받음 | 모델 생성, 청크당 한 번 |
| `<EMPTY_AUDIO>` | 실제 입력 종료 뒤 잔여 출력을 처리 | 시스템 입력, 무음 대체 아님 |
| activity/VAP | 현재 활동/미래 활동의 조밀한 확률 | audio 위치에서 head 예측, 문자열 토큰 아님 |

`<start_of_turn_A>`처럼 화자별 vocabulary를 늘리지 않고 `<SPK_A><start_of_turn>`, `<SPK_A><end_of_turn>`로 귀속한다. 모델이 선택한 token ID를 parser가 해석한다. 두 화자의 `turn_id`는 각자의 start 토큰에서 증가시키는 API 상태이며 gold ID를 모델에 주지 않는다.

```text
prefix    기존 E2 prefix + <DELAY_δ>
chunk k   [AUDIO_k] (A payload)? (B payload)? <NEXT_AUDIO>
payload   (<SPK_s> 필요 시) (start / lexical / end 이벤트의 화자 내부 순서열)
```

G1은 청크 안에서 A payload → B payload 순서다. **payload가 없는 화자의 빈 블록은 만들지 않는다.** 현재 화자가 직전 출력의 소유자와 같으면 selector를 생략할 수 있으나, 모든 turn marker 앞에는 명시적 `<SPK_s>`를 요구하는 v1.2 규칙으로 이벤트 귀속을 분명히 한다. 예: `안녕 <SPK_A><end_of_turn>`. 태그가 더 늘므로 v1.1의 ‘청크당 최대 태그 2개’는 더 이상 전체 토큰열의 상한이 아니다.

행동이 없는 `<SPK_A><SPK_B>`는 불허하지만 **텍스트 없이 이벤트만 있는 화자 블록은 허용**한다. 예: `[AUDIO_k]<SPK_A><end_of_turn><NEXT_AUDIO>`. EOF까지 B가 나타나지 않는 1화자 세션에는 B 태그·B start/end를 만들지 않는다.

## 4. 경우별 블록과 감독 신호

아래 문자열은 읽기 쉬운 lexical 예시다. 실제 tokenizer의 여러 subword/byte 조각은 라벨 생성 시 확정한다. 예시의 ‘출력 있음’은 **현재 발화 여부가 아니라 방출 예정 전사/이벤트가 있음**을 뜻한다. 모든 행에서 `[AUDIO_k]`는 실제 입력이고 activity/VAP는 별도 타깃이다.

| 상황 | 청크의 정답 블록 예시 | activity | 핵심 학습 |
|---|---|---|---|
| 시작 전 무음 | `[AUDIO_k] <NEXT_AUDIO>` | `[0,0]` | 대화/화자/턴을 지어내지 않음 |
| A 첫 speech, 아직 전사 없음 | `[AUDIO_k] <SPK_A><start_of_turn> <NEXT_AUDIO>` | `[1,0]` | start를 단어 완료 전에도 생성 가능 |
| A만 말함, 전사 준비됨 | `[AUDIO_k] <SPK_A>안녕하세요 <NEXT_AUDIO>` | `[1,0]` | B 블록 없음, A 귀속 전사 |
| A만 말하지만 이번 청크 토큰 없음 | `[AUDIO_k] <NEXT_AUDIO>` | `[1,0]` | NEXT는 무음/턴 종료가 아님 |
| A 발화 중간 짧은 pause | `[AUDIO_k] <NEXT_AUDIO>` | `[0,0]` | A OPEN 유지, 조기 EOT를 억제하는 negative event 학습 |
| A pause 중 지연된 단어 도착 | `[AUDIO_k] 그래서 <NEXT_AUDIO>` | `[0,0]` | 무음에서도 지연 전사 가능; 이전 소유자 A 가정 |
| A 종료 결정, 새 텍스트 없음 | `[AUDIO_k] <SPK_A><end_of_turn> <NEXT_AUDIO>` | `[0,0]` | 이벤트-only 블록, A만 닫음 |
| A 마지막 전사와 종료가 함께 도착 | `[AUDIO_k] 끝입니다 <SPK_A><end_of_turn> <NEXT_AUDIO>` | `[0,0]` | lexical drain 뒤 EOT, 이전 소유자 A 가정 |
| EOT 이후 긴 무음 | `[AUDIO_k] <NEXT_AUDIO>` 반복 | `[0,0]` | EOT 반복 생성 금지, 시간은 계속 흐름 |
| A 종료 후 B 시작 | `[AUDIO_k] <SPK_B><start_of_turn> 네 <NEXT_AUDIO>` | `[0,1]` | A/B 정체성 유지, B 새 기여 시작 |
| A 계속 + B 첫 맞장구 | `[AUDIO_k] <SPK_A>설명하면 <SPK_B><start_of_turn> 응 <NEXT_AUDIO>` | `[1,1]` | 둘 다 OPEN, B 시작으로 A를 닫지 않음 |
| B 맞장구 종료, A 계속 | `[AUDIO_k] <SPK_A>이렇습니다 <SPK_B><end_of_turn> <NEXT_AUDIO>` | `[1,0]` | B만 닫음, A floor 유지 가능 |
| 두 사람 지속 겹침 | `[AUDIO_k] <SPK_A>내 생각은 <SPK_B>잠깐만 <NEXT_AUDIO>` | `[1,1]` | 두 전사 모두 보존, 두 start를 반복하지 않음 |
| 현재 B만 발화, A 지연 전사/종료도 도착 | `[AUDIO_k] <SPK_A>알겠어 <SPK_A><end_of_turn> <SPK_B>그러면 <NEXT_AUDIO>` | `[0,1]` | 현재 VAD로 A의 지연 payload를 막지 않음 |
| 같은 청크에 두 화자 종료 | `[AUDIO_k] <SPK_A><end_of_turn> <SPK_B><end_of_turn> <NEXT_AUDIO>` | `[0,0]` | 두 독립 EOT; 전역 EOS 아님 |
| A가 끝낸 뒤 A가 새로 시작 | `[AUDIO_k] <SPK_A><start_of_turn> 그리고 <NEXT_AUDIO>` | `[1,0]` | 화자 동일, turn_id만 새로 증가 |
| 겹침인데 이번 청크 lexical/event 없음 | `[AUDIO_k] <NEXT_AUDIO>` | `[1,1]` | activity는 둘 다 1, 생성 출력은 NEXT뿐일 수 있음 |
| 기침·잡음만 존재 | `[AUDIO_k] <NEXT_AUDIO>` | speech 기준 `[0,0]` | 음향 에너지 존재만으로 전사/start 생성하지 않음 |

같은 사람이 pause 뒤 이어 말하면 기존 OPEN을 유지한다. 이미 EOT를 확정한 뒤 말을 재개하면 새 start/turn_id로 기록한다. 이때 앞선 EOT가 적절했는지는 reference 기준 false endpoint로 평가한다. parser가 과거 EOT를 지워 정답처럼 고치지 않는다.

## 5. 화자별 상태 머신과 모델의 역할

내부 최소 상태는 `speaker_seen[s]`, `turn_open[s]`, `turn_id[s]`, 전사 버퍼와 현재 payload 소유자다. 음향 활동과 OPEN은 독립이다. pause는 OPEN+activity=0으로 표현하며 매 청크 `<hold>` 토큰을 추가하지 않는다.

| 현재 상태 | 예측 | 적용 |
|---|---|---|
| CLOSED | `<SPK_s><start_of_turn>` | OPEN, turn_id 증가 |
| OPEN | lexical token | 해당 화자·turn 버퍼에 append |
| OPEN | `<SPK_s><end_of_turn>` | CLOSED, 그 turn transcript final 이벤트 |
| OPEN 또는 CLOSED | `<NEXT_AUDIO>` | 라운드 종료, turn 상태는 유지 |
| B의 start/end | 어떤 A 상태든 | A 상태에 자동 변화 없음 |

문법 mask는 중복 start·닫힌 turn의 end·소유자 없는 이벤트·invalid special을 차단한다. **정답 VAD, 남은 gold 단어 수, 미래 turn 종료 여부로 추론 mask를 만들지 않는다.** 같은 청크에서 A의 옛 turn 전사 → A EOT → A 새 start → A 새 전사가 필요하면 그 순서로 동일 A payload 안에 넣는다. 이를 위해 학습 타깃은 화자별 turn ID와 의존 순서를 갖는다.

모델이 실제로 언제 start/end를 낼지는 CE로 학습한 logits가 결정한다. 규칙 기반 무음 timeout은 대조군으로만 두며 모델의 EOT 토큰을 생성한 것처럼 대신 삽입하지 않는다. activity와 EOT가 충돌하면 둘 다 로그에 남겨 평가한다. 현재 활동이 1이어도 이전 turn의 지연 EOT가 올 수 있다.

## 6. 라벨 생성: acoustic 후보 → contribution → 생성 타깃

### 6.1 중간 레코드

대화별 아래 레코드를 버전 관리한다. 숫자 파라미터·teacher·prompt·tokenizer·TN·채널 매핑·annotation version을 함께 저장한다.

```text
conversation_id, speaker_slot, turn_id
speech_segments[]                 # 원 채널 VAD, 물리적 speech 구간
turn_start_ref_s, turn_end_ref_s   # 검수된 기여 경계; 끝이 불명확하면 null
tokens[{id, end_ref_s, align_quality}]
end_reason                       # complete / interrupted / other / uncertain
label_source                     # human / validated_pseudo / heuristic
start_observed_until_s, end_observed_until_s
label_confidence, onset_mask, eot_mask, observed_until_s
```

`end_observed_until_s`는 끝이라는 판단을 학습시키는 데 사용한 **오디오 prefix의 끝**이다. `turn_end_ref_s`와 다르다. suffix가 있는 annotation으로 reference 사건을 확인할 수 있지만 그 미래를 이미 들은 것처럼 사건 시각에 모델 출력을 소급 배정하지 않는다.

### 6.2 정답 생성 순서

1. 분리 채널에서 VAD·정렬을 만들고 실제 화자와 매핑한다. 모델 입력은 mono다. VAD의 짧은 무음은 IPU 후보 경계일 뿐 EOT 확정이 아니다.
2. 실제 대화 문맥과 prefix 의미 완결성·운율을 검토해 같은 화자의 IPU를 contribution으로 묶는다. 같은 화자 유지, terminal overlap, 맞장구, 중단을 구분한다. 웃음/기침과 주석 누락도 분리한다.
3. **인식 가능한 시작**에 start 후보, **기여 종료가 확인된 지점**에 EOT 후보를 만든다. B의 onset이나 A의 VAD offset만으로 A EOT를 자동 확정하지 않는다. B 맞장구의 EOT는 B 기여만 닫는다.
4. boundary 후보 뒤 0/80/160/320/640ms prefix를 검토하는 파일럿에서 onset/end decision 시각을 정한다. 이는 초기 후보 그리드이며 고정 무음 timeout 규칙이 아니다. 텍스트 LLM만으로 운율 기반 판단 시각을 확정할 수 없으므로 audio 검수 표본으로 품질을 측정한다.
5. 사람 또는 검증된 pseudo contribution만 event supervision에 사용한다. `uncertain`, 잘린 발화, 채널 오류는 event supervision에서 제외한다. label 개수·coverage·마스킹 비율을 언어/유형별 보고한다.
6. 원 token 시각과 event decision 시각을 §7의 청크·의존 순서로 투영한다. G1으로 화자별 payload를 만들고 NEXT를 정확히 한 번 붙인다. decoder/parser로 재생하여 turn_id별 전사가 round-trip되는지 확인한다.

기존 `derive_events()`의 SHIFT/HOLD/INT/BC는 후보 탐색에 재사용할 수 있으나 이미 검수된 EOT gold는 아니다. LibriSpeech의 파일 끝·마침표·Kspon의 전사 행 끝도 EOT 정답으로 자동 변환하지 않는다. [[output-vap-target-pipeline]]

### 6.3 라벨이 부족한 데이터

합성 겹침·ASR replay는 전사/activity에는 사용하지만 semantic EOT를 꾸며 넣지 않는다. 이벤트가 주석되지 않은 창에서 EOT가 없는 것을 negative label로 학습하면 안 된다. 첫 joint event 학습은 **시작·끝·pause·negative가 완전히 주석된 자연 대화 창**을 별도 batch로 사용한다.

unlabeled batch는 ASR 보조 손실을 사용한다. 이때 turn-marker logits를 lexical/NEXT CE 정규화 집합에서 제외하는 partial-supervision 경로를 명시해 ‘EOT 없음’을 정답으로 강제하지 않는다. event-labeled batch에서는 event를 포함한 전체 유효 vocabulary로 CE를 계산한다. 이 방법은 일반적 event marginalization과 동등한 완전 likelihood는 아니므로 joint/ASR batch 혼합 비율과 calibration을 검증한다. 단순히 삽입 예정 event의 label만 `-100`으로 두고 다른 NEXT에서 EOT를 계속 벌주는 구현은 금지한다.

시작·종료의 신뢰도가 일부만 확보된 창은 우선 event batch에서 제외하고 ASR-only로 보낸다. 라벨 coverage가 높아진 뒤 프레임별 부분 감독을 확장한다. 합성 reference start/end를 디버깅 fixture로 쓸 수 있지만 자연 의미 EOT 성능에 합산하지 않는다.

## 7. 방출 시각과 전사·이벤트 순서

시간 계산은 가능한 한 sample 정수로 하고 부동소수점 경계 오차를 테스트한다. `Δ=0.08`, `b_k=(k+1)Δ`이며 실제 encoder 오디오 가용 시각이 다르면 그 값을 별도로 기록한다.

- lexical: 기존 `k_txt=floor(t_end/Δ)+δ_text` 유지.
- start: 검수된 `start_observed_until_s`를 모두 관측한 최초 청크에 배정. 고정 δ_text를 기계적으로 더하지 않는다. 아직 어떤 화자인지 불명확하면 식별 가능 시각까지 지연되고 latency에 포함된다.
- EOT: `k_eot=max(k_end_decision, k_last_text_of_turn, k_start)`로 배정하고 같은 청크에서 마지막 lexical 뒤에 둔다. `k_end_decision`은 `b_k ≥ end_observed_until_s`인 최초 청크다. **last lexical 이후 종료**라는 출력 계약 때문에 생긴 추가 지연을 보고한다.
- 동일 화자의 다음 start는 이전 EOT 이후의 시퀀스 위치를 가져야 한다. 필요하면 다음 start 및 해당 turn lexical도 뒤로 밀고 추가 지연을 기록한다. teacher에서만 의존성을 적용하고 추론에서 gold를 확인하지 않는다.
- 다른 화자는 독립적이다. A의 EOT 대기 때문에 B의 speech/전사를 막지 않는다. G1의 A→B 계산 순서 비용만 별도 잰다.

예: A 마지막 단어 끝이 1.12s, δ_text=2면 k=16에서 1.36s에 목표 방출된다. 종료를 판단 가능한 prefix가 1.28s이면 EOT도 k=16에서 마지막 단어 뒤에 놓인다. 판단 prefix가 1.60s까지 필요하면 k=19(1.60s)에 EOT를 둔다. 끝 시각 1.12s로 출력을 소급하지 않고 semantic decision 지연과 transcript drain 지연을 분리한다.

처음 보는 EOT는 그 시점까지 생성한 해당 turn transcript를 확정한다. 추론은 누락된 gold 단어를 모르므로 너무 이른 EOT로 빠진 단어는 실제 삭제 오류로 남는다. 자동 ‘완료 검사’로 제거할 수 있는 문제가 아니다. 반대로 미래 교대 확률은 VAP/head가 last lexical 전에 제공할 수 있다. **end 토큰의 확정과 미래 turn 예측을 구분**한다.

## 8. 무음·EOF·잘림·backlog

- **무음:** 실제 무음 PCM도 encoder를 통과하고 `[AUDIO_k]`를 유지한다. 둘 다 조용하고 pending 출력이 없으면 NEXT만 생성한다. 과거 전사·EOT가 지연되어 도착하면 무음 청크에서 출력한다. `<SILENCE>`와 `<EMPTY_AUDIO>`로 대체하지 않는다.
- **아직 B가 없는 단일 화자:** activity의 B target은 음성이 실제 없는 한 0이지만 미래 신규 B 가능성을 모든 세션에서 0으로 고정하는 전역 정책은 없다. start/end는 A에만 붙인다. B는 관측 후 새 슬롯으로 등장한다.
- **자연 pause:** A OPEN을 유지한 `[AUDIO] NEXT`를 여러 번 학습한다. 200ms보다 긴 pause도 HOLD일 수 있고 짧은 무음도 endpoint일 수 있도록 길이·완결성 교차 표본을 둔다.
- **crop 끝:** 데이터 crop이 끝났다고 EOT를 추가하지 않는다. carry 또는 warm prefix로 OPEN 상태를 복원한다. prefix가 없는 mid-turn crop은 event 학습에서 제외하고 lexical-only로 쓴다. future label 구간이 잘리면 censor/mask한다.
- **실제 EOF:** `<EMPTY_AUDIO>`는 transport 종료다. 알려진 결정 prefix가 실제 관측 오디오 안에 있던 pending lexical/EOT만 bounded flush한다. 미완료 OPEN은 API `stream_closed, truncated=true`로 남기며 EOT를 강제하지 않는다. padding zero를 실제 추가 무음 증거로 사용하지 않는다.
- **cap:** lexical·selector·start/end 예산과 전체 safety cap을 따로 기록한다. 골드 target은 충분한 cap에서 생성해 누락을 없앤다. decoder는 selector를 내기 전에 selector+최소 payload를 낼 2자리 여유를 확인해 `<SPK_A><NEXT_AUDIO>` 같은 미완성 블록을 만들지 않는다. 선택한 turn marker와 직전 selector는 하나의 문법 단위로 완결한다. cap 때문에 다음 라운드로 밀린 event·누락 end를 별도 집계하고 cap/NEXT로 EOT를 대체하지 않는다. 무제한 재귀나 무제한 flush는 허용하지 않는다.

실제 녹음에 종료 후 무음이 포함된 자연 endpoint 사례도 학습해야 한다. 모든 파일이 발화 직후 잘려 있으면 `<EMPTY_AUDIO>`만 보고 EOT를 내는 지름길을 배울 수 있다.

## 9. 교사 강제 시 실제 학습 타깃

예시는 A가 이미 OPEN이고 마지막 단어 ‘네’가 하나의 lexical 단위인 경우다. 실제 subword 분할이면 같은 shift를 각 토큰에 적용한다.

```text
sequence: [AUDIO_k] <SPK_A> 네 <SPK_A> <end_of_turn> <NEXT_AUDIO> [AUDIO_k+1]
predict :             SPK_A  네   SPK_A      end_of_turn      NEXT       (audio 입력, loss 없음)
```

정확한 학습은 causal next-token shift다. `[AUDIO_k]` 위치 hidden state는 `<SPK_A>`를 맞히고, ‘네’ 뒤에는 event selector, selector 뒤에는 EOT, EOT 뒤에는 NEXT를 맞힌다. prefix/audio/EMPTY 자체는 예측 label에서 제외하지만 **audio 위치가 NEXT/텍스트/event를 예측하는 loss까지 지우지는 않는다.**

무출력 블록 `[AUDIO_k]<NEXT_AUDIO>`에서는 audio hidden state가 NEXT를 맞히는 CE를 받는다. 같은 audio 위치에서 activity BCE도 받으므로 `[1,0] + NEXT`, `[0,0] + NEXT`, `[1,1] + NEXT`를 구분해 학습한다. labeled pause 창의 NEXT는 너무 이른 EOT를 억제하는 negative supervision이고, 종료 창에서는 EOT CE가 positive supervision이다.

`L_total = L_AR(weighted lexical/selector/NEXT/start/end) + λ_activity L_activity + λ_VAP L_VAP + λ_semantic L_semantic (+ λ_hazard L_hazard)`

초기 AR 가중 후보는 lexical/selector/start/end=1, NEXT는 E2의 EN 0.3/KO 0.15다. 희귀 end 누락 시 start/end 가중 {1,2,4}를 동일 exposure에서 비교한다. 이벤트 sample oversampling을 사용하면 자연 빈도의 dev에서 calibration·FPR를 측정한다. activity는 유효 청크 평균, VAP는 유효 미래 horizon 평균으로 별도 정규화한다.

학습 초기는 fully labeled 32창 overfit와 이벤트 CE를 먼저 검증하고, 자연 대화 확장 뒤 실제 생성 **텍스트+selector+start/end history**로 적응한다. teacher-only의 완벽한 turn state를 추론에 제공하지 않는다. 위 state machine의 false early end, missing start, wrong speaker에 대한 오류 복구도 자체 예측으로 시험한다.

## 10. 새 단계·산출물·평가

| 단계 | 추가 작업 | 완료 확인 |
|---|---|---|
| Q0 | token registry·라벨 중간 schema·18개 경우 fixture·per-speaker parser·시각 scheduler | round-trip, EOF/pause negative, no-gold/no-future 의존성 |
| Q1 | 신뢰도 높은 비중첩 자연 contribution의 start/end labels, fully labeled 32창 overfit → joint | 단독 A·단독 B·교대·pause에서 이벤트와 전사 일치 |
| Q2 | 겹침·맞장구·중단의 독립 turn label와 두 OPEN 상태 | A 지속 중 B start/end, 두 EOT 동시, 지연 lexical 보호 |
| Q3 | 의미 S 트랙·VAP 결합·자체 생성 event history 적응 | 의미 기여 및 endpoint quality-latency curve |
| Q4 | live API에 speaker/turn_id별 start/end, 장문·cap·EOF | p99·연속 ID·중복/누락 이벤트·truncation 검증 |

필수 파일 제안: `vapasr/data/dialogue_turn_labels.py`(후보·quality mask), `dialogue_interleave.py`(target), `vapasr/hf/dialogue_state.py`(generated event parser), `experiments/p2_build_turn_labels.py`(dataset artifacts/QA), `tests/test_dialogue_turn_sequence.py`(fixtures). 기존 tokenizer ID는 보존하고 신규 start/end ID를 저장한다. 새 vocabulary 크기·tied embedding·금지 mask·HF/MLX save/load parity를 확인한다.

현재 `vapasr/data/interleave.py::Specials`에 onset/endpoint 필드가 있어도 실제 start/end event 스케줄링은 구현되어 있지 않다. `SPECIAL_TOKENS`에도 본 문서의 두 turn marker는 없다. 기존 placeholder의 존재를 새 기능 완료로 취급하지 않는다.

기본 보고에는 ASR/DER 외에 **화자별 start/end precision·recall, false-end/min, missed-end, repeated-end, wrong-speaker-end, onset/EOT latency p50/p90/p99**를 추가한다. pause 길이·맞장구·overlap·종료사유별로 나눠 본다. 종료 token은 해당 turn 마지막 lexical 이후이므로 `semantic_decision_at`, `transcript_final_at`, `emitted_at`을 구분한다. last lexical 뒤 EOT를 내기 위해 발생한 비용을 VAP의 선행 예측 지연과 섞지 않는다.

TurnBench EOT와 본 contribution 종료는 label 의미가 완전히 같다고 가정하지 않는다. 공식 task 매핑·화자 역할을 dev 검수 후 고정하고, native per-speaker endpoint 지표와 공식 event 지표를 별도 낸다. 중복·부정확한 early-end를 규칙 필터로 지운 결과만 보고하지 않는다.

**새 완료 기준:** start/end 생성과 이 시나리오별 검증은 선택적 ablation이 아니다. Q1/Q2의 MVP도 이벤트를 포함하고, 의미 기반 미래 turn 예측의 가설 검증은 Q3의 별도 필수 조건으로 남는다.
