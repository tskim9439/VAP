---
type: output
status: superseded
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 통합 초안 — P1–P9 검토 전 통합 기록이며 실행 규약·예산은 개정 정본 v1.1과 검토 답변을 우선 적용
sources:
  - [[output-phase2-streaming-asr-diarization-plan]]
  - [[output-phase2-plan-critique]]
  - [[output-phase2-plan]]
  - [[output-phase1-report]]
  - [[output-interleaved-streaming-slm-architecture]]
  - [[decision-mono-input]]
  - [[output-vap-target-pipeline]]
  - [[turn-taking-objectives]]
  - [[turn-taking-evaluation-protocol]]
  - [[source-turnbench]]
  - [[source-conversation-corpora]]
  - [[source-soulx-duplug]]
  - [[source-muse-voice-transcribe]]
  - [[output-stage2-e2-final-eval]]
---

# Phase 2 최종 계획안: mono 2 화자 스트리밍 ASR · Diarization · Turn-Taking

> **2026-09-11 상태**: 사용자 판단으로 [[output-phase2-streaming-asr-diarization-plan]] 을 기본 계획으로 채택했다. turn 토큰·시퀀스 사례·데이터 생성 파이프라인은 그 문서 §4.4·§5.3 으로 옮겼고(청크 내 직렬화는 정본의 종료 시각 순 교차를 따른다), 이 문서는 일정·데이터 실측·ablation 후보 메뉴로만 참고한다.

이 문서는 정본 계획([[output-phase2-streaming-asr-diarization-plan]])의 계약·관문과 비판([[output-phase2-plan-critique]])의 권고, 실행 요약([[output-phase2-plan]])을 합친 **비판 검토 전 통합 기록**이다. 작성 이후 P1–P9의 조건과 계산을 재검토했다. **실행 시 개정 정본 v1.1과 [[output-phase2-critique-response]]의 정정을 우선 적용한다.** 수치 관문은 Q0에서 기준선을 재고 동결하기 전까지 제안값이다.

2026-09-11 후속 정정: 그룹화는 G0/G1 비교 후보이며 시간순과 동등하지 않다. 의미 완결성을 실제 발화 종료로 자동 정답화하지 않는다. R-E2/R-low 제한 비교, 예측 기반 slot memory 검증, Q1 이전 A* 승격 또는 Q1/Q2 재학습, train 후보 약 301.5h 상한을 적용한다. 아래 ‘KV 2배’, 단계별 1–3시간, 자연 train 248h, test 표본으로 승격하는 문구는 채택하지 않는다. 모델 선택은 dev, test는 고정 보고용이다. 상세 구현·문맥 예산·완료 범위는 개정 정본을 따른다. 이 안내는 동시에 추가된 통합 초안을 보존하면서 잘못된 실행 규칙의 사용을 막기 위한 것이다.

## 0. 한 줄 요약
Phase 1 의 E2(Nemotron 스트리밍 인코더 → adapter → Qwen3-ASR thinker, 80 ms 인터리브)를 바꾸지 않고, **(1) 청크마다 화자별 블록으로 묶은 `<SPK_A/B>` 태그 전사와 블록 안 turn 토큰(`<ONSET>`/`<EOT>`/`<HOLD>`/`<BC>`, Muse 의 onset/endpoint 토큰에 해당)**, **(2) `[AUDIO_k]` 위치 hidden state 위 activity·VAP·hazard 병렬 헤드**, **(3) 인과적 화자 슬롯 메모리**를 더해, 마이크 하나의 혼합 음성에서 두 화자의 전사·겹침·미래 turn 을 한 모델이 낸다. 9 주(2026-09-15 → 11-14) 동안 Q0 데이터 계약 → Q1 태그 전사 → Q2 overlap → Q3 turn 헤드·의미 검증 → Q4 장문·실시간 순으로 진행하고, 실시간 ASR 강화는 병렬 트랙 A 로 돌린다. **MVP 는 Q1 + Q2**(화자 구분·overlap 검출·화자별 전사)이며, "의미 기반 turn 예측" 은 Q3 의 대조군 실험이 통과할 때만 주장한다.

## 1. 목표 · 범위 · 비목표

| # | 사용자 목표 | 산출물 | 달성 판정 |
|---|---|---|---|
| G1 | 실시간 ASR 강화 | E2 대비 δ=2/4 WER·CER 개선, tick p99 < 80 ms | 트랙 A + Phase 1 test 표본 |
| G2 | 2 화자 turn-taking 예측 | 화자별 활동·미래 활동(VAP)·다음 onset hazard, EOT/INT 이벤트 | TurnBench dev 공식 규약(mono 입력 명시), KO 검수 held-out |
| G3 | 의미 이해 기반 | LLM 상태 위 헤드가 audio-only 대조군보다 유의하게 좋음 | §9 대조군 4 종, paired bootstrap |
| G4 | overlap 구분 | activity 헤드 A∧B | overlap 프레임 F1 |
| G5 | overlap 구간 화자별 전사 | 같은 청크의 A 블록·B 블록 | overlap 구간 cpWER/cpCER, 화자 소실률 |
| G6 | 최대 2 명 | 슬롯 2 개 | 제3 화자는 비목표(오류로 계수) |

**비목표**: 3 명 이상, 세션 간 화자 신원, 전사 수정(revision), 파형 분리 출력, 응답 생성·TTS, EN 대소문자·구두점 복원. A/B 는 세션 내 익명 상대 ID(먼저 식별된 화자 = A)다.

## 2. 설계 원칙
1. **입력은 mono 한 채널**([[decision-mono-input]]). 분리 채널은 라벨·정렬·혼합 합성에만.
2. **모델 하나, 출력 클럭 둘**: 희소 출력(텍스트·태그·`<NEXT_AUDIO>`)은 토큰, 밀도 출력(80 ms 마다 activity·VAP·hazard)은 병렬 헤드([[output-interleaved-streaming-slm-architecture]] §4).
3. **인과성**: 시각 t 의 모든 출력은 x≤t 와 이미 방출한 토큰만 쓴다. 헤드 위치는 `[AUDIO_k]` 직후·그 청크 텍스트 생성 전. lookahead 는 이벤트 시각에 접어 넣는다.
4. **단일 화자 입력도 같은 규약**(`<SPK_A>` 만 방출). 모드 토큰을 두지 않는다.
5. **관문은 test 를 보기 전에 동결**하고, 단일 화자 ASR 회귀 ≤ 5 % 상대를 모든 단계의 가드레일로 둔다.
6. **합성 데이터는 전사·activity 학습에만**, turn 손실(VAP·hazard·이벤트)은 자연 대화에만.

## 3. 모델 구조

```
16 kHz mono ──▶ Nemotron 3.5 streaming 0.6B 인코더 [56,0] (해동, LR 1e-5) ──▶ 80 ms × 1024
            ──▶ adapter (E2) ──▶ a_k
            ──▶ [옵션 P3] 화자 슬롯 메모리 m_A, m_B (인코더 특징 EMA, activity 헤드가 단독 활동으로 본 청크에서만 갱신)
                z_k = RMSNorm(a_k + g_k ⊙ W[m_A; m_B]),  g_k zero-init gate
            ──▶ Qwen3-ASR thinker (E2, LR 2e-5)  시퀀스: prefix · [AUDIO_k] <SPK_A> tok… <SPK_B> tok… <NEXT_AUDIO> · …
                 ├─ lm_head: 텍스트·화자 태그·turn 토큰(<ONSET>/<EOT>/<HOLD>/<BC>)·<NEXT_AUDIO>
                 └─ h_audio[k] ──▶ MLP(1024→512) ──▶ activity 2 sigmoid │ VAP 256 softmax │ hazard 6 구간 × 2 화자
```
- 새 파라미터: 특수 토큰 6 개(`<SPK_A>` `<SPK_B>` `<ONSET>` `<EOT>` `<HOLD>` `<BC>`, §4.1) 임베딩·lm_head 행(`<asr_text>` 임베딩 평균으로 초기화), 헤드 MLP ≈ 0.8 M, 슬롯 메모리 게이트 ≈ 0.1 M.
- 억제 목록(`vapasr/hf/modeling_vapasr.py:154`)에서 `<SPK_A/B>` 를 빼고, `save_pretrained` 는 학습 여부와 무관하게 인코더를 항상 저장하며 출처 해시를 config 에 기록한다(동결 E2 를 저장하면 원본 `.nemo` 로 바뀌는 결함 방지).
- 실패 대비 구조(Q2 관문 미달 시): **슬롯 조건 오디오 토큰 2 개** — 슬롯 메모리를 query 로 인코더 프레임에 cross-attention pooling 해 `[AUDIO_k^A][AUDIO_k^B]` 를 만든다(오디오 토큰 25 Hz, KV 2 배). multi-output decoder 는 그다음이다.

## 4. 시퀀스 규약 — 블록 구성 · turn 토큰 · 학습 · 디코딩

### 4.1 토큰 집합
| 토큰 | 뜻 | 위치 | 손실 가중 |
|---|---|---|---|
| `<SPK_A>` `<SPK_B>` | 뒤따르는 토큰의 화자(블록 머리) | 블록 시작 | 1.5 |
| `<ONSET>` | 그 화자가 말을 **시작했다**(VAD onset) | 블록 안, 텍스트 앞 | 1.0 |
| `<EOT>` | 그 화자의 **turn 이 끝났다**(floor 를 넘김) — Muse 의 `\|speech_endpoint\|` 에 해당 | 블록 끝 | 2.0 |
| `<HOLD>` | 말을 멈췄지만 **turn 을 쥐고 있다**(같은 화자가 이어 말함) | 블록 끝 | 2.0 |
| `<BC>` | 이 블록은 **맞장구**(상대 발화 중 짧게, floor 를 가져가지 않음) | 블록 끝 | 2.0 |
| `<NEXT_AUDIO>` `<EMPTY_AUDIO>` `<DELAY_δ>` | Phase 1 과 동일 | — | 0.3 / 0.15(EN/KO) |

Muse 는 `|speech_onset|`·`|speech_endpoint|` 를 시퀀스 안 special token 으로 내고([[source-muse-voice-transcribe]]) 화자는 별도 태그였다. 우리는 **이벤트 토큰이 항상 화자 블록 안에** 있으므로 "누구의" onset/EOT 인지가 블록 머리 태그로 정해진다. INTERRUPT 는 토큰이 아니다 — "상대 활동 중 `<ONSET>` + 상대가 곧 `<EOT>`" 로 이벤트 층에서 유도한다(§7.1).

### 4.2 블록 문법

```
chunk   := [AUDIO_k] block_A? block_B? <NEXT_AUDIO>
block_X := (<SPK_X>)? <ONSET>? text* end?            end := <EOT> | <HOLD> | <BC>
```
- 한 청크에 화자당 블록 최대 1 개, 순서는 **A → B 고정**.
- `<SPK_X>` 는 **직전에 방출된 블록의 화자와 다를 때만** 낸다. 첫 블록은 항상 태그. 빈 블록(태그만) 금지.
- `<ONSET>` 은 텍스트보다 먼저 나올 수 있다(텍스트는 δ 뒤에 오지만 onset 은 δ_on=2 로 고정 지연). 같은 청크에 `<ONSET>` 과 텍스트가 함께 있으면 `<ONSET>` 이 앞.
- `end` 는 VAD 세그먼트(≥0.2 s 침묵으로 구분) 하나당 정확히 하나. 청크 배정은 텍스트와 같은 δ: `k_end = ⌊t_off/80 ms⌋ + δ`. 마지막 단어 토큰(`⌊t_end/80⌋+δ`, t_end ≤ t_off) 뒤에 온다.

### 4.3 시각 규칙과 라벨 출처
| 토큰 | 청크 | 라벨 출처 | 미래 정보 |
|---|---|---|---|
| 텍스트 | `⌊t_end/80⌋ + δ` | 깨끗한 채널 강제 정렬 | 없음(δ 지연) |
| `<ONSET>` | `⌊t_on/80⌋ + 2`(160 ms 고정) | 채널 VAD@50 Hz onset(직전 침묵 ≥0.2 s) | 없음 |
| `<EOT>` / `<HOLD>` | `⌊t_off/80⌋ + δ` | `derive_events`(`vapasr/data/targets.py`): SHIFT·terminal overlap·3 s 무발화 → `<EOT>`, 같은 화자 재개 → `<HOLD>` | **있다**(최대 3 s) — 모델이 예측해야 하는 것이 의도. Muse endpoint 와 같음 |
| `<BC>` | `⌊t_off/80⌋ + δ` | BACKCHANNEL(상대 발화 중 시작, ≤1 s, 상대 계속) | 있다(≤1 s) |
- `<EOT>`/`<HOLD>`/`<BC>` 는 **위원회 예측**이다: 방출 시각(offset + 160–320 ms)에 아직 상대가 시작하지 않았어도 "끝났다/쥐고 있다" 를 의미(문맥·운율)로 판단한다. 이것이 사용자 목표 3 이 시퀀스 안에서 실현되는 자리이고, 병렬 헤드(VAP·hazard)는 그보다 **앞선** 연속 확률(−600 ms … 0)을 준다. 둘은 역할이 다르며 둘 다 TurnBench 규약으로 채점한다(§10).
- 3 s 안에 아무도 말하지 않으면 `<EOT>`(turn 이 침묵으로 끝남). 종료 시점에 상대가 이미 말하는 중이면(방해당함) `<EOT>`(terminal overlap 은 SHIFT 로 이미 처리, 방해당한 경우도 floor 를 잃었으므로 EOT). 맞장구 종료는 `<BC>`.
- 휴리스틱 라벨은 TurnBench dev gold 로 정확도를 검증한 뒤 쓴다(정확도 <0.8 인 유형은 손실 가중 0).

### 4.4 사례별 블록 구성

아래는 δ=2 기준이며 한 줄이 한 청크다. `#` 줄은 설명이다.

**사례 1 — 스트림 시작, 무음**

```
[AUDIO_0] <NEXT_AUDIO>
[AUDIO_1] <NEXT_AUDIO>
# 블록 없음. 손실은 <NEXT_AUDIO> 에만 (가중 EN 0.3 / KO 0.15)
```

**사례 2 — 단일 화자 A 만 말함 (Phase 1 replay 도 동일)**

```
[AUDIO_5]  <SPK_A> <ONSET> <NEXT_AUDIO>
# 첫 블록이라 태그. onset 은 +160 ms 고정 지연
[AUDIO_8]  안녕 <NEXT_AUDIO>
# 같은 화자 → 태그 생략. 텍스트는 단어 종료 + δ
[AUDIO_9]  하세요 <NEXT_AUDIO>
[AUDIO_12] <HOLD> <NEXT_AUDIO>
# 0.3 s 멈춤, 곧 이어 말함 → turn 유지
[AUDIO_16] <ONSET> <NEXT_AUDIO>
# 같은 화자 재개 → 태그 생략
[AUDIO_20] 저는 <NEXT_AUDIO>
[AUDIO_40] 입니다 <EOT> <NEXT_AUDIO>
# turn 종료 (3 s 침묵 또는 B 가 시작)
[AUDIO_41] <NEXT_AUDIO>
# 다시 무음
```

**사례 3 — 화자 교대 (A 끝 → 0.4 s gap → B 시작)**

```
[AUDIO_40] 입니다 <EOT> <NEXT_AUDIO>
[AUDIO_47] <SPK_B> <ONSET> <NEXT_AUDIO>
# 화자 바뀜 → 태그
[AUDIO_50] 네 <NEXT_AUDIO>
# B 계속 → 태그 생략
```

**사례 4 — A 발화 중 B 맞장구 (overlap)**

```
[AUDIO_60] 그래서 <SPK_B> <ONSET> <NEXT_AUDIO>
# A 블록(태그 생략) 다음 B 블록(태그)
[AUDIO_62] <SPK_A> 제가 <SPK_B> 응 <BC> <NEXT_AUDIO>
# 직전 블록이 B 였으므로 A 에 태그. B 는 맞장구로 종료
[AUDIO_63] <SPK_A> 말씀드린 <NEXT_AUDIO>
# 직전 블록 B → A 태그. 이후 A 만 이어지면 생략
[AUDIO_64] 건 <NEXT_AUDIO>
```

**사례 5 — B 가 끼어들어 A 가 멈춤 (interruption)**

```
[AUDIO_70] 그런데 <SPK_B> <ONSET> <NEXT_AUDIO>
[AUDIO_72] <SPK_A> 제 <SPK_B> 잠깐만요 <NEXT_AUDIO>
[AUDIO_74] <SPK_A> <EOT> <SPK_B> 그건 <NEXT_AUDIO>
# A 는 방해당해 floor 를 잃음 → EOT. INTERRUPT 는 이벤트 층에서 유도
[AUDIO_75] 아니에요 <NEXT_AUDIO>
# 직전 블록 B → 생략
```

**사례 6 — 동시 시작**

```
[AUDIO_80] <SPK_A> <ONSET> <SPK_B> <ONSET> <NEXT_AUDIO>
# A → B 순서. A/B 배정은 먼저 식별된 화자.
# 스트림 첫 onset 이 같으면 50 Hz VAD 가 이른 쪽, 동률이면 라벨 permutation 무작위(일관)
```

**사례 7 — 스트림 끝 (flush)**

```
<EMPTY_AUDIO> 입니다 <EOT> <NEXT_AUDIO>
# δ 때문에 넘긴 토큰·이벤트
<EMPTY_AUDIO> <NEXT_AUDIO>
# 빈 라운드
```

### 4.5 학습 규칙
- **손실 위치**: 텍스트·태그·이벤트·`<NEXT_AUDIO>` 전부(§4.1 가중). `[AUDIO_k]`·`<EMPTY_AUDIO>`·prefix 는 입력.
- **무음 청크**: 시퀀스의 75–85 % 가 `<NEXT_AUDIO>` 뿐이다. Phase 1 과 같이 가중 0.3/0.15 로 균형을 맞추고, activity 헤드는 이 청크들에서 `00` 을 학습한다(헤드 손실은 가중 없이 모든 청크).
- **단일 화자 데이터(replay 30 %)**: 같은 문법. `<SPK_A>` 한 번, VAD 로 `<ONSET>`/`<HOLD>`/`<EOT>` 유도(스트림 안 발화 사이 pause ≥0.2 s 이고 이어지면 `<HOLD>`, 스트림 끝은 `<EOT>`). 이 데이터는 "혼자 말할 때 B 를 지어내지 않기" 와 완결/미완결(`<EOT>` vs `<HOLD>`)의 의미 단서를 가르친다.
- **합성 대화**: 텍스트·태그·`<ONSET>` 은 정상 손실. `<EOT>`/`<HOLD>`/`<BC>` 는 시퀀스에 넣되 **손실 가중 0**(타이밍이 가짜). 헤드의 VAP·hazard 도 마스크.
- **태그 오염(Q1 후반)**: 학습 시퀀스의 태그 5 % 를 뒤집고 이후 라벨은 원래대로 두어 오류 회복을 가르친다. `<EOT>`↔`<HOLD>` 도 3 % 교란.
- **이벤트 균형**: `<EOT>`·`<HOLD>`·`<BC>` 는 텍스트 토큰의 1 % 미만이라 가중 2.0 + 이벤트 단위 평가. class 별 recall 을 sentinel 에 넣는다.
- **텍스트 전용 사전학습(§7.2)** 은 같은 문법의 텍스트 시퀀스(오디오 자리 마스킹)로 `<EOT>`/`<HOLD>` 를 먼저 배운다.

### 4.6 디코딩(상태 기계, logit 마스크)
상태: `cur`(직전 블록 화자), `active[A|B]`(모델 자신의 `<ONSET>`/end 토큰으로 갱신), `seen[A|B]`(이 청크에서 낸 블록).
| 상태 | 허용 토큰 |
|---|---|
| 청크 시작 | `<SPK_A>`, `<SPK_B>`, (cur 의) `<ONSET>`/텍스트/end, `<NEXT_AUDIO>` |
| 블록 X 안, X 비활동 | `<ONSET>`, 텍스트(δ 지연 토큰 허용), `<SPK_Y>`(Y>X), `<NEXT_AUDIO>` |
| 블록 X 안, X 활동 | 텍스트, `<EOT>`/`<HOLD>`/`<BC>`, `<SPK_Y>`(Y>X), `<NEXT_AUDIO>` |
| end 방출 직후 | `<SPK_Y>`(Y>X), `<NEXT_AUDIO>` |
| B 블록 뒤 | `<NEXT_AUDIO>` 만(A 블록 재진입 금지) |
- 청크당 상한: 구조 토큰 4 + lexical 8(Q1 에서 밀도 p99 로 재조정). 상한 도달 시 `<NEXT_AUDIO>` 강제, 남은 토큰은 다음 청크로 이월(Phase 1 과 같음).
- 이벤트 시각 = 청크 k 의 오디오를 모두 들은 시각(`(k+1)·80 ms`) + 디코드 시간. TurnBench 제출 시 이 값을 쓴다.
- 데모: 화자 레인 2 개, `<ONSET>` 에서 레인 활성, `<EOT>` 에서 문장 확정 표시, `<BC>` 는 작은 말풍선. 헤드의 P(EOT) 막대는 별도.

### 4.7 왜 이 설계인가
- 청크 안 순서를 시간 순 교차가 아니라 화자 블록으로 고정하면 태그 ≤2/청크, BPE 조각 교차 없음, 상태 기계가 단순하다. 시간은 청크 인덱스가 준다.
- 이벤트를 블록 안 토큰으로 두면 Muse 처럼 LLM 이 **문맥으로** endpoint 를 판단하고, 헤드는 그보다 이른 연속 예측을 맡는다. 둘의 불일치(헤드는 EOT 확률 높음, 토큰은 `<HOLD>`)는 진단 신호로 기록한다.
- INTERRUPT 를 토큰으로 두지 않는 이유: 정의가 미래(상대가 멈추는가)에 걸려 있고 `<ONSET>` + 상대 `<EOT>` 로 재구성되므로 어휘를 늘릴 이유가 없다.

## 5. 출력 계약(API)
- 전사 이벤트: `speaker(A|B), text, chunk_k, audio_seen_until, emitted_at`. turn 토큰 이벤트: `speaker, type(onset|eot|hold|bc), chunk_k, audio_seen_until, emitted_at`(§4.3 의 위원회 예측, TurnBench 제출의 1 차 EOT 소스). 화자별 누적 버퍼로 Unicode·공백 복원. 청크 시각(80 ms) 외의 단어 시각은 추정값으로 표시하거나 내지 않는다.
- 활동·turn 이벤트(80 ms 마다): `activity[2], vap_probs[256], onset_hazard[2][6], p_eot, p_interrupt`. `p_eot`/`p_interrupt` 는 VAP 의 `p_now/p_future`(원 VAP 계산) 와 hazard 에서 유도하고 임계값은 dev 에서 고른다.
- 이벤트 시각 = 그 판단에 쓴 오디오를 모두 들은 시각(TurnBench 규약). 서비스 지연은 wall-clock 으로 별도.

## 6. 데이터

### 6.1 원천(mxc 실측 2026-09-11)
| 자원 | 규모 | 역할 | 확인 필요 |
|---|---|---|---|
| AI Hub 71631 stereo (`/soundai/DB/raw/aihub/71631_audio`) | TS_01.실내_5 757 대화 196.6 h + VS_02.실외 186 대화 51.7 h = 248 h; 라벨 JSON 11,023 개 전체 | KO 자연 대화(학습·dev) | 추가 반입 100–200 h + **untouched KO test 20–30 대화**(결정 2) |
| otoSpeech 16 k (`/soundai/DB/raw/otoSpeech16k`) | 420 대화 104.9 h, 화자별 wav + SRT | EN 자연 대화(학습·dev) | 라이선스 학습 사용 확정(결정 1) |
| TurnBench dev (`/soundai/DB/raw/turnbench`) | 38 대화 7.3 h, gold EOT/INT | **EN 평가 전용** | scorer 재클론 |
| NIKL 일상대화 | ≈3,800 h 발화 단위 mono, 화자 ID·시각 | 단일 화자 replay, 준자연 대화 합성 원료, **텍스트 전용 사전학습**(515 만 발화) | 겹침 발화 quarantine 유지 |
| Switchboard · CallHome(`…/EN/TRAIN/OPEN`) | 발화 단위 wav(A/B 표기), CallHome 19.9 h | 시간축 복원 가능하면 EN 자연 대화 확장; 전사는 텍스트 사전학습에 | Q0 에서 CSV 시각 유무 |
| LibriSpeech · KsponSpeech | 1,033 / 1,189 h | 합성 대화 원료(train split 만), replay | — |
| CANDOR | 없음 | — | 확보 여부(결정 1) |

### 6.2 혼합(mixer) 계약 — Phase 2 전용 `vapasr/data/dialogue.py`
- 입력: 동기화된 두 채널 → `mono = g_A·A + g_B·B`, gain 차 U(−6, +6) dB(스트레스 조건 −12 dB), 합산 후 정규화·클리핑 검사, 위상 반전·채널 누설·샘플 오프셋 QC. 선택적으로 RIR 1 개를 공통 적용(두 close-talk 채널의 합은 "한 마이크" 가 아니므로).
- 라벨은 혼합 **전** 채널에서: 채널별 에너지 VAD@50 Hz([[output-vap-target-pipeline]]), 채널별 강제 정렬(단일 화자 aligner 를 혼합 파형에 쓰지 않음), 화자↔채널 매핑 자동 판정·파일별 기록.
- 레코드: `conversation_id, source_speaker_id, slot(A|B), source_channel, src_offset_s, mix_offset_s, dur, gain_db, rir, split, alignment_quality, task_mask{asr, activity, vap, hazard, event}`. 데이터 스키마 v1 은 건드리지 않고 v2 카드·변환기를 둔다.
- 창: 자연 대화는 원 시간축 보존(무음 삭제·발화 당겨 붙이기 금지), 20–60 s 창, crop 뒤 2.56 s 는 라벨에만 쓰고 입력에 넣지 않는다.
- split: 대화·화자 단위. E2 학습에 노출된 71631 대화(159 h 분)는 Phase 2 train 으로만, dev 는 VS_02, **untouched test 는 새로 반입한 대화**. 무작위 crop split 금지.

### 6.3 합성 대화(overlap 제어) — 자연형 우선
- 원료: 같은 언어의 서로 다른 두 화자 발화(LibriSpeech·Kspon·NIKL, train split). 70 % 는 **자연 유형 모사**: 짧은 맞장구(≤1 s)를 상대 발화 꼬리·중간에 얹기, terminal overlap 0.2–0.5 s, 드문 interruption. 30 % 는 스트레스: overlap 10–50 %, 동시 시작, 한쪽 −3/−6/−12 dB, 비슷한 음색.
- overlap 비율 분모 = "적어도 한 명이 말하는 시간". 목표량 EN 500 h + KO 500 h.
- turn 손실은 마스크(activity·전사만 학습).

### 6.4 단계별 혼합 비율(오디오 초 기준, 초기값)
| 구분 | Q1 | Q2 | Q3 |
|---|---|---|---|
| Phase 1 단일 화자 replay(turn 마스크) | 30 % | 30 % | 30 % |
| 자연 2 화자 mono(71631·oto[·Swbd/CallHome]) | 60 % | 35 % | 70 % |
| 합성 2 화자(§6.3, turn 마스크) | 10 %(무겹침·짧은 맞물림) | 35 % | 0 % |

## 7. 라벨과 학습 목표

### 7.1 밀도 라벨(50 Hz VAD → 80 ms 샘플링)
- **activity** (T,2): 청크 안 any-활동. **VAP 256**: 원 경계 [0,.2],[.2,.6],[.6,1.2],[1.2,2.0] s 를 50 Hz 에서 계산해 80 ms 마다 샘플(12.5 Hz 근사 경계는 쓰지 않음, 라벨 버전 v2 기록). **hazard** (T,2,6): 현재 비활동 화자의 다음 onset 까지 구간 [.16,.32,.64,1.28,2.56] s, 관측 종료는 censored. 활동 중 화자는 위험집합 제외.
- **이벤트**(SHIFT/HOLD/INTERRUPT/BACKCHANNEL): [[output-vap-target-pipeline]] 휴리스틱을 TurnBench dev gold 로 정확도 검증한 뒤 평가·보조 손실에 사용.

### 7.2 의미 라벨 — 대규모 약라벨 + 소규모 사람 검증
- **텍스트 전용 사전학습 타깃**: 전사(NIKL·71631·oto·Swbd·CallHome)의 모든 발화 경계와 발화 내부 단어 경계에 (a) `complete/incomplete`(이 prefix 가 발화로 완결인가 — 실제 발화 종료 여부로 자동 산출 + LLM 판정으로 보정), (b) `next = same/other/none`(다음 화자). 수백만 경계.
- **LLM 약라벨**: 큰 LLM 으로 발화 단위 `complete/incomplete/uncertain` 과 backchannel 여부를 매긴다([[source-soulx-duplug]] 의 절차). 학습용.
- **사람 검증**: EN/KO 각 500–1,000 경계를 이중 검토(≥20 %)해 LLM 라벨 agreement·confusion 을 보고. 평가용 subset(같은 침묵 길이의 완결/미완결 쌍, 질문/진술, 한국어 연결어미/종결어미, 맞장구 뒤 계속 발화)도 여기서 만든다.

### 7.3 손실

```
L = L_text + w_tag·L_tag + w_evt·L_turn-token(<ONSET>/<EOT>/<HOLD>/<BC>) + w_next·L_next + λ_act·L_activity + λ_vap·L_VAP + λ_haz·L_hazard
```
- 손실별로 유효 토큰/프레임 수로 정규화, 언어별 기록. λ 는 헤드만 학습 단계에서 고정 1.0 으로 시작하고 joint 단계에서 uncertainty weighting(Kendall)으로 자동 균형. 가드레일: 단일 화자 WER/CER 회귀 ≤ 5 % 상대.
- 도입 순서: Q1 텍스트+태그+`<ONSET>`+NEXT+activity(`<EOT>`/`<HOLD>`/`<BC>` 는 가중 0.5 로 시작) → Q2 동일 → Q3 turn 토큰 가중 2.0 + VAP → hazard.
- **free-running history 단계**(Q3 후반): 같은 체크포인트로 생성한 전사 이력 위에서 헤드를 추가 학습. gold history 결과는 oracle 상한으로만 보고.

## 8. 학습 레시피
| 항목 | 값 | 근거 |
|---|---|---|
| 초기화 | E2 final(인코더 포함), 새 행·헤드만 초기화 | [[output-stage2-e2-final-eval]] |
| LR | thinker 2e-5, adapter 1e-3, 인코더 1e-5(해동), 헤드 1e-4, warmup 500, cosine | E2 레시피. 인코더 동결·저 LR 은 Q1 대조군 |
| 배치 | rank 당 EN 12 / KO 24, 4 노드 32 GPU(`apex`) | E2 실측 메모리 |
| epoch | Q1 2, Q2 2, Q3 헤드만 1 + joint 2 | 노출량으로 환산해 기록 |
| 창 | Q1–Q3 20–60 s, Q4 60–120 s carry | §11 장문 |
| 비용 | Q1 ≈ 1.5 h, Q2 ≈ 2.5 h, Q3 < 1 h, 트랙 A run 당 ≈ 4 h | E2 1.02 s/step |
| 텍스트 사전학습(§7.2) | thinker + 헤드, 오디오 자리 마스킹, 1 epoch ≈ 1 h | 새 항목 |

## 9. 단계 · 관문 · 일정

| 단계 | 기간 | 내용 | 진입/통과 관문(제안, Q0 후 동결) |
|---|---|---|---|
| **Q0 데이터 계약·기준선** | 09-15 → 09-19 | mixer·serializer·라벨 v2·QC pack(EN/KO 각 50 창, 로더 경유 청취), 고정 평가 ID, E2 δ=2/4 기준선, 인코더 저장 provenance 수정, 텍스트 전용 사전학습, 32 창 overfit, 인과성(prefix 절단·미래 교체) 검사 | 누락·덮어쓰기·화자 누출 0, 미래 정보 누출 0, serializer 왕복 무손실, 평가 재현 |
| **Q1 비중첩 2 화자 전사** | 09-22 → 10-03 | 자연 대화 + replay, 태그·activity 헤드, 슬롯 메모리 옵션 구현(켜기는 비교), 인코더 해동 vs 동결 대조 | ASR 가드레일 ≤5 %, 비중첩 DER ≤10 %, 토큰 귀속 오류 ≤5 %, `<ONSET>` recall ≥0.9 @ FPR ≤0.1, 채널 교환 대칭, tick p99 < 80 ms |
| **Q2 overlap 전사** | 10-06 → 10-17 | 자연 overlap + 자연형 합성, cap 재측정, 캐스케이드(diarization+E2)·clean-channel oracle 대조 | Q1 모델 대비 overlap cp 오류 ≥20 % 상대 감소, 화자 소실 ≤5 %, overlap 프레임 F1 ≥0.7, 가드레일 유지. 미달 시 슬롯 조건 오디오 토큰(§3) |
| **Q3 turn 헤드·의미 검증** | 10-20 → 11-07 | 헤드만(자연 대화) → joint → free-running; VAP → hazard → event; 대조군 4 종(§10); 기본 δ=2, δ=4 비교; H2(encoder-only probe) | `<EOT>` 토큰과 헤드 각각 TurnBench 규약 채점; 같은 FPR(≤0.10)에서 audio-only 대비 EOT/INT recall +3 %p 또는 p50 −80 ms(paired bootstrap 95 % CI, seed 2), 가드레일 유지. 참고 목표 TurnBench dev EOT recall ≥0.80 @ FPR ≤0.055(mono 입력) |
| **Q4 장문·실시간·보고** | 11-10 → 11-14 | 60–120 s carry, 10–60 분 자유실행, bounded KV + 슬롯 메모리, 데모 2 레인 + P(EOT), Phase 2 보고서 | RTF <1, 1 시간 backlog 비발산, ID switch ≤1 회/10 분, viol80 ≤1 %, 방출 p99 ≤1 s |
| **트랙 A ASR 강화(병렬)** | 09-22 → 10-17 | E2 기준 별도 run: `<NEXT_AUDIO>` 가중·δ 분포(δ=2 정확도), 대화체 재가중, 잔향·잡음 증강, 디코더 static KV·CUDA graph, MLX 인코더 | Phase 1 test 표본에서 E2 대비 개선·회귀 없음. **Q3 시작 전 한 번만** Phase 2 초기값으로 승격(Q0 pack 재평가) |

**MVP**: Q1 + Q2 통과 = 목표 G2(활동 부분)·G4·G5·G6 달성. Q3 가 미달이면 "공유 모델로 화자별 전사와 활동 예측" 까지만 주장하고 의미 기반 turn 예측은 보류한다. 일정은 데이터 반입·사람 라벨링 대기를 포함하지 않는다.

## 10. 평가 프로토콜

| 축 | 지표 | 셋 | 규약 |
|---|---|---|---|
| 단일 화자 ASR | WER/CER δ=2/4, S/D/I | Phase 1 test 표본 s300/u1000 | E2 와 같은 TN·δ |
| 대화체 ASR | CER/WER | 71631 VS_02, untouched KO test, oto held-out | — |
| 화자 귀속 | 토큰 귀속 오류율, cpWER/cpCER, tcpWER(collar 5 s) | 위 + 합성 유형별 | meeteval 버전 고정, 대화 단위 permutation 1 개, 누적 후 채점 |
| overlap | 활동 overlap P/R/F1, overlap 구간 cp 오류, 화자 소실률(정의: 창 안 한 화자 lexical recall <10 %) | 합성 유형별(BC/terminal/INT/장시간), 자연 overlap | 정렬 품질별 병기 |
| diarization | DER(miss/FA/conf, collar 0 주·250 ms 보조), JER, ID switch | 71631, TurnBench dev | 80 ms 해상도 명시 |
| turn-taking | EOT/INT recall@FPR 0.055/0.10, latency p10/50/90, P(EOT) at −600/−300/0 ms, BC F1, calibration | TurnBench dev 공식 scorer, KO 검수 held-out | 임계값 dev 선택, mono 입력 명시, teacher forcing 없이 |
| 실시간 | tick p50/p99, RTF, backlog, 청크당 토큰 수 | H200 단독, M4 MLX | 오디오 가용/계산/네트워크 분리 |

**대조군**: E2(무태그), 캐스케이드(공개 diarization + E2), clean-channel oracle(E2 를 분리 채널에 각각), VAP oto(stereo, 입력 조건 다름 명시), 사람 −151 ms. **의미 기여 대조군 4 종**: audio-only causal 헤드(같은 인코더·문맥 예산) / integrated(제안) / text-ablation(lexical 이력 마스킹, 맞춰 학습한 대조군 포함) / oracle-history(관측 시각까지의 gold 전사). 사람 검수 의미 subset 결과는 자연 대화 결과와 별도 보고.

## 11. 구현 작업 지도
| 작업 | 위치 | 검증 |
|---|---|---|
| mixer·스키마 v2·QC | 신규 `vapasr/data/dialogue.py`, `experiments/p2_build_dialogue.py` | 합산·offset·채널·split·task_mask, 청취 |
| 화자별 블록 serializer + turn 토큰 | 신규 `vapasr/data/dialogue_interleave.py` | 두 전사·이벤트로 완전 복원, 태그 생략·`<ONSET>`/end 배치·flush·cap, §4.4 사례 7 종 단위 테스트 |
| 디코드 상태 기계(logit 마스크) | `vapasr/hf/modeling_vapasr.py`, `live.py`, `live_mlx.py` | §4.6 허용 표, 상한, 이벤트 시각 |
| 라벨 v2(activity/VAP/hazard/event @80 ms) | `vapasr/data/targets.py` 버전 분기 | 50 Hz 참조 경계, censoring, permutation 일치 |
| 텍스트 전용 사전학습 | 신규 `experiments/p2_text_pretrain.py` | 경계 수·agreement 보고 |
| 모델: 태그 억제 해제, 헤드, 슬롯 메모리, 인코더 항상 저장 | `vapasr/hf/modeling_vapasr.py`, `configuration_vapasr.py` | save/load logits·헤드·인코더 parity, deferred `<NEXT_AUDIO>` 묶음에서 마지막 오디오 위치 hidden 수집 |
| 학습 | `experiments/s3_train_hf.py` 확장 또는 `p2_train_hf.py`, `slurm/p2_train.sbatch` | 초 기준 mixture, 재개, rollout provenance |
| 평가 | 신규 `experiments/p2_eval.py` | cpWER/DER/turn/latency, dev/test 불변 |
| live | `vapasr/hf/live.py`, `live_mlx.py`(hidden 반환 경로), `experiments/live/` | A/B 레인·80 ms 헤드·장문 |

검증 순서: serializer 왕복 → 로더 청취 → 32 창 overfit → save/load parity → 인과성 검사 → multi-rank smoke → held-out 자유실행 → live parity → 장문.

## 12. 리스크와 중단 조건
| 리스크 | 대응 | 중단/전환 조건 |
|---|---|---|
| 인코더가 겹친 음성을 두 화자로 표현 못 함 | 해동 기본, overlap 비율·LR 상향, 슬롯 조건 오디오 토큰 | Q2 관문 2 회 미달 → G5 를 "overlap 검출 + 우세 화자 전사" 로 축소(사용자 판단) |
| 태그 오류 전파·장문 ID 반전 | 태그 가중, 태그 오염 주입, 채널 교환 증강, 슬롯 메모리 | ID switch >1 회/10 분 → 슬롯 메모리 필수화, t-vector 보조 손실 |
| 의미 기여 미검출 | 텍스트 사전학습 헤드, LLM 약라벨, δ=2 | Q3 대조군 미통과 → 의미 기반 주장 보류(MVP 유지) |
| EN 자연 대화 부족(oto 104 h) | 합성으로 전사·activity 커버, turn 은 oto 학습·TurnBench dev 평가 | CANDOR 미확보 시 EN turn 결과에 데이터 한계 명시 |
| 라벨 노이즈(71631 끝 시각·휴리스틱) | VAD 기반, TurnBench gold 로 휴리스틱 검증 | 검증 정확도 <0.8 이벤트는 손실에서 제외 |
| tick 예산 초과(overlap 청크 토큰 집중) | cap 재측정, p99 감시 | backlog 지속 증가 → cap 유지·flush 이월 또는 δ 상향 |
| 긴 대화 KV | bounded KV(최근 60–120 s 오디오) + 텍스트 KV 보존 + 슬롯 메모리, RoPE 연속성 테스트 | 전환 시 WER·turn 변화 >2 % 상대 → 창 단위 재시작 + 텍스트 요약 carry |
| 멀티태스크 간섭 | 헤드만 → joint, uncertainty weighting | 가드레일 위반 → λ 축소·헤드만 유지 |
| 라이선스(oto non-commercial, AI Hub 재배포) | 학습 사용 범위 확인, KO 벤치마크는 규약·스크립트만 공개 | — |

## 13. 사용자 결정 사항과 기본값
| # | 결정 | 기본값(결정 없을 때) |
|---|---|---|
| 1 | otoSpeech 학습 사용 확정, CANDOR 확보 | oto 학습 사용(비상업 연구), CANDOR 미확보 가정 |
| 2 | 71631 추가 반입량 + untouched KO test 20–30 대화 | test 용 30 대화만 반입, 학습은 248 h |
| 3 | 단일 모델·항상 태그 | 채택 |
| 4 | turn 출력 = 병렬 헤드 기본 + 상태 토큰 ablation | 채택(상태 토큰은 Q3 ablation 1 조건) |
| 5 | 슬롯 메모리(P3)를 Q1 부터 켤지 | Q1 은 비교, Q4 전 결정 |

## 불확실성
- 텍스트 전용 사전학습이 음향 헤드로 전이되는지, 슬롯 메모리가 실제 ID 반전을 줄이는지는 가설이다(비용은 작다).
- 0.6B thinker 하나로 겹친 두 화자를 한 시퀀스에 전사하는 것은 LLM 디코더 규약에서 처음 시도한다.
- Switchboard·CallHome 시간축 복원, NIKL 준자연 합성의 타당성, 자연 overlap 정렬 신뢰도는 Q0 측정 대상.
- 외부 참고(t-SOT, Sortformer, meeteval)는 원문 재확인 없이 인용했다.

## 근거
- [[output-phase2-streaming-asr-diarization-plan]] — 계약·관문·평가 규율·구현 지도의 원안
- [[output-phase2-plan-critique]] — 제안 P1–P9 와 코드 검증
- [[output-phase2-plan]] — 서버 데이터 실측·일정·비용
- [[output-phase1-report]], [[output-stage2-e2-final-eval]] — E2 구조·레시피·결과
- [[output-interleaved-streaming-slm-architecture]], [[decision-mono-input]] — 설계 원칙
- [[output-vap-target-pipeline]], [[turn-taking-objectives]] — 라벨·손실
- [[turn-taking-evaluation-protocol]], [[source-turnbench]] — 평가 규약·기준선
- [[source-conversation-corpora]], [[source-soulx-duplug]] — 코퍼스·LLM 라벨링 선례
