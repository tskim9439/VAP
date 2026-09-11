---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 turn 토큰 제안서 — 정본 계획의 교차 직렬화 위에 Muse 식 <ONSET>/<EOT>/<HOLD>/<BC> 를 화자 태그에 귀속시켜 시퀀스에 넣는 규약, 무음·단일 화자·교대·맞장구·끼어들기·동시 시작·flush 사례, 분리 채널에서 전자동 유도하는 데이터 파이프라인(VAD → 채널별 정렬 → 이벤트 → 슬롯 → 혼합 → 직렬화 → QC)과 데이터 종류별 손실 규칙. 정본에 반영 전 검토용
sources:
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[source-muse-voice-transcribe]]'
  - '[[output-vap-target-pipeline]]'
  - '[[turn-taking-objectives]]'
  - '[[source-turnbench]]'
  - '[[source-conversation-corpora]]'
  - '[[output-phase2-plan-critique]]'
---

# Phase 2 제안서: 시퀀스 안 turn 토큰과 데이터 생성 파이프라인

## 질문
정본 계획([[output-phase2-streaming-asr-diarization-plan]])은 turn-taking 을 병렬 헤드로만 낸다. Muse 처럼 각 화자의 turn 이벤트를 시퀀스 안 토큰으로도 내려면 (1) 규약을 어떻게 정하고 무음·단일 화자 같은 사례에서 시퀀스가 어떻게 생기며, (2) 그 학습 데이터를 어떻게 만드는가. 이 문서는 정본에 반영하기 전의 검토용 제안이다.

## 요약
- 토큰 4 종 `<ONSET>` `<EOT>` `<HOLD>` `<BC>` 를 추가하고, 화자 태그가 뒤따르는 토큰의 귀속이라는 정본 규칙에 따라 turn 토큰도 직전 태그의 화자에 귀속시킨다. 청크 내 직렬화는 정본의 종료 시각 순 교차를 그대로 쓴다.
- `<EOT>`/`<HOLD>`/`<BC>` 의 라벨은 미래(최대 3 s)로 정해지므로 모델이 문맥·운율로 예측해야 한다. 병렬 헤드는 그보다 이른 연속 확률을 내는 별개 출력이고 둘 다 TurnBench 규약으로 채점한다.
- 데이터는 손 라벨 없이 분리 채널 코퍼스에서 전부 자동 유도한다: 채널별 VAD → 채널별 강제 정렬(기존 파이프라인) → `derive_events` 확장 → 슬롯 배정 → 혼합 → 직렬화 → QC. 사람은 EN/KO 각 300 종료의 일치율만 검증한다.
- 데이터 종류별 손실 규칙이 핵심이다. 자연 대화와 NIKL 준자연 대화는 전부 손실, 낭독 replay 와 무관 화자 합성은 종료 토큰 가중 0.
- 정본에 반영한다면 §4.4(규약·사례)와 §5.3(파이프라인)으로 들어가고, §3 어휘·§6.1 손실·§9 serializer 행이 함께 바뀐다.

## 2. Turn 토큰과 시퀀스 사례 (교차 직렬화 기준)

정본 계획 §4.2 의 종료 시각 순 교차 직렬화를 유지한 채, Muse 의 `|speech_onset|`/`|speech_endpoint|` 처럼([[source-muse-voice-transcribe]]) **turn 토큰을 시퀀스 안에** 둔다. 화자 태그가 뒤따르는 토큰의 귀속이므로 turn 토큰도 "직전 태그의 화자" 에 귀속된다. 병렬 헤드(정본 §3.2, §6)는 그보다 이른 연속 확률을 내는 별개 출력이며 둘 다 정본 §8 규약으로 채점한다.

| 토큰 | 뜻 | 시각(청크) | 라벨 출처 | 손실 가중(초기값) |
|---|---|---|---|---|
| `<ONSET>` | 그 화자가 말을 시작했다 | `⌊t_on/80ms⌋ + 2` (160 ms 고정) | 채널 VAD 세그먼트 시작(직전 침묵 ≥ 0.2 s) | 1.0 |
| `<EOT>` | turn 이 끝났다(floor 를 넘김·잃음·침묵으로 종료) | `⌊t_off/80ms⌋ + δ` | `derive_events` SHIFT, terminal overlap, 3 s 무발화, 방해당한 종료 | 2.0 |
| `<HOLD>` | 멈췄지만 turn 을 쥐고 있다 | `⌊t_off/80ms⌋ + δ` | HOLD(같은 화자가 3 s 안에 먼저 재개) | 2.0 |
| `<BC>` | 이 발화는 맞장구였다 | `⌊t_off/80ms⌋ + δ` | BACKCHANNEL(상대 발화 중 시작, ≤ 1 s, 상대 계속) | 2.0 |

- 세그먼트 하나당 `<ONSET>` 1 개와 종료 토큰(`<EOT>`/`<HOLD>`/`<BC>`) 정확히 1 개. INTERRUPT 는 토큰이 아니라 "상대 활동 중 `<ONSET>` + 상대의 `<EOT>`" 로 이벤트 층에서 유도한다.
- `<EOT>`/`<HOLD>`/`<BC>` 의 라벨은 **미래(최대 3 s)** 로 정해진다. 방출 시각(종료 + 160–320 ms)에 모델이 문맥·운율로 판단해야 하며, 그것이 의도다. 입력에 미래 오디오는 들어가지 않는다.
- 같은 시각의 정렬: `<ONSET>` → 텍스트 → 종료 토큰, 화자 간은 A → B. 종료 토큰은 그 화자의 마지막 텍스트 토큰 뒤에 온다(`t_end ≤ t_off`).
- 태그 규칙은 정본 §4.2 그대로: 직전 방출 항목과 화자가 다를 때만 `<SPK_X>`. 내용 없는 태그 반복 금지.

사례(δ=2, 한 줄이 한 청크, `#` 은 설명):

**무음** — 블록 없음. 손실은 `<NEXT_AUDIO>` 에만(EN 0.3 / KO 0.15). activity 헤드는 `00`.

```
[AUDIO_0] <NEXT_AUDIO>
[AUDIO_1] <NEXT_AUDIO>
```

**단일 화자만** — 첫 항목에만 태그. 멈춤은 `<HOLD>`, 재개는 `<ONSET>`(태그 생략), turn 끝은 `<EOT>`. Phase 1 replay 스트림도 같은 문법(§3 의 손실 규칙 참조).

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

**동시 시작** — A → B 순. A/B 는 관측 prefix 안 첫 식별 화자(정본 §4.1), 동률이면 50 Hz VAD 가 이른 쪽, 그래도 동률이면 라벨 permutation 무작위(일관).

```
[AUDIO_80] <SPK_A> <ONSET> <SPK_B> <ONSET> <NEXT_AUDIO>
```

**스트림 끝(flush)** — δ 때문에 넘긴 텍스트·종료 토큰을 `<EMPTY_AUDIO>` 라운드로 낸다.

```
<EMPTY_AUDIO> 입니다 <EOT> <NEXT_AUDIO>
<EMPTY_AUDIO> <NEXT_AUDIO>
```

**디코드 제약(logit 마스크)**: 상태는 `cur`(직전 화자), `active[A|B]`(자신이 낸 `<ONSET>`/종료 토큰으로 갱신). 비활동 화자에게는 종료 토큰 금지, 활동 화자에게는 `<ONSET>` 금지, 빈 태그 금지, 청크당 상한(정본 §4.2 의 밀도 실측 후 확정) 도달 시 `<NEXT_AUDIO>` 강제·이월. 이벤트 시각은 청크 오디오를 모두 들은 시각 + 디코드 시간(정본 §4.3).

## 3. 데이터 생성 파이프라인

§2 의 시퀀스는 손으로 라벨하지 않는다. 분리 채널 코퍼스에서 아래 순서로 **전부 자동 유도**하고, 사람은 검증만 한다.

```
분리 채널 (A.wav, B.wav) + 발화 라벨(텍스트, 대략 시각)
  ① 채널별 VAD@50Hz ──▶ 세그먼트(≥0.2 s 침묵으로 분리, <0.1 s 조각 제거)      → t_on, t_off (화자별)
  ② 채널별 강제 정렬 ──▶ 토큰 종료 시각 t_end (깨끗한 채널, 발화 창 = 라벨 ±0.5 s)  [기존 s2_prep · align-asr-tn-v1]
  ③ 두 채널 VAD ──▶ derive_events ──▶ 세그먼트 종료마다 EOT | HOLD | BC, 시작마다 ONSET
  ④ 슬롯 배정 ──▶ crop 안 첫 세그먼트 화자 = A (전사·이벤트·activity·VAP·hazard 라벨에 같은 permutation)
  ⑤ 혼합 ──▶ mono = g_A·A + g_B·B (+RIR)   ※ 채널이 동기화돼 있으므로 ①②③ 의 시각은 그대로
  ⑥ 직렬화 ──▶ 항목 (시각, 화자, 종류, payload) 를 정본 §4.2 규칙으로 정렬 → 청크 배정 → 태그 삽입 → flush
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
2. 정렬 키 `(chunk = ⌊t/80ms⌋, spk(A<B), 종류 순서 ONSET<TEXT<END, 원 순서)`. 같은 청크·같은 화자 안에서는 원 발화 순서 유지, 화자 간은 종료 시각 순(정본 §4.2).
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


## 4. 정본 계획에 반영할 때 바뀌는 곳
| 정본 절 | 변경 |
|---|---|
| §3 구조 | lm_head 출력에 turn 토큰 4 종 추가, 억제 목록에서 제외 |
| §4.2 | turn 토큰의 정렬 순서(`<ONSET>` → 텍스트 → 종료, A → B)와 태그 귀속 규칙 한 문단 |
| §4.3 API | turn 토큰 이벤트 `speaker, type, chunk_k, audio_seen_until, emitted_at` |
| §5.2 로더 | `task_mask` 에 `turn_token` 항목, NIKL 준자연 대화 리더 |
| §6.1 손실 | `w_evt·L_turn-token` 항, 도입 순서(Q1 은 `<ONSET>` 만 가중 1.0·종료 토큰 0.5 → Q3 2.0) |
| §7 관문 | Q1 에 `<ONSET>` recall ≥0.9 @ FPR ≤0.1, Q3 에 `<EOT>` 토큰의 TurnBench 채점 |
| §9 구현 | serializer 에 이벤트 항목, 디코드 logit 마스크 상태 기계 |

## 불확실성
- 낭독 replay 에서 종료 토큰 손실을 0 으로 두면 단일 화자 모놀로그(강연 등)에서 `<EOT>` 를 잘 내지 못할 수 있다. NIKL 준자연 대화가 그 공백을 메우는지 Q1 sentinel 로 본다.
- `<EOT>`/`<HOLD>` 의 휴리스틱 정의와 TurnBench gold 정의의 차이(기록상 2 배)가 해소되지 않으면 EN 평가가 흔들린다. Q0 의 gold 대조가 선행 조건이다.
- 교차 직렬화에서 overlap 청크의 태그 수·생성량 증가는 정본이 이미 실측 항목으로 둔 것이며, turn 토큰이 그 위에 청크당 최대 2 개를 더한다.

## 근거
- [[output-phase2-streaming-asr-diarization-plan]] §4.1–4.3, §5.2, §6 — 직렬화·태그·라벨 규약
- [[source-muse-voice-transcribe]] — `|speech_onset|`/`|speech_endpoint|` 토큰
- [[output-vap-target-pipeline]] — VAD·이벤트 휴리스틱·코퍼스 통계
- [[turn-taking-objectives]], [[source-turnbench]] — 이벤트 정의·평가 규약
- [[source-conversation-corpora]] — 71631 실물 검증(onset 오차 30 ms), NIKL 구조
- 코드: `vapasr/data/targets.py::derive_events`, `experiments/s1_align.py`(토큰 `end_time`), `vapasr/data/nikl.py`(발화 start/end·speaker_id)
