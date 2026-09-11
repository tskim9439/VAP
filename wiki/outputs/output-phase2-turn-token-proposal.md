---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 turn 토큰 제안서 — 교차 직렬화 위에 <ONSET>/<EOT>/<HOLD>/<BC> 를 화자 태그에 귀속시켜 시퀀스에 넣는 규약과 사례 7 종, 데이터 파이프라인(위치는 VAD, 종류는 의미), 의미 라벨은 사후 문맥 LLM 시드 10 만 → 증류 분류기 → 전량 1 천만 경계 → 사람 κ 검증으로 저비용 구축, 완결성 정확도를 주 지표로 하고 TurnBench 는 외부 검증. 검토용
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
- 데이터는 손 라벨 없이 자동 유도한다. **토큰 위치는 음향(VAD 멈춤), 종류는 의미**: `<EOT>` = 의미적 완결, `<HOLD>` = 미완결, `<BC>` = 내용상 맞장구. 의미 라벨은 텍스트만으로 사후 문맥을 보는 LLM 시드 10 만 → 증류 분류기 → 전량(≈1 천만 경계) → 사람 1,000/언어 κ 검증(§3b). 행동(누가 실제로 받는가)은 VAD 라벨로 헤드가 맡는다.
- 주 지표는 사람 라벨 기준 완결성 정확도·응답 기회 P/R 이고 TurnBench 는 EN 외부 검증 하나다(§3b.5).
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


## 3b. 의미 기반 라벨링 (2026-09-11 개정 — 사용자 검토 반영)

§3 의 ③ 은 종료 토큰의 종류를 **누가 다음에 말했는가(타이밍)** 로 정한다. 이는 행동 통계이지 의미가 아니며, TurnBench 규약에 맞춘 정의다. 개정안은 **위치는 음향, 종류는 의미** 로 나눈다.

### 3b.1 원칙: 토큰 = 의미, 헤드 = 행동
| 출력 | 라벨의 뜻 | 라벨 출처 |
|---|---|---|
| `<EOT>` | 이 시점까지의 발화가 **의미적으로 완결**됐다(상대가 받아도 자연스러운 자리, 전이 적정 지점). 다음에 누가 말하는지와 무관 | 텍스트 → LLM 판정(사후 문맥 포함) |
| `<HOLD>` | 멈췄지만 **미완결**(이어 말할 것) | 텍스트 → LLM 판정 |
| `<BC>` | 내용상 **맞장구**(응·네·그렇죠·uh-huh 류, 새 내용 없음) | 텍스트 + 상대 발화 중 여부 |
| `<ONSET>` | 말을 시작했다 | VAD(음향) |
| activity · VAP · hazard 헤드 | 실제로 누가 언제 말했나/말할 것인가 | VAD(행동) — 변경 없음 |

이렇게 나누면 (1) 토큰은 SoulX-Duplug 의 complete/incomplete 상태와 같은 **의미 상태**가 되어 에이전트의 "지금 응답해도 되는가" 에 직접 쓰이고, (2) 행동 예측(누가 실제로 받는가)은 헤드가 맡으므로 TurnBench 규약과도 여전히 이어지며, (3) TurnBench gold EOT 가 휴리스틱 SHIFT 의 2 배였던 이유(사람 주석자는 같은 화자가 이어 말해도 완결 지점을 EOT 로 찍는다)가 자연스럽게 해소된다. 방해당해 미완결로 멈춘 화자는 `<HOLD>` 가 되고, INTERRUPT 는 "A `<HOLD>` + B 활동" 으로 이벤트 층에서 유도한다.

### 3b.2 후보 위치(음향)와 종류(의미)의 결합
| 후보 위치 | 종류 결정 |
|---|---|
| 세그먼트 종료(뒤 침묵 ≥ 0.2 s) | LLM 완결성: COMPLETE → `<EOT>`, INCOMPLETE → `<HOLD>`, AMBIGUOUS → 손실 가중 0(또는 soft 0.5) |
| 세그먼트 종료인데 상대가 이미 말하는 중 | 같은 규칙. 미완결이면 `<HOLD>`(방해당함) |
| 상대 발화 중 시작한 짧은 세그먼트(≤ 1.5 s) | LLM 이 맞장구로 판정하면 `<BC>`, 아니면 위 규칙(짧은 실질 발화도 `<EOT>`/`<HOLD>`) |
| 세그먼트 내부(멈춤 없음) | 후보로 두지 않는다(토큰 위치는 청크라 멈춤 없는 완결점은 이후 과제) |

### 3b.3 라벨러: 사후 문맥을 보는 LLM → 증류 분류기 → 전량 라벨 → 사람 검증
라벨은 **미래를 봐도 된다**(모델 입력이 아니라 정답이므로). 사후 문맥을 주면 완결성 판단이 훨씬 쉽고 정확하다.
```
프롬프트 (텍스트만, 오디오 불필요)
  이전 문맥: 최근 6 턴(양 화자, 화자 표시)
  현재 화자의 발화(후보 지점까지)
  [사후] 같은 화자의 다음 발화 + 상대의 다음 발화 + 멈춤 길이
  질문: (a) 후보 지점에서 현재 화자의 기여가 완결됐는가 COMPLETE / INCOMPLETE / AMBIGUOUS
        (b) 이 세그먼트는 맞장구인가 BACKCHANNEL / SUBSTANTIVE
        (c) 근거 한 줄
```
| 단계 | 규모 | 비용(추정) | 산출 |
|---|---|---|---|
| L1 LLM 시드 라벨 | 언어별 10 만 후보(멈춤 길이·코퍼스·발화 길이로 층화) | 오픈 LLM(8–30B) vLLM, H200 8 장에서 수 시간 | 시드 라벨 + 근거 |
| L2 증류 분류기 | L1 로 텍스트 분류기 학습(Qwen3-0.6B thinker 텍스트 전용 fine-tune 또는 KO/EN 인코더) | GPU 1 시간 | 완결성·맞장구 분류기 |
| L3 전량 라벨 | 71631 332 만 + NIKL 515 만 + oto·Swbd·CallHome 발화 경계 전부(≈ 1 천만) | 분류기 추론, 수 시간 | 모든 종료 토큰의 종류 |
| L4 사람 검증 | 언어별 1,000 경계 이중 검토(20 % 삼중), AMBIGUOUS 우선 | 사람 2 명 × 2 일 | κ, confusion, 프롬프트·분류기 보정 |
| L5 반복 | L4 불일치 유형을 L1 프롬프트에 few-shot 으로 추가 → L2–L3 재실행 | 반나절 | v2 라벨 |
- 텍스트만 쓰므로 오디오 파이프라인(§3 ①②⑤⑥)과 **독립·병렬**로 지금 시작할 수 있다. 전사가 있는 코퍼스는 전부 라벨 가능하다(NIKL 은 오디오 혼합 없이도 텍스트 사전학습 데이터가 된다).
- 사람 검증 목표: LLM/분류기 vs 사람 κ ≥ 0.7. 미달 유형(예: 한국어 연결어미 뒤 멈춤)은 AMBIGUOUS 로 돌려 손실 0.
- 한국어는 종결어미가 강한 단서라 완결성 판단이 쉽고, 연결어미("~는데", "~고")가 어려운 구간이다. 영어는 구두점 없는 lexical 전사라 사후 문맥이 특히 중요하다.

### 3b.4 학습에서의 사용
- 토큰 손실: `<EOT>`/`<HOLD>`/`<BC>` 는 L3 라벨로. AMBIGUOUS 는 손실 0. 낭독 replay 는 텍스트가 대화가 아니므로 L3 라벨을 만들지 않고 종료 토큰 가중 0(§3 표와 동일).
- 텍스트 전용 사전학습(정본 반영 시 §6.3 대안): 같은 L3 라벨로 thinker 위 완결성 헤드를 오디오 없이 먼저 학습한다. 수백만 경계라 의미 신호가 충분하다.
- 헤드 라벨(activity·VAP·hazard)은 그대로 VAD. 토큰(의미)과 헤드(행동)의 불일치 — 예: `<EOT>` 인데 상대가 안 받음 — 는 오류가 아니라 정상이며, 진단 통계로 기록한다.

### 3b.5 평가: TurnBench 를 하나의 외부 검증으로
| 축 | 지표 | 셋 |
|---|---|---|
| **완결성 판정(주 지표)** | prefix 만 들은 시점의 `<EOT>`/`<HOLD>` 정확도·F1, 사람 라벨 기준, 멈춤 길이별·어미 유형별 | KO: 71631 held-out 1,000 경계(사람 검증), EN: oto held-out 1,000 |
| 응답 기회 | 사람이 "여기서 응답해도 된다" 고 표시한 지점의 precision/recall, 지연 | 위와 같은 셋 |
| 맞장구 | `<BC>` P/R/F1 | 위 |
| 행동 예측 | EOT/INT recall@FPR·latency(헤드 + 토큰) | TurnBench dev(EN 외부 검증), KO 검수 held-out |
| 의미 기여 | audio-only / integrated / text-ablation / oracle-history 대조군 | 정본 §8.2 |

## 3c. 왜 이것이 효율적인가
- 라벨링 입력이 텍스트뿐이라 GPU 시간이 작고 오디오 처리와 병렬이다.
- 시드 10 만 → 증류 → 전량 구조라 LLM 호출이 전체의 1 % 이다.
- 사람은 검증(수천 건)만 하고, 불일치는 프롬프트 few-shot 으로 되먹인다.
- 오디오 파이프라인(§3)은 바뀌지 않는다. ③ 의 타이밍 규칙은 헤드 라벨과 이벤트 평가에 그대로 남고, 토큰 종류만 의미 라벨로 교체된다.

## 3d. `<HOLD>` 의 운영 규칙 — 잠정 판단과 승격 (2026-09-11 개정)

**문제**: `<HOLD>` 는 "미완결, 이어 말할 것" 이라는 약속이다. 추론에서 `<HOLD>` 를 냈는데 후속 발화가 없으면 전사는 확정되지 않고 에이전트는 응답하지 못한 채 기다린다.
**같은 문제는 `<HOLD>` 를 없애도 남는다**: `<EOT>` 만 있는 설계에서 "EOT 를 내지 않은 침묵" 이 정확히 같은 정체 상태다. 차이는 `<HOLD>` 가 그 상태를 **명시적**으로 만들어 타이머와 되돌림을 걸 수 있게 한다는 점이다. 따라서 `<HOLD>` 는 유지하되 **최종 판단이 아니라 기한 있는 잠정 판단**으로 정의한다.

### 3d.1 정의 변경
| 토큰 | 성격 | 되돌림 |
|---|---|---|
| `<EOT>` | **최종**. 전사 확정, floor 개방 | 없음(이후 같은 화자가 말하면 새 `<ONSET>`) |
| `<HOLD>` | **잠정**. "지금까지는 미완결로 보이며, 곧 이어질 것으로 예상" | 같은 화자의 `<ONSET>` 없이도 이후 청크에서 `<EOT>` 로 **승격** 가능 |
| `<BC>` | 최종 | 없음 |

문법에 한 줄 추가: `<HOLD>` 뒤 같은 화자의 텍스트·`<ONSET>` 이 없는 상태에서 `<EOT>` 를 낼 수 있다(승격). 승격 `<EOT>` 는 태그 규칙에 따라 필요하면 `<SPK_X>` 를 앞세운다.

### 3d.2 학습: 승격을 데이터로 가르친다
| 상황(라벨 관점) | 시퀀스 |
|---|---|
| 미완결 멈춤 후 같은 화자가 τ_max 안에 재개 | `<HOLD>` … `<ONSET>` 텍스트 (기존) |
| 미완결 멈춤 후 **아무도** τ_max(초기값 2.0 s) 동안 말하지 않음(포기된 발화) | `<HOLD>` 를 멈춤+δ 에, **`<EOT>` 를 멈춤+τ_max 청크에** — "침묵이 길어지면 끝난 것으로 본다" 를 모델이 직접 배운다 |
| 미완결 멈춤 후 상대가 먼저 말함 | `<HOLD>` … `<SPK_B> <ONSET>` (A 의 floor 는 헤드·이벤트 층에서 B 로 이동, INTERRUPT 유도) |
| 미완결 멈춤이 τ_max 를 넘긴 뒤 같은 화자가 재개(드묾) | `<HOLD>` … `<EOT>`(τ_max) … `<ONSET>` 텍스트 — 재개는 새 turn 으로 취급. 빈도를 Q0 에서 측정(라벨 통계상 gap 중앙값 ≈1 s 이므로 소수) |
- τ_max 는 dev 에서 "정체 지연 vs 느린 화자 끊김" 의 교환으로 고른다(후보 1.5 / 2.0 / 3.0 s). 학습 라벨과 추론 정책에 같은 값을 쓴다.
- `<HOLD>` 손실 가중은 `<EOT>` 보다 낮게(1.0 vs 2.0) 두어 애매하면 `<EOT>` 쪽으로 기울게 한다. 잘못된 `<EOT>` 는 응답이 한 박자 이르게 나오는 비용이고, 잘못된 `<HOLD>` 는 정체 비용이라 후자가 더 비싸다.

### 3d.3 추론: 정책 계층(모델 밖의 안전장치)
```
상태: last_token ∈ {none, HOLD, EOT}, t_silence(마지막 활동 이후 경과), 헤드: P_resume(같은 화자 Δ 안 재개) = hazard 헤드
매 80 ms:
  if 모델이 <EOT> 를 냄                      → EOT(reason=semantic)
  elif last_token == HOLD and t_silence ≥ τ_max → 강제 <EOT> 를 KV 에 주입 → EOT(reason=timeout)
  elif last_token == HOLD and P_resume(1.0 s) < θ → 강제 <EOT> 주입 → EOT(reason=hazard)      # 조기 승격
  elif last_token == none and t_silence ≥ τ_max  → 강제 <EOT> 주입 → EOT(reason=timeout)      # EOT 를 안 낸 침묵
```
- 강제 토큰을 KV 에 넣어 모델 상태와 API 상태가 어긋나지 않게 한다(정본 §4.3 의 "세 개의 시각" 유지).
- API 의 EOT 이벤트에 `reason ∈ {semantic, hazard, timeout}` 을 실어 다운스트림이 구분한다(에이전트는 timeout EOT 에 더 보수적으로 응답 가능).
- hazard 헤드가 "같은 화자의 재개" 확률을 내므로(정본 §6.2 의 화자별 다음 onset 정의) 고정 타임아웃보다 이른 승격이 가능하다.

### 3d.4 평가 지표 추가
| 지표 | 정의 |
|---|---|
| 정체율(stuck rate) | 사람 라벨 완결 지점 중 τ_max 안에 EOT(어떤 reason 이든)가 나오지 않은 비율 |
| 잘못된 HOLD 비용 | `<HOLD>` 를 냈으나 후속 발화가 없던 경우의 EOT 지연 분포(semantic 대비 추가 지연) |
| 끊김률 | `<EOT>`(semantic·hazard·timeout)가 나온 뒤 같은 화자가 1 s 안에 이어 말한 비율 |
| reason 별 비율 | semantic / hazard / timeout EOT 의 비율 — timeout 비율이 높으면 모델이 승격을 못 배운 것 |

### 3d.5 대안과 채택 근거
| 안 | 장점 | 단점 |
|---|---|---|
| **A. `<HOLD>` 유지 + 승격 + 정책 계층(채택)** | 미완결을 명시해 학습 신호·UI 신호가 생기고 타이머를 걸 수 있다. 포기된 발화로 승격을 학습 | 토큰 1 종·규칙 추가 |
| B. `<HOLD>` 삭제, 침묵 = 암묵적 hold | 단순 | 정체 문제는 동일하게 남고 명시 신호가 없어 타이머만 남는다. 미완결 학습 신호 상실 |
| C. `<HOLD>` 에 기한 토큰 세분(`<HOLD_SHORT>`/`<HOLD_LONG>`) | 재개 예상 시각까지 표현 | 라벨 노이즈 증가, hazard 헤드와 중복 |

B 는 Q1 에서 `<HOLD>` 정밀도가 낮을 때의 대체안으로 남긴다.

## 4. 정본 계획에 반영할 때 바뀌는 곳
| 정본 절 | 변경 |
|---|---|
| §3 구조 | lm_head 출력에 turn 토큰 4 종 추가, 억제 목록에서 제외 |
| §4.2 | turn 토큰의 정렬 순서(`<ONSET>` → 텍스트 → 종료, A → B)와 태그 귀속 규칙, `<HOLD>` → `<EOT>` 승격 문법 |
| §4.3 API | EOT 이벤트에 `reason ∈ {semantic, hazard, timeout}`, 정책 계층(§3d.3) |
| §4.3 API | turn 토큰 이벤트 `speaker, type, chunk_k, audio_seen_until, emitted_at` |
| §5.2 로더 | `task_mask` 에 `turn_token` 항목, NIKL 준자연 대화 리더 |
| §6.1 손실 | `w_evt·L_turn-token` 항, 도입 순서(Q1 은 `<ONSET>` 만 가중 1.0·종료 토큰 0.5 → Q3 2.0) |
| §6.3 의미 라벨 | 사람 파일럿 500 건 → "LLM 시드 → 증류 → 전량 → 사람 검증"(§3b.3)으로 교체, 텍스트 전용 사전학습 추가 |
| §8 평가 | 완결성 정확도·응답 기회 P/R 을 주 지표로 추가, TurnBench 는 외부 검증 |
| §7 관문 | Q1 에 `<ONSET>` recall ≥0.9 @ FPR ≤0.1, Q3 에 `<EOT>` 토큰의 TurnBench 채점 |
| §9 구현 | serializer 에 이벤트 항목, 디코드 logit 마스크 상태 기계 |

## 불확실성
- LLM 완결성 라벨이 구두점 습관(문장 단위)에 치우쳐 대화의 실제 전이 지점과 어긋날 수 있다. 사람 κ 와 어미 유형별 confusion 으로 확인한다.
- `<HOLD>` 승격의 τ_max 는 언어·도메인별로 다를 수 있다(한국어 자유대화의 긴 멈춤). dev 에서 고르되 reason 별 비율을 감시한다.
- 멈춤 없는 완결점(세그먼트 내부)은 후보에서 빠진다. 청크 단위 토큰으로는 표현이 어렵고, 헤드(VAP)가 일부를 맡는다.
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
