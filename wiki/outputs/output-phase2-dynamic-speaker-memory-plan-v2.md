---
type: output
status: active
created: 2026-09-15
updated: 2026-09-15
summary: 동적 화자 메모리 대안의 실행안 v2 — lazy-free lane(R=6, N≤R이면 정본 K슬롯과 동일)·EOT 즉시 방출 soft label(§11, SEG_END 없음)·병렬 학습·보유 DB(71631/otoSpeech/AMI/ICSI/NOTSOFAR) 배치·AMI/ICSI/NOTSOFAR 실측 근거·D0–D5 관문
sources:
  - '[[output-phase2-dynamic-speaker-memory-plan]]'
  - '[[output-phase2-speaker-representation-comparison]]'
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[output-phase2-training-db]]'
  - '[[output-phase2-data-inventory]]'
  - '[[task-secure-meeting-corpora]]'
---

# 대안 실행안 v2: 재사용 lane + 동적 화자 메모리

> 2026-09-15: 이 문서의 §3·§5·§6·§9·§10·§11 을 바탕으로 정본 [[output-phase2-lane-plan]] 이 채택됐다([[decision-phase2-canonical-lane-plan]]). 이 페이지는 원안 검토·실측·검증 기록으로 남는다. §4 의 모델 내 화자 메모리(매처·메모리 헤드)는 정본 범위 밖이다.

## 질문

[[output-phase2-dynamic-speaker-memory-plan]](이하 원안)을 현행 코드·정본 계획·보유 DB 위에서 실제로 실행할 수 있게 구체화하고, 논리적으로 성립하지 않거나 근거가 없는 부분을 고쳐, 검증된 실행 계획으로 만든다.

## 요약

- 원안의 핵심인 **lane(전사 채널) → episode(발화 구간) → speaker(세션 화자)의 3층 분리**는 타당하며 유지한다.
- 세 가지를 바꾼다. **(1) lane 해제를 즉시(SEG_END → FREE)에서 지연(lazy-free)으로.** 그러면 세션 화자 수 N이 lane 수 R 이하인 동안 lane = 도착순 슬롯이 되어 정본의 K슬롯 시퀀스와 **완전히 같아지고**, SEG_END 오류가 lane 재배정으로 번지는 원안 최대 위험 (b)가 사라진다. **(2) R=4 → R=6, EOT의 episode pointer head → lane 결합 규칙.** AMI·ICSI·NOTSOFAR-1 어노테이션 실측에서 R=6이면 lane 재배정이 episode의 2.3–2.4 %, 이전 소유자의 EOT가 미결일 수 있는 3초 안 재배정은 0.4 % 이하(상한)라 pointer head 없이 결합 규칙 + unknown 처리로 충분하다. **(3) 학습을 청크 순차 TBPTT에서 정본과 같은 단일 병렬 forward로.** 메모리가 decoder 입력으로 되먹임되지 않으므로 causal한 같은 forward 안에서 과거 episode의 표현으로 메모리 snapshot을 만들 수 있다.
- 보유 DB만으로 실행 가능하다. 2화자(71631·otoSpeech·Switchboard 후보)는 lexical·SEG_END·EOT·재등장, AMI(N=4)·NOTSOFAR-1(N=3–8, 화자별 close-talk 확인)·ICSI(N=3–10, 헤드셋)는 N>2 메모리와 lane 재사용을 공급한다. NOTSOFAR-1은 이번 조사에서 화자별 close-talk wav와 단어 시각이 있음을 확인해 A등급이다. KO는 N>2 A등급 자료가 없어 2화자 대화 결합(stitching) 합성으로 보완하되 합성 표시와 EOT 마스크를 지킨다.
- 이 문서는 대안의 실행안이며 정본 [[output-phase2-streaming-asr-diarization-plan]]을 대체하지 않는다. D1까지는 정본 Q1 모델의 상위 호환(K=6 + SEG_END + 재사용)이므로 별도 코퍼스·별도 초기화 없이 같은 파이프라인에서 비교한다. 채택 관문은 §9다.

## 1. 원안 검토

| 원안 항목 | 검토 결과 | 조치 |
|---|---|---|
| lane/episode/speaker 3층 분리(§0·§2) | 타당. 채널 재사용과 지연 EOT를 함께 다루려면 episode가 필요하다 | 유지 |
| SEG_END 출력 즉시 lane FREE(§5) | **논리적 결함.** 조기 SEG_END → lane 해제 → 같은 화자의 후속 lexical이 다른 lane/새 episode로 가고, 그 사이 다른 화자가 그 lane을 받으면 오귀속이 전파된다. 원안 스스로 최대 위험 (b)로 적었지만 해결책이 없다. 또 N≤R에서도 lane이 화자와 어긋나 정본 K 모델과 같은 조건의 비교가 불가능하다 | **lazy-free**(§3). 해제는 새 화자가 lane을 필요로 할 때만, 가장 오래 닫힌 lane부터 |
| episode pointer `<EVENT_REF>` + pointer head(§4) | 필요성은 "lane 재배정 뒤 이전 소유자의 EOT가 미결"인 경우에만 생긴다. 실측(§3.3)에서 R=6일 때 그런 재배정은 episode의 0.44 %(ICSI)·0.38 %(NOTSOFAR) 이하다. 그 비율을 위해 tokenizer로 재생 불가능한 sidecar·pointer head·확률 분해를 도입하는 것은 비용 대비 근거가 없다 | 결합 규칙(§3.2)으로 귀속, 모호 사례는 `event_unknown`(학습 마스크·평가 계수). pointer head는 모호율 ≥1 %가 실측될 때 재개 |
| R=4 시작값(§1) | 근거 없음. 실측에서 R=4는 ICSI·NOTSOFAR에서 lane 부족 1.8–2.2 %, 재배정 11–18 % | **R=6**(§3.3). R 초과 순간이 있는 회의는 ICSI 27/75이나 부족 episode는 0.24 % |
| 세션 메모리 상한 64(§1) | 보유 자료의 최대 N은 10(ICSI). 64는 실측 불가한 합성 조건 | 학습 padding N_max=16, 런타임 상한 32. 64는 D5 합성 시험만 |
| `<LANE_1..R>` 별도 등록, `<SPK_A/B>` 의미 불변(§4) | lazy-free에서는 N≤R일 때 lane_i = 정본 슬롯 i이므로 의미가 같다. 별도 등록은 registry 이중화이고 정본 모델과 초기화·비교를 어렵게 한다 | 정본 Q0 결정대로 lane 1/2 = `<SPK_A>/<SPK_B>`, lane 3..6 = `<SPK_3..6>`. 신규는 `<SEG_END>`뿐 |
| 청크 순차 teacher forcing + TBPTT(§9) | 필요 조건이 잘못 잡혔다. 순차가 필요한 것은 (i) 모델 출력이 decoder **입력**으로 되먹임되거나 (ii) 모델 예측 상태로 라벨이 바뀔 때다. pointer 되먹임을 없애고 lane 라벨을 정본처럼 teacher-forced로 두면 둘 다 없다 | 단일 병렬 forward + causal 메모리 snapshot(§6). 순차 rollout은 평가·D3b 미세조정 전용 |
| z = g(h_payload, q_episode, audio ctx)(§3.1) | 방향은 맞지만 "겹침에서 source별로 분리되는가"가 미검증이고 원안은 이를 D2까지 미룬다 | D2를 **E2 체크포인트로 지금** 실행(§7). z_dec(decoder 상태)·z_ext(외부 화자 인코더) 두 후보를 같은 데이터로 비교 |
| 메모리 보호 규칙(§6.2), 미래 배제(§8.3), 회귀 fixture(§12) | 타당 | 유지, fixture 4개 추가(§8) |
| 데이터(§10.1 "N=2/4/8, S=1/2/3/4") | 코퍼스·시간·split이 없어 실행 불가 | §5에 보유 DB로 배치 |
| 관문(§11) | 항목은 타당하나 수치 근거 없음 | §9에 실측 기반 수치 |

원안이 "episode는 semantic turn이 아니다", "SEG_END는 turn 종료가 아니다", "EOT는 정책으로 강제 생성하지 않는다"고 못 박은 것은 모두 유지한다.

## 2. 고정할 설계(수정판)

| 항목 | 값 | 근거 |
|---|---|---|
| lane 수 R | **6** | §3.3 실측. AMI(N≤5)는 재배정 0, ICSI/NOTSOFAR 재배정 ≤2.4 % |
| lane 토큰 | lane 1/2 = `<SPK_A>`(151707)/`<SPK_B>`(151708), lane 3–6 = `<SPK_3..6>` 신규 | 정본 §4.1 Q0 registry 결정과 동일. 정본 K를 6으로 두면 vocabulary가 같다 |
| 구조 토큰 | `<ONSET>`·`<EOT>` 정본 공유. **`<SEG_END>` 없음**(2026-09-15 결정, §11) | `<EVENT_REF>` 미등록. 정본 대비 새 구조 토큰 0개 |
| 음향 segment | 화자별 VAD, gap <0.25 s 병합 | 정본 §6.3 |
| EOT 후보 위치 | 해당 segment의 마지막 lexical 목표 청크(δ=4 → offset+0.32 s 부근)와 offset+0.24 s 중 늦은 청크, 그 lexical 직후. 이 자리가 이전 판의 SEG_END 자리다 | §11 |
| lane 해제 | lazy-free(§3.1) | 원안 최대 위험 (b) 제거 |
| EOT 의미·target | 구간 끝 즉시 후보, soft target p_end(§11.2). C-mode 3 s 는 ablation | [[decision-eot-immediate-soft-label]] |
| 세션 메모리 | prototype ≤4/화자, 256-d 정규화, 학습 N_max=16, 런타임 상한 32 | 보유 자료 최대 N=10 |
| 미결 event TTL | 8 s | 원안. C horizon 3 s + 검출 지연 여유 |
| 입력 | E2 mono encoder, 80 ms clock, δ_text=4 주·2 보조 | 정본 |
| 학습 forward | 단일 병렬(정본과 동일), 메모리는 causal snapshot | §6 |

## 3. lane 규약: lazy-free

### 3.1 규칙

1. 새 음향 segment가 시작하면 그 화자가 **현재 소유한 lane이 있으면 그 lane**을 쓴다(episode만 새로 연다).
2. 없으면 **FREE lane** 중 첫 번째를 준다.
3. FREE lane이 없으면 **닫힌 lane(SEG_END 출력 뒤 새 episode가 없는 lane) 중 가장 오래 전에 닫힌 lane**을 새 화자에게 넘긴다. 이때 이전 소유자의 episode는 episode table에 남고, lane은 `generation+1`이 된다.
4. 닫힌 lane도 없으면 `lane_capacity_exhausted`를 보고하고 덮어쓰지 않는다(원안 규칙 유지).

**lane 상태 판정 기준(2026-09-15 보충).** lane 은 `FREE / OPEN / HELD` 세 상태와 `owner, generation, closed_at` 을 가진다.

| 전이 | 라벨 측(reference allocator) | 추론 측(runtime parser) |
|---|---|---|
| FREE → OPEN | 참조 segment 시작 | 그 lane 의 `<SPK_r><ONSET>` 방출 |
| OPEN → HELD (episode 닫힘) | 참조 화자 VAD 가 0.25 s 이상 꺼진 시점 = segment 끝(병합 gap 과 같은 값) | (a) `<SPK_r><EOT>` 방출, 또는 (b) lane r 활동 헤드가 마지막 lexical/ONSET 이후 0.25 s(3 청크) 이상 임계 미만. 둘 중 먼저 오는 것. `closed_at` 기록 |
| HELD → OPEN | 같은 화자의 다음 segment 시작 | 그 lane 의 `<SPK_r><ONSET>` (규칙 1: 같은 소유자) |
| HELD → OPEN, generation+1 | FREE 없음 + 새 화자 시작 → `closed_at` 이 가장 오래된 lane | 같음(규칙 3). 모델이 그 lane 토큰으로 ONSET 을 냄 |
| HELD → FREE | 없음(시간 경과로 해제하지 않음) | 없음 |

HELD 는 "episode 는 닫혔지만 소유는 남은" 상태다. EOT 는 episode 를 닫을 뿐 lane 을 해제하지 않으며, 해제는 규칙 3 의 수요가 있을 때만 일어난다. 규칙 3 의 후보는 HELD lane 뿐이므로 EOT 도 활동 비활성도 관측되지 않은 OPEN lane 은 재배정되지 않는다(→ 필요하면 `lane_capacity_exhausted`). EOT 뒤 같은 lane 에 ONSET 없이 늦은 lexical 이 오면 정본대로 보존·프로토콜 지연 오류로 계수하고 `closed_at` 을 그 청크로 갱신한다. 0.25 s 는 라벨의 segment 병합 gap 과 같은 값으로 두어 라벨 측과 추론 측의 닫힘 기준을 일치시킨다.

**따름정리.** N≤R이면 규칙 3이 한 번도 발동하지 않으므로 lane i = 도착순 i번째 화자이고, 생성되는 토큰열은 정본 §4.2·§4.4의 K=R 슬롯 시퀀스에 `<SEG_END>`만 더한 것이다. 따라서 D1 모델은 정본 Q1 모델과 같은 데이터·같은 registry·같은 초기 backbone으로 학습되고, `<SEG_END>`를 무시하면 정본 평가에 그대로 들어간다. 이것이 두 안을 공정하게 비교할 수 있는 조건이다.

**SEG_END 오류의 영향.** 조기 SEG_END: lane은 유지되므로 후속 lexical은 같은 lane·같은 소유자의 새 episode가 된다. 오귀속은 없고 episode 분절 수만 늘며 이를 계수한다. 누락·지연 SEG_END: 그 lane이 재배정 후보에서 빠지므로 N>R 상황에서 `lane_capacity_exhausted`가 조금 늘 뿐이다. 두 경우 모두 다른 화자의 전사에 영향을 주지 않는다. 원안 §13 (b)의 "lane/memory 전체 오염" 경로는 규칙 1·3 때문에 존재하지 않는다.

### 3.2 EOT 귀속 규칙

`<SPK_r><EOT>`는 **lane r에서 lexical이 닫힌(SEG_END 출력됨) 뒤 아직 EOT를 받지 않은 가장 최근 episode**에 귀속한다.

- lane이 재배정되지 않았으면 그 lane의 직전 episode이며, 이는 정본의 `<SPK_s><EOT>`와 같은 의미다.
- lane이 재배정됐고 새 소유자가 아직 SEG_END를 내지 않았으면 여전히 이전 소유자의 episode다. C-mode EOT는 종료 관측 뒤에만 나오므로 열려 있는 episode를 가리킬 수 없다.
- 재배정 뒤 새 소유자도 SEG_END를 냈고 이전 소유자의 EOT가 TTL 안에 미결이면 **모호**하다. 학습에서는 두 event를 모두 unknown으로 마스크하고, 평가에서는 `event_ambiguous_binding`으로 계수한다.

모호 사례의 상한은 "이전 소유자가 닫힌 뒤 3 s 안에 재배정"으로 셀 수 있다(새 소유자가 그 안에 닫히기까지 해야 실제 모호). §3.3의 값이다.

### 3.3 실측: lane 수와 재배정

AMI(manual 1.6.2 segments), ICSI(NXT core segments), NOTSOFAR-1(gt_transcription, train/dev1/eval_full 237회의)의 화자별 구간을 gap 0.25 s로 병합해 episode를 만들고, 라벨 시각의 SEG_END(offset+0.32 s)로 lane 시뮬레이션을 했다. 스크립트·출력: `raw/sources/experiments/2026-09-15-phase2-lane-sim/`.

| 코퍼스 | 회의 | 세션 화자 수 N | 3명 이상 동시(음성 프레임) | episode/분 | R=4 lane 부족 | R=4 재배정 | R=4 3 s 내 재배정(상한) | R=6 재배정 | R=6 3 s 내 | R=8 재배정 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMI | 171 | 3: 5 · 4: 163 · 5: 3 | 3.9 % | 15.7 | 0.01 % | 0.17 % | 12건 (0.01 %) | 0 (N≤5) | 0 | 0 |
| ICSI | 75 | 3–4: 4 · 5: 14 · 6: 21 · 7: 18 · 8: 10 · 9: 7 · 10: 1 | 4.2 % | 24.2 | 1.83 % | 10.9 % | 3.2 % | 2.41 % | 0.44 % | 0.19 % |
| NOTSOFAR-1 | 237 | 3: 16 · 4: 75 · 5: 70 · 6: 47 · 7: 21 · 8: 8 | 9.0 % | 32.3 | 2.23 % | 17.8 % | 7.9 % | 2.26 % | 0.38 % | — |

읽는 법: lane 재배정 비율은 "N>R 상황에서 lane 하나가 다른 사람에게 넘어간 episode 비율"이고, "3 s 내" 열이 §3.2 모호 사례의 상한이다. R=6이면 두 회의 코퍼스 모두 재배정 약 2.3 %, 모호 상한 0.5 % 미만이다. R=4는 모호 상한이 3–8 %라 pointer head가 필요해지고, R=8은 이득이 0.2 %p뿐이라 lane 토큰·활동 헤드 희소성만 늘린다. **R=6을 채택하고, 정본 Q0의 K 결정에도 같은 값을 제안한다.**

주의: ICSI segments에는 비음성(nonvocalsound) segment가 섞여 있어 episode 수·동시 발화가 다소 과대하다. NOTSOFAR-1은 발화 단위 GT를 병합한 값이다. 둘 다 정렬 파이프라인 구축 후 재계산한다.

### 3.4 시퀀스 예시

정본 §4.4 표기와 같고, lane 재배정만 추가된다. `<SPK_r>`은 lane r의 실제 토큰이다.

```text
[AUDIO_k]   <SPK_A><ONSET> 안녕하세요 <NEXT_AUDIO>
[AUDIO_k+1] <SPK_A> 저는 <SPK_B><ONSET> 네 <NEXT_AUDIO>
[AUDIO_j]   <SPK_A> 김입니다 <SPK_A><SEG_END> <NEXT_AUDIO>      # A 닫힘, lane 유지
[AUDIO_j+3] <SPK_A><ONSET> 그리고 <NEXT_AUDIO>                   # 같은 화자 재개 → 같은 lane, 새 episode
[AUDIO_m]   <SPK_A><EOT> <SPK_B> 그렇군요 <NEXT_AUDIO>            # C 판단 → lane A의 닫힌 최근 episode
# N>R: lane 1–6이 모두 소유 중이고 새 화자 등장. 가장 오래 닫힌 lane(예: 3)을 넘긴다.
[AUDIO_p]   <SPK_3><ONSET> 잠깐만요 <NEXT_AUDIO>                  # lane 3 generation 2, 새 episode·새 화자 후보
[AUDIO_p+9] <SPK_3><EOT> <NEXT_AUDIO>                            # lane 3에서 닫힌 최근 미결 episode = 이전 소유자(§3.2)
```

## 4. 화자 메모리: 표현·매칭·헤드

### 4.1 화자 표현 z

두 후보를 D2에서 같은 데이터로 비교한다.

- **z_dec:** episode의 payload 위치(lexical·ONSET·SEG_END)에서 decoder hidden state를 episode query로 pooling(작은 cross-attention + projection, 256-d 정규화). 원안 §3.1의 정의다. 겹침에서 source별 분리가 되는지가 핵심 가설이다.
- **z_ext:** 외부 화자 인코더(공개 가중치·라이선스 확인 후 선택)를 각 lexical 청크 직전 1.0 s causal 창의 mono에 적용해 episode 단위로 평균. 겹침에서는 섞이지만 비겹침 구간의 안정된 기준선이다.
- **z_cat:** 두 벡터 연결 + projection.

어느 경우도 미래 오디오·깨끗한 채널을 입력하지 않는다. 깨끗한 채널은 라벨(같은/다른 사람)에만 쓴다.

### 4.2 매칭

episode의 z와 메모리 entry의 prototype 사이 cosine을 계산해 KNOWN/NEW/UNRESOLVED를 판정한다(원안 §6.1). 임계값·margin·필요 증거 길이는 dev EER 곡선에서 정한다. 증거 길이 0.25/0.5/1/2 s 구간별 EER을 보고하고, ASR은 provisional episode로 즉시 출력하며 ID 확정만 늦춘다(원안 유지). 결정 위치는 episode의 SEG_END 위치(학습 기본)와 ONSET+τ_ev(지연 측정용)다.

**decoder는 메모리를 읽지 않는다.** 동시에 진행 중인 ≤R 명의 구별은 정본 K 모델과 같이 decoder 문맥(KV)이 담당하고, lane 밖(N>R, 재배정 후 재등장, 오랜 침묵 후 재등장)의 정체성은 매처가 담당한다. 이 분리 덕에 학습이 병렬로 가능하다(§6). 메모리 요약을 decoder prefix로 넣는 변형은 D3b ablation이며 순차 학습이 필요하다.

### 4.3 메모리 헤드

activity `[N]`, 미래 4-bin `[N,4]`, hazard `[N,B]`는 memory entry를 query, `h_audio[k]`를 key/value로 하는 공유 cross-attention 헤드로 낸다. 정본의 bin 경계·censor·risk-set 정의를 그대로 쓴다. D1(메모리 없음)에서는 정본과 같은 lane 행 헤드(K=6)를 쓰며, D4에서 두 헤드를 같은 라벨로 비교한다. 손실은 유효 entry 수로 정규화하고, 미등장·unresolved entry는 마스크한다(원안 §7 유지).

### 4.4 보호 규칙

원안 §6.2를 유지한다. 추가로: 같은 lane의 연속 episode(같은 소유자, 재배정 없음)는 매처 관점에서 하나의 "owner run"으로 묶어 prototype을 갱신한다. episode/분이 16–32(§3.3)라 episode마다 독립 판정하면 짧은 증거 판정이 대부분이 되기 때문이다.

## 5. 데이터: 보유 DB 배치

A등급(화자별 채널 또는 채널 기원 조각)만 쓴다. 정렬은 화자별 채널에 Qwen forced aligner를 적용해 다시 뽑고, 모델 입력은 mono 혼합이다([[output-phase2-training-db]]).

| 자료 | 언어 | N | 규모 | 서버 | 이 계획에서의 역할 | split |
|---|---|---|---|---|---|---|
| 71631 실내 stereo 원본 | KO | 2 | 757대화 196.6 h | `/soundai/DB/raw/aihub/71631` 계열 | lexical·SEG_END·EOT·2인 재등장(E2 노출 감사 뒤) | 정본 split, VS_02는 dev |
| 134-1 실외 조각 | KO | 2 | 1,492대화 ≤344 h | NIA24 134-1 | 위와 같음, 새 KO test 후보 분리 | Q0 |
| otoSpeech | EN | 2 | 420대화, train ≈90 h | `/soundai/DB/raw/otoSpeech16k` | 위와 같음, TurnBench와 화자 비겹침 재확인 | actor 기준 |
| Switchboard(조건부) | EN | 2 | ≈230 h | asr_db | 파일명 시각 검증 통과 시 | 원 split |
| AMI headset | EN | 3–5 (4가 163/171) | 171회의 98.1 h | `/soundai/DB/raw/ami/<회의>/*.Headset-*.wav` + manual 1.6.2 | N=4 메모리·3중 겹침·S≤4 활동 헤드. **재배정 없음**(N≤R) | AMI 공식 scenario split |
| NOTSOFAR-1 | EN | 3–8 | 237회의 24.2 h(train 72·dev1 36·eval 129) | `/soundai/DB/raw/notsofar/240825.1_*/…/MTG_*/close_talk/CT_*.wav`, `gt_transcription.json`(단어 시각) | N 5–8 메모리, lane 재배정 2.3 %, 3중 이상 겹침 9 %. **화자별 close-talk 확인(A등급)** | 공식 train/dev1/eval |
| ICSI headset | EN | 3–10 (≥6이 57/75) | 75회의 71.6 h | `/soundai/DB/raw/icsi/<회의>/chan*.sph` + NXT core | N≥6 메모리·재배정의 주 공급원. R 초과(부족 episode 0.24 %)는 unknown/coverage로 평가 | 회의 계열 기준 hold-out |
| DiPCo | EN | 4 | dev/eval ≈5 h | `/soundai/DB/raw/dipco/DipCo.tgz` | 평가 전용(원격·잡음) | eval |
| CHiME-6 | EN | 4 | train ≈40 h | `/soundai/DB/raw/chime6/*.tar.gz` | 착용 마이크 누설 검사 통과 시 train 보조 | 공식 |

시간은 [[output-phase2-data-inventory]]·[[task-secure-meeting-corpora]]와 §3.3 어노테이션 합산이며, 정렬·QC 후 유효 감독 시간을 다시 센다. 두 채널 시간을 대화 시간으로 더하지 않는다.

**KO의 한계.** KO에는 N>2 A등급 자료가 없다(NIA23 002_Meeting은 혼합 mono B등급). 따라서 KO 메모리 학습은 N=2 자연 대화와 아래 합성으로만 한다. KO N>2 성능은 합성 조건 결과로만 보고하고 자연 회의 성능이라고 하지 않는다.

**대화 결합(session stitching) 합성.** 같은 언어의 서로 다른 2화자 대화 2–4개를 골라 시간축에 배치한다. 각 대화 내부의 발화·간격은 원본 그대로 두고, 대화 사이는 turn 블록 단위로 교대·부분 겹침(offset 0–1.5 s)시켜 N=4/6/8, S≤4를 만든다. 감독: lexical·lane·SEG_END·매처·activity는 유지, **EOT는 전부 마스크**(정본 §5.1: 합성에 자연 floor EOT를 붙이지 않음). 목적은 (i) N>R 재배정 사례를 통제된 비율(전체 episode의 약 5 %)로 공급, (ii) KO N>2 메모리, (iii) 같은 목소리의 60 s 이상 간격 재등장. 합성은 manifest에 `synthetic=stitch`로 표시하고 자연 자료와 섞어 평균을 내지 않는다.

**평가.** TurnBench dev(EN 2인 event), AMI test(N=4), ICSI hold-out(N 5–10, R 초과 포함), NOTSOFAR-1 dev1/eval(N 3–8, 원격 잡음), DiPCo, KO 71631 VS_02(E2 dev 재사용 명시), KO stitched(합성 표시). 지표는 §9.

## 6. 학습 경로

### 6.1 단일 병렬 forward

정본과 같이 전체 teacher-forced 시퀀스를 한 번에 forward한다. lane·episode 라벨은 reference allocator(§3.1을 정답 segment에 적용)에서 오므로 입력열은 고정이다. 메모리는 다음처럼 같은 forward 안에서 만든다.

1. forward로 hidden `h`를 얻는다.
2. 각 episode e에 대해 z_e = pool(h[positions(e)])를 계산한다(§4.1). causal decoder이므로 z_e는 episode e의 마지막 payload 위치까지의 정보만 담는다.
3. 청크 k의 메모리 snapshot M_k = {정답 소유자별 prototype}을 **k 이전에 SEG_END가 나온 episode의 z**로만 구성한다(stop-grad 또는 EMA). 미래 episode는 들어가지 않는다(fixture 12).
4. 매처 손실: episode e의 결정 위치에서 M_{k(e)} ∪ {NEW}에 대한 CE. 정답은 같은 세션 안의 사람 ID로만 정한다. unresolved·짧은 증거는 유효 마스크.
5. 헤드 손실: audio 위치 k에서 M_k를 query로 §4.3 헤드를 계산한다.

메모리 내용이 정답 소유자로 묶이므로 추론(예측 귀속으로 메모리를 만듦)과 차이가 생긴다. 이 exposure bias는 **noisy-memory augmentation**(prototype 교환·삭제·병합을 확률 0.1–0.2로 적용)으로 완화하고, 자유실행 평가로 실제 크기를 잰다. 원안의 순차 TBPTT는 D3b에서 마지막 10–20 % step의 미세조정과 평가 rollout에만 쓴다. Liger·gradient checkpointing·DDP 경로는 정본과 같다.

### 6.2 손실과 순서

`L = L_AR + λ_metric L_metric + λ_match L_match + λ_act L_act + λ_fut L_fut + λ_haz L_haz`. AR 가중 시작값은 정본대로 lexical/SPK/ONSET/SEG_END=1, EOT=2, NEXT EN/KO=0.3/0.15. λ_match=1, λ_metric=0.5(같은 세션 내 대조, 다른 대화의 같은 index를 positive로 쓰지 않음). 켜는 순서: D1 AR만 → D3 매처·metric → D4 헤드. 새 구조 토큰 정확도는 lexical과 분리해 보고한다.

### 6.3 코드 연결점

| 위치 | 책임 | 정본 §9와의 관계 |
|---|---|---|
| `vapasr/data/dialogue_interleave.py`(정본 예정) | 시각순 직렬화. `<SEG_END>` kind 추가(우선순위 ONSET < lexical < SEG_END < EOT) | 공용, 옵션 |
| 신규 `vapasr/data/lane_alloc.py` | reference allocator: `never_free`(정본 K) / `lazy_free`(본 안) 두 정책, generation·episode table | D0 |
| 신규 `vapasr/hf/lane_state.py` | 추론 parser: lane 상태·episode·EOT 결합 규칙·TTL·capacity 보고 | D0 |
| 신규 `vapasr/hf/speaker_memory.py` | z pooling·prototype·매처·revision | D2/D3 |
| 신규 `vapasr/hf/memory_heads.py` | cross-attention activity/future/hazard | D4 |
| `vapasr/hf/modeling_vapasr.py` | `forward`에 episode 위치·메모리 snapshot 입력과 부가 손실; `stream_decode`는 lane parser 사용 | config `speaker_representation=lane_memory`, R, dims, caps, registry version |
| `tests/test_lane_protocol.py` | §8 fixture | |

## 7. D2 화자 표현 probe: 지금 실행 가능

D1 학습 전에 E2 체크포인트로 위험 (a)를 잰다. 71631 stereo 원본(N=2)과 AMI headset(N=4)에서 mono 혼합을 만들고, 정답 화자 segment를 (i) 비겹침 (ii) 겹침으로 나눈다.

- E2 audio 위치 hidden `h_audio`를 segment별로 pooling한 벡터, 외부 화자 인코더 z_ext를 같은 창에서 계산.
- 같은 대화 안 같은/다른 사람 EER을 (i)/(ii)별, 증거 길이 0.25/0.5/1/2 s별로 보고.
- 60 s 이상 떨어진 재등장 쌍의 EER을 따로 보고.

판정: (i)에서 z_ext EER ≤ 10 %면 매처 기준선은 확보된 것이다. (ii)에서 z_ext가 무너지고 h_audio도 무너지면 z_dec(D1 모델의 payload 조건 표현)가 유일한 후보이므로 D3 전 D1 모델로 같은 probe를 반복한다. 겹침 EER 목표는 D2 결과를 본 뒤 동결한다. 이 probe는 학습 없이 추론만 필요하므로 SLURM 1 job 규모다.

## 8. 검증: 이 실행안에 대한 자체 점검

| 주장 | 근거 | 상태 |
|---|---|---|
| N≤R이면 lazy-free 시퀀스 = 정본 K슬롯 시퀀스 + SEG_END | 규칙 3은 FREE lane이 없을 때만 발동, N≤R이면 FREE lane 항상 존재 | 논리적 귀결. D0 테스트로 기계 검증 |
| SEG_END 오류가 오귀속으로 전파되지 않는다 | 규칙 1(소유 lane 우선)·규칙 3(요청 시에만 해제) | 논리적 귀결. fixture 7·17 |
| R=6이면 재배정 ≈2.3 %, 모호 상한 <0.5 % | §3.3 실측(AMI/ICSI/NOTSOFAR 어노테이션) | 실측. ICSI 비음성 segment로 약간 과대 |
| pointer head 불필요 | 위 모호율과 unknown 처리 비용 비교 | 추론. 모호율 ≥1 %면 철회 |
| 병렬 학습 가능 | decoder 입력에 모델 출력 되먹임 없음, lane 라벨 teacher-forced | 논리적 귀결. 대신 exposure bias는 측정 대상 |
| 보유 DB로 N=2–10 자연 자료 확보 | §5, NOTSOFAR-1 close-talk 확인(2026-09-15) | 실측(파일 존재). 정렬 품질은 Q0 |
| KO N>2는 합성만 | A등급 KO 회의 없음 | 사실. 한계로 보고 |
| 겹침에서 z가 분리된다 | 없음 | **미검증.** D2가 관문 |
| N_max=16 충분 | 자료 최대 N=10 | 사실. N>16은 합성 시험만 |

**반론 검토.** (1) "lazy-free는 lane을 오래 붙잡아 R이 더 필요하다" → 실측에서 R=6 lazy 부족(exhausted)은 eager와 거의 같다(ICSI 231 vs 248건). 부족은 순간 겹침이 아니라 N 자체가 크기 때문이다. (2) "정본 K=6로 충분하면 메모리가 왜 필요한가" → N≤6 자료에서는 필요 없다. 필요한 곳은 ICSI N≥7(36/75회의)·NOTSOFAR N≥7(29/237)·장문 재등장이며, 이득은 그 구간에서만 측정한다(§9). (3) "decoder가 메모리를 못 보면 재배정 lane의 재등장 화자를 같은 lane으로 못 보낸다" → 재등장 화자가 어느 lane을 받는지는 정체성이 아니라 용량 문제이고, 정체성은 매처가 episode→speaker로 연결한다. cpWER/DER는 speaker_id로 regroup해 채점한다.

**추가 fixture(원안 §12에 더함).** 17) 조기 SEG_END 뒤 같은 화자 lexical이 같은 lane·새 episode로 감. 18) N≤R 세션에서 `never_free`와 `lazy_free` 출력 동일. 19) 재배정 후 이전 소유자 EOT가 §3.2로 귀속, 새 소유자도 닫힌 경우 unknown. 20) 메모리 snapshot에 미래 episode 없음(위치 기준 검사).

## 9. 실행 계획과 관문

| 단계 | 작업 | 산출물 | 통과 기준 / 중단 |
|---|---|---|---|
| **D0** 프로토콜 | `lane_alloc.py`(두 정책)·`lane_state.py`·serializer 확장·fixture 20개 | 테스트 통과, AMI ES2002a 480블록 round-trip | never_free ≡ lazy_free (N≤R), 소유자 오류 0 |
| **D2** 표현 probe(D0와 병행) | §7 E2 probe | EER 표 | 비겹침 z_ext EER ≤10 %. 겹침 결과로 z 후보 결정 |
| **Q0 공유** | 정렬·QC·manifest(71631·otoSpeech·AMI·NOTSOFAR·ICSI), stitching 생성기 | Dataset | 정본 Q0 관문과 동일 + NOTSOFAR/ICSI 정렬 coverage 보고 |
| **D1** lane 전사 | 정본 Q1 모델을 K=6·`<SEG_END>`·lazy_free로 학습(메모리 없음). 32창 overfit → 실학습 | 체크포인트, 자유실행 결과 | 정본 K=6 대비 WER/CER·cpWER 상대 회귀 ≤5 %; SEG_END 조기율·lane 부족률 보고; ICSI N≥7에서 재배정 처리 오류 계수 |
| **D3** 메모리 | `speaker_memory.py`, §6.1 병렬 학습, noisy-memory aug; D3b 순차 미세조정 ablation | ID 부착 결과 | ICSI/NOTSOFAR N≥7·stitched N=8에서 speaker-attributed 오류가 D1(lane을 ID로 채점) 대비 개선; N≤6에서 D1 대비 회귀 ≤2 %; false-new/false-known/unresolved/최초 확정 지연 보고 |
| **D4** 헤드 | `memory_heads.py` vs lane 행 헤드 | activity/future/hazard | 같은 라벨에서 FPR·지연 동등 이상, N=4/8/16 비용 실측 |
| **D5** 장문 | 60–120 s carry → 10–60 분 자유실행, 세션 재시작 | 장문 로그 | RTF<1, backlog 비발산, ID switch·메모리 오염률 보고 |

D0·D2는 GPU 학습 없이 지금 시작할 수 있다. D1은 정본 Q1과 동일 잡이므로 별도 예산이 아니라 K=6·SEG_END 옵션이다. D3부터가 대안 고유 비용이며, D2에서 겹침 표현이 무너지면 D3 전에 z_dec 재probe를 먼저 한다. 모든 결과는 최초 출력 기준과 수정 후 기준을 분리하고, oracle 귀속 결과는 상한으로만 낸다(원안 §11 유지).

## 불확실성

- 겹침 구간의 화자 표현 분리는 미검증이다(§7이 관문).
- ICSI episode 통계는 비음성 segment를 포함해 다소 과대이며, NOTSOFAR-1은 발화 단위 GT를 병합한 값이다. 정렬 후 재계산한다.
- lazy-free의 병렬 학습은 메모리 내용의 exposure bias를 남긴다. noisy-memory 증강이 충분한지는 D3 자유실행에서만 알 수 있다.
- CHiME-6 착용 마이크 누설과 71631 E2 노출 감사는 정본 Q0 항목으로 아직 미완이다.
- KO N>2는 합성 조건뿐이다.

## 근거

- [[output-phase2-dynamic-speaker-memory-plan]] — 원안 전체(§1·§4·§5·§6·§9·§11·§12)
- [[output-phase2-speaker-representation-comparison]] — N/S/R 구분, t-SOT·t-vector·Streaming Sortformer 대조
- [[output-phase2-streaming-asr-diarization-plan]] — §4.1 Q0 registry, §4.2·§4.4 직렬화, §6.3 C-mode, §9 구현 지도
- [[output-phase2-training-db]]·[[output-phase2-data-inventory]]·[[task-secure-meeting-corpora]] — 보유 DB·서버 경로
- `raw/sources/experiments/2026-09-15-phase2-lane-sim/lane_sim.py`, `lane_sim.out` — AMI/ICSI/NOTSOFAR 어노테이션 lane 시뮬레이션(2026-09-15, mxc)
- NOTSOFAR-1 서버 사본 `…/MTG_30860/close_talk/CT_21..25.wav`, `gt_meeting_metadata.json`(`ParticipantAliasToCtDevice`), `gt_transcription.json`(`word_timing`) — 화자별 close-talk 확인(2026-09-15)

## 10. 변형: 세션 정체성을 모델 밖(외부 화자 인식기)에서 해결할 때

사용자 질문(2026-09-15): "3번째 등장한 화자를 1번째 화자와 같은 lane으로 붙일 필요가 없고, 구간만 뽑히면 추론 때 외부 화자 인식기로 재매핑한다"면 계획이 어떻게 바뀌는가.

**결론: 계획은 §4.2 매처·§4.3 메모리 헤드·§6.1 메모리 snapshot 학습·D3/D4를 빼고 D0·D1·D5만 남는다.** 남는 것은 정본 K=6에 `<SEG_END>`·lazy-free 재사용을 더한 것이므로 이 변형은 별도 대안이 아니라 정본의 옵션이다. 조건부로 남는 것은 per-episode 임베딩 출력 헤드 하나다(아래).

**lazy-free는 이 변형에서 더 중요해진다.** 외부 인식기는 mono 혼합 위에서 동작하므로 겹침 속 짧은 episode에는 증거가 없다. 같은 어노테이션으로 잰 값(`clean_evidence.out`):

| 코퍼스 | episode 자체 비겹침 음성 <0.5 s | 그 episode의 음성 시간 비중 | lazy-free R=6 lane run 누적 비겹침 <0.5 s | 음성 시간 비중 |
|---|---:|---:|---:|---:|
| AMI | 53.3 % | 16.2 % | 1.0 % | 0.3 % |
| ICSI | 48.9 % | 20.7 % | 6.1 % | 3.7 % |
| NOTSOFAR-1 | 66.4 % | 30.8 % | 10.8 % | 4.7 % |

episode마다 독립으로 판별하면 절반 이상이 증거 부족이지만, lane이 유지되어 이전 episode의 증거를 이어받으면 1–11 %로 준다. 따라서 "같은 사람 = 같은 lane"을 학습 라벨에서 없애고 lane을 순수 채널로 무작위 배정하면 외부 재매핑이 실측상 성립하지 않는다. 라벨은 lazy-free 그대로 두고, 모델에게 세션 정체성을 **평가·주장하지 않을 뿐**이다.

남는 비용 세 가지. (1) N>R 재배정(episode의 2.3 %)에서 lane 세대 변경은 모델 출력만으로 구별되지 않으므로 외부 모듈이 "이 lane run과 같은 목소리인가"를 판정해야 한다. (2) 위 표의 잔여 1–11 % episode는 어떤 외부 모듈도 mono 증거로 못 푼다. 이를 줄이려면 모델의 payload 조건 표현(z_dec)을 per-episode 임베딩으로 내보내는 헤드가 필요하며, 이것은 metric loss 하나를 §6.1 병렬 forward에 더하는 것으로 토큰 규약·메모리 상태를 바꾸지 않는다. D2 probe가 z_ext만으로 겹침 EER이 충분하다고 나오면 이 헤드도 뺀다. (3) 화자별 미래 활동·hazard 헤드는 lane 단위가 된다. N≤6에서는 lane=화자이므로 손실이 없고, ICSI·NOTSOFAR N≥7에서만 재배정 뒤 예측이 끊긴다.

R=6·lazy-free는 이 변형에서도 유지한다. R=4 eager는 lane 부족 1.8–2.2 %·EOT 결합 모호 3–8 %에 더해 lane run이 짧아져 위 잔여 비율이 커진다.

## 11. EOT 즉시 방출과 soft label (2026-09-15 결정)

[[decision-eot-immediate-soft-label]]에 따라 §2·§3.2·§3.4·§6 의 `<SEG_END>` 서술을 이 절이 대체한다. lane 규약(§3.1 lazy-free)·R=6·병렬 학습·데이터 배치는 그대로다.

### 11.1 규칙

1. **토큰.** 구조 토큰은 `<ONSET>`·`<EOT>` 뿐이다. `<SEG_END>` 는 등록하지 않는다. 정본 registry 에 더할 것은 lane 3–6 토큰뿐이다.
2. **후보 위치.** 각 발화 구간(화자별 VAD, gap <0.25 s 병합)의 끝에서 `k_eot = max(k_last_text, floor((offset+0.24)/0.08))` 청크, 그 구간의 마지막 lexical 직후에 `<SPK_r><EOT>` 후보를 둔다. 방출 지연은 offset 기준 약 320 ms + 연산이며 미래 관측을 기다리지 않는다.
3. **lane 닫힘.** EOT 가 방출되면 lane 은 HELD 가 된다. 방출되지 않았어도 활동 헤드가 그 lane 을 0.25 s 이상 비활성으로 보면 HELD 로 둔다. 재개·재배정은 §3.1 그대로다. EOT 오방출·미방출은 lane 소유를 바꾸지 않으므로 오귀속으로 번지지 않는다.
4. **귀속.** EOT 는 그 lane 에서 방금 닫힌 구간의 것이다. 3 s 대기가 없으므로 §3.2 의 "재배정 뒤 미결 EOT" 사례와 pointer head 는 더 이상 존재하지 않는다.
5. **후보 이후.** v1 에서는 후보 위치 하나에서만 결정한다. 그 뒤 본인 재개 없이 침묵이 이어지면 정본의 `timeout_policy`(emission_source 구분)가 맡는다. 후보 뒤 청크에서 재결정을 허용하는 변형은 ablation 이다.

### 11.2 soft target

라벨은 구간 끝 뒤 3 s 의 참조 결과로 만든다(미래는 정답 산출에만 쓴다). 후보 위치의 next-token target 은 두 점 분포다: `<EOT>` 에 p_end, 참조열에서 EOT 를 건너뛴 다음 토큰에 1−p_end.

| 구간 끝 뒤 3 s 의 결과 | p_end (초기값, Q0 동결) |
|---|---:|
| 교대: 다른 화자가 말하고(끝 시점에 겹치거나 3 s 안 시작) 본인 재개 없음 | 1.0 |
| 침묵: 아무도 말하지 않음 | 0.8 |
| 혼재: 다른 화자도 말하고 본인도 3 s 안 재개 | 0.5 |
| 유지(긴 pause): 본인만 1–3 s 뒤 재개 | 0.3 |
| 유지(짧은 pause): 본인만 1 s 안 재개 | 0.0 |
| 미래 미관측(EOF·결손) | mask |

EOT 는 teacher-forced 입력열에 항상 넣고, 그 토큰을 **예측하는 위치**의 label 만 soft 로 둔다. 구현은 collator 가 해당 위치에 `(label_alt, weight=p_end)` 를 추가하고 forward 의 CE 를 `p·CE(EOT) + (1−p)·CE(label_alt)` 로 합치는 것이다. 위치 가중치는 정본대로 EOT=2. hard 대조(교대=1, 유지=0, 혼재·침묵 mask)를 같은 데이터로 ablation 한다.

### 11.3 왜 soft 인가: 구간 끝 뒤 실제 결과

같은 어노테이션(§3.3)에서 각 구간 끝 뒤 3 s 를 분류했다(`raw/sources/experiments/2026-09-15-phase2-lane-sim/segment_end_outcomes.out`).

| 코퍼스 | 교대 | 침묵 | 혼재 | 유지 1–3 s | 유지 <1 s |
|---|---:|---:|---:|---:|---:|
| AMI | 54.2 % | 3.4 % | 35.5 % | 5.9 % | 1.0 % |
| ICSI | 36.7 % | 0.8 % | 40.4 % | 4.5 % | 17.7 % |
| NOTSOFAR-1 | 44.4 % | 0.5 % | 50.4 % | 0.2 % | 4.4 % |
| 71631 (KO 2인, 10,961대화) | 29.0 % | 4.7 % | 35.0 % | 9.3 % | 21.9 % |

구간 끝 뒤 3 s 안에 본인이 다시 말하는 경우(혼재+유지)가 회의에서 42–62 %, KO 2인 대화에서 66 % 다. 모든 구간 끝을 hard EOT 로 두면 절반이 오라벨이지만, soft target 이면 모델이 배우는 것은 "이 시점의 단서로 본 종료 확률"이고 p(EOT) 가 그 신뢰도다. 임계값·bias 는 런타임 정책이다. ICSI 의 "유지 <1 s" 가 큰 것은 어노테이션 구간이 더 잘게 나뉜 탓이며 정렬 후 재계산한다.

### 11.4 평가

- 후보 위치에서 p(EOT) 의 AUC 와 calibration(ECE): 양성 = 교대∪침묵, 음성 = 유지, 혼재는 제외.
- 임계값별 precision/recall 과 **유지 구간 내 오방출률**(barge-in 위험의 직접 지표). 코퍼스·언어·N 별.
- 지연은 offset+320 ms + 연산으로 고정이므로 별도 분포 대신 C-mode ablation 과의 지연 차를 보고.
- TurnBench dev 는 이 P 출력으로 채점하고, 기존 C-mode 결과와 섞지 않는다.
- D1 관문(§9)에 "유지 구간 내 오방출률 ≤ 임계(Q0 에서 dev 로 동결)"를 추가한다.
