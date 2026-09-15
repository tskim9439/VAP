---
type: output
status: active
created: 2026-09-15
updated: 2026-09-15
summary: Phase 2 정본(2026-09-15 채택) — lane 유지형(lazy-free, R=6) 다화자 스트리밍 전사, 구간 끝 즉시 EOT soft label, 세션 정체성은 외부 재매핑; 보유 DB 배치·학습·평가·D0–D5 관문
contributors:
  - tskim
sources:
  - '[[decision-phase2-canonical-lane-plan]]'
  - '[[decision-eot-immediate-soft-label]]'
  - '[[decision-multi-speaker-scope]]'
  - '[[decision-mono-input]]'
  - '[[output-phase2-dynamic-speaker-memory-plan-v2]]'
  - '[[output-phase2-lane-proposal-report]]'
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[output-phase2-training-db]]'
  - '[[output-phase2-data-inventory]]'
  - '[[task-secure-meeting-corpora]]'
---

# Phase 2 정본 계획: Lane 유지형 다화자 스트리밍 전사

채택: 2026-09-15, 사용자 결정 [[decision-phase2-canonical-lane-plan]]. 이 문서가 이전 정본 [[output-phase2-streaming-asr-diarization-plan]](2026-09-14 개정판)을 대체한다. 이전 정본 중 여기서 바꾸지 않은 절(§3.2 audio-clock 헤드, §4.2 직렬화 순서 계약, §4.4 블록 계약의 무음·flush 규칙, §5.2 스키마, §5.3 파이프라인, §6.2 80 ms 정합, §8 평가 원칙, §9 구현 지도·QC, §10 예산)은 **참조로 계속 유효**하며, 충돌하면 이 문서가 우선한다. 근거 분석과 실측은 [[output-phase2-dynamic-speaker-memory-plan-v2]], 그림 설명은 [[output-phase2-lane-proposal-report]]에 있다.

## 1. 한 줄 정의와 범위

모델은 **"지금 말하는 사람들을 서로 다른 lane 에 나눠 적는 것"**(전사·발화 구간·구간 끝 즉시 EOT)까지 책임진다. "이 사람이 아까 그 사람인가"(세션 정체성)는 lane 을 오래 유지하는 규칙과 모델 밖의 화자 재매핑 모듈이 맡는다.

| 항목 | 값 |
|---|---|
| 입력 | mono 16 kHz, E2 encoder(동결, [56,0]) + adapter + Qwen3 thinker, 80 ms audio clock, δ_text=4 주·2 보조 ([[decision-mono-input]]) |
| 언어 | KO·EN |
| 화자 수 | 세션 인원 N 상한 없음(외부 재매핑), lane 수 **R=6**, 순간 겹침은 학습 자료가 허용하는 범위(S≤4) |
| 구조 토큰 | `<ONSET>`, `<EOT>`, `<NEXT_AUDIO>`, `<EMPTY_AUDIO>` — 이전 정본과 같음. `<SEG_END>`·`<EVENT_REF>`·화자 메모리 없음 |
| lane 토큰 | lane 1/2 = `<SPK_A>`(151707)/`<SPK_B>`(151708) 재사용, lane 3–6 = `<SPK_3..6>` 신규 등록 |
| EOT | 구간 끝 즉시(offset+δ, 약 320 ms), soft target ([[decision-eot-immediate-soft-label]]) |
| 헤드 | lane 별 활동(현재), Q3 에서 미래 4-bin·hazard. 모두 lane 행(6행) |
| 학습 | 이전 정본과 같은 단일 병렬 teacher-forced forward |
| Stage 3 이월 | `<HOLD>`·`<BC>`·의미 라벨(변경 없음) |

## 2. 세 가지 수: N · S · R

- **N** 세션 전체 인원, **S(t)** 순간 동시 발화 수, **R** 모델이 동시에 붙잡는 전사 채널(lane) 수. 이전 정본의 K 는 N 과 R 을 하나로 묶었다. 이 정본은 R 을 고정하고 N 을 모델 밖으로 보낸다.
- 보유 자료: 2인(71631·134-1·otoSpeech·Switchboard 후보), AMI 3–5인(163/171 이 4인), NOTSOFAR-1 3–8인, ICSI 3–10인. S≥3 은 음성 프레임의 3.9 %(AMI)·4.2 %(ICSI)·9.0 %(NOTSOFAR-1).

## 3. lane 규약 (lazy-free)

### 3.1 배정 규칙

1. 새 발화 구간이 시작하면 그 화자가 **현재 소유한 lane 이 있으면 그 lane**(구간만 새로 연다).
2. 없으면 **FREE lane** 중 첫 번째.
3. FREE 가 없으면 **HELD lane 중 `closed_at` 이 가장 오래된 lane** 을 새 화자에게 넘긴다(`generation+1`). 이전 소유자의 구간 기록은 남는다.
4. HELD 도 없으면 `lane_capacity_exhausted` 를 보고하고 덮어쓰지 않는다.

**따름정리.** N≤R 이면 규칙 3 이 발동하지 않으므로 lane i = 도착순 i번째 화자이며, 토큰열은 이전 정본의 K=6 슬롯 시퀀스와 같다(EOT 후보 위치만 구간 끝). 따라서 이 정본의 모델은 이전 정본 K=6 모델과 같은 데이터·registry·초기 backbone 으로 학습되고, 두 allocator 정책(`never_free` / `lazy_free`)의 출력이 N≤R 에서 같음을 테스트로 고정한다.

### 3.2 lane 상태

lane 은 `FREE / OPEN / HELD` 와 `owner, generation, closed_at` 을 가진다.

| 전이 | 라벨 측(reference allocator) | 추론 측(runtime parser) |
|---|---|---|
| FREE → OPEN | 참조 segment 시작 | `<SPK_r><ONSET>` 방출 |
| OPEN → HELD | 참조 화자 VAD 가 0.25 s 이상 꺼진 시점 = segment 끝 | (a) `<SPK_r><EOT>` 방출, 또는 (b) lane r 활동 헤드가 마지막 lexical/ONSET 이후 0.25 s(3 청크) 이상 임계 미만. 먼저 오는 것. `closed_at` 기록 |
| HELD → OPEN | 같은 화자의 다음 segment | `<SPK_r><ONSET>` (규칙 1) |
| HELD → OPEN, generation+1 | FREE 없음 + 새 화자 → `closed_at` 최소 lane | 같음(규칙 3) |
| HELD → FREE | 없음 | 없음(시간 경과로 해제하지 않음) |

HELD 는 "구간은 닫혔지만 소유는 남은" 상태다. 규칙 3 의 후보는 HELD 뿐이므로 닫힘이 관측되지 않은 OPEN lane 은 재배정되지 않는다. EOT 뒤 같은 lane 에 ONSET 없이 늦은 lexical 이 오면 보존하고 프로토콜 지연 오류로 계수하며 `closed_at` 을 갱신한다. 0.25 s 는 라벨의 segment 병합 gap 과 같은 값이다.

**오류의 국소성.** EOT 오방출·미방출, 활동 헤드 오판은 lane 소유를 바꾸지 않으므로 다른 화자의 전사로 번지지 않는다. 화자 오귀속(`<SPK_A>` 대신 `<SPK_3>`)은 그 payload 의 귀속 오류로 계수되며 역시 lane 소유를 바꾸지 않는다.

### 3.3 R=6 의 근거

AMI 171·ICSI 75·NOTSOFAR-1 237 회의 어노테이션에 lazy-free 를 적용한 실측(`raw/sources/experiments/2026-09-15-phase2-lane-sim/`):

| 코퍼스 | N | R=4 lane 부족 | R=4 재배정 | R=6 재배정 | R=8 재배정 |
|---|---|---:|---:|---:|---:|
| AMI | 3–5 | 0.01 % | 0.17 % | 0 | 0 |
| ICSI | 3–10 | 1.83 % | 10.9 % | 2.41 % | 0.19 % |
| NOTSOFAR-1 | 3–8 | 2.23 % | 17.8 % | 2.26 % | — |

R=4 는 재배정이 11–18 % 로 lane run 이 짧아져 §7 의 외부 재매핑 근거가 준다. R=8 은 이득이 0.2 %p 뿐이다. ICSI 는 비음성 segment 포함, NOTSOFAR-1 은 발화 단위 GT 병합값이며 정렬 후 재계산한다.

## 4. 직렬화와 토큰열

이전 정본 §4.2·§4.4 의 시각순 교차 직렬화·`(target_chunk, reference_sample, slot, kind_priority, source_ordinal)` 정렬·무음 블록·flush·cap 규칙을 그대로 쓴다. kind_priority 는 ONSET < lexical < EOT. 모든 ONSET/EOT 앞에 lane 태그를 둔다.

설명용 예시(R=3 으로 줄임, 4번째 화자 영희 등장). 실제 R=6 이면 영희는 lane 4 를 받고 재배정은 없다.

```text
[AUDIO_k]    <SPK_A><ONSET> 안녕하세요 저는 <SPK_B><ONSET> 네 <NEXT_AUDIO>              # 민수 lane 1, 지수 맞장구 lane 2(EOT 후보 p≈0)
[AUDIO_k+n]  <SPK_A> 김입니다 <SPK_A><EOT> <NEXT_AUDIO>                                # 민수 구간 끝 즉시 EOT 후보, lane 1 은 HELD
[AUDIO_j]    <SPK_A><ONSET> 그리고요 <SPK_A><EOT> <NEXT_AUDIO>                          # 같은 화자 재개 = 같은 lane
[AUDIO_m]    <SPK_B><ONSET> 그렇군요 <SPK_3><ONSET> 동의해요 <SPK_B> 그러면 <NEXT_AUDIO>  # 철수 시작 → FREE 인 lane 3, 시간순 교차
[AUDIO_m+n]  <SPK_B><EOT> <SPK_3> 다만 <SPK_3><EOT> <NEXT_AUDIO>                       # 각자 자기 lane 의 EOT
[AUDIO_p]    <SPK_A><ONSET> 잠깐만요 <SPK_A><EOT> <NEXT_AUDIO>                          # FREE 없음 → closed_at 최소인 lane 1 을 영희가 받음
```

## 5. EOT: 구간 끝 즉시, soft target

1. **후보 위치.** 각 발화 구간(화자별 VAD, gap <0.25 s 병합) 끝에서 `k_eot = max(k_last_text, floor((offset+0.24)/0.08))`, 그 구간의 마지막 lexical 직후. 미래 관측을 기다리지 않는다.
2. **target.** 구간 끝 뒤 3 s 의 참조 결과(미래는 정답 산출에만 사용)로 p_end 를 정하고, 후보 위치의 next-token target 을 `<EOT>`: p_end, 참조열에서 EOT 를 건너뛴 다음 토큰: 1−p_end 의 두 점 분포로 둔다.

| 구간 끝 뒤 3 s | p_end(초기값, Q0 동결) |
|---|---:|
| 교대: 다른 화자가 끝 시점에 겹치거나 3 s 안 시작, 본인 재개 없음 | 1.0 |
| 침묵: 아무도 말하지 않음 | 0.8 |
| 혼재: 다른 화자도 말하고 본인도 3 s 안 재개 | 0.5 |
| 유지: 본인만 1–3 s 뒤 재개 | 0.3 |
| 유지: 본인만 1 s 안 재개 | 0.0 |
| 미래 미관측(EOF·결손) | mask |

3. **구현.** EOT 는 teacher-forced 입력열에 항상 넣고, 그것을 예측하는 위치의 label 만 soft 로 둔다. collator 가 `(label_alt, weight=p_end)` 를 붙이고 forward 의 CE 를 `p·CE(EOT) + (1−p)·CE(label_alt)` 로 합친다. 위치 가중치는 EOT=2, 그 외 lexical/SPK/ONSET=1, NEXT EN/KO=0.3/0.15.
4. **추론.** `p(EOT)` 를 API 에 `turn_end_prob` 로 노출하고 방출 임계값·bias 는 런타임 정책이다. v1 은 후보 위치 하나에서만 결정하며, 이후 침묵은 `timeout_policy`(emission_source 구분)가 맡는다. 후보 뒤 청크의 재결정은 ablation.
5. **왜 soft 인가.** 구간 끝 뒤 3 s 안에 본인이 다시 말하는 비율(혼재+유지): AMI 42 %, ICSI 62 %, NOTSOFAR-1 55 %, 71631 KO 2인 66 %(`segment_end_outcomes.out`). hard EOT 는 절반 이상이 오라벨이지만 soft 면 모델이 배우는 것은 "이 시점 단서로 본 종료 확률"이다.
6. **C-mode(3 s 관측 뒤)**는 ablation 으로 내려간다. 결과를 섞어 보고하지 않는다.

## 6. 헤드

이전 정본 §3.2 그대로: `[AUDIO_k]` hidden state 위의 작은 MLP. Q1 은 lane 별 현재 활동 6 sigmoid(§3.2 (b) 닫힘 판정에도 쓴다), Q3 은 lane 별 미래 4-bin BCE·hazard. 미등장 lane 은 mask. 화자 메모리 조건 헤드는 두지 않는다. 재배정된 lane 의 미래 예측은 새 소유자 기준이며, N>R 에서만 사람 단위 예측이 끊긴다.

## 7. 세션 정체성: 외부 재매핑

- 모델 출력 단위는 `(lane, generation, episode_id, 시작·끝 청크, 토큰, p(EOT))` 다. episode_id 는 재사용하지 않는다.
- 외부 모듈은 **lane run**(같은 generation 의 연속 구간)을 단위로 화자 임베딩을 모아 온라인 클러스터링(same/new/unresolved)한다. 첫 구현은 공개 화자 인코더를 mono 의 각 구간 비겹침 부분에 적용한 것이다.
- lane 을 유지하는 이유가 여기 있다. 구간 하나만 보면 비겹침 증거 <0.5 s 인 구간이 AMI 53 %·ICSI 49 %·NOTSOFAR-1 66 % 이지만, lane run 으로 모으면 1.0 %·6.1 %·10.8 % 다(`clean_evidence.out`).
- 잔여 1–11 % 와 N>R 재배정(2.3 %)의 세대 변경 검출이 외부 모듈의 한계다. 이를 줄이는 유일한 모델 측 수단은 payload 조건 표현을 구간 임베딩으로 내보내는 헤드(metric loss 하나, 토큰 규약 불변)이며, **D2 probe 결과로 넣거나 뺀다.**
- 평가는 speaker_id 로 regroup 한 뒤 cpWER/DER 로 하며 lane 을 사람으로 채점하지 않는다. 최초 출력과 수정 후를 분리 보고한다.

## 8. 데이터

A등급(화자별 채널 또는 채널 기원 조각)만 쓴다. 화자별 채널에 Qwen forced aligner 로 시각을 다시 뽑고, 모델 입력은 mono 혼합이다([[output-phase2-training-db]]).

| 자료 | 언어 | N | 규모 | 역할 | split |
|---|---|---|---|---|---|
| 71631 실내 stereo 원본 | KO | 2 | 757대화 196.6 h | 전사·EOT·2인 재개(E2 노출 감사) | 정본 split, VS_02 dev |
| 134-1 실외 조각 | KO | 2 | 1,492대화 ≤344 h | 위와 같음, 새 KO test 후보 | Q0 |
| 134-2 청소년 실외 조각(완전 대화) | KO | 2 | 523대화 90.1 h (+≥95 % 427대화 75.4 h 는 결손 주변 마스크) | 채널 기원 검증 통과(2026-09-15), 성인과 같은 역할 | Q0 |
| 134-2 청소년 실내 조각 | KO | 2 | ≥80 % 대화 4,095 782.8 h | 전사·활동 감독만, turn 감독은 완전 창 한정 | Q0 |
| otoSpeech | EN | 2 | 420대화, train ≈90 h | 위와 같음 | actor |
| Switchboard(조건부) | EN | 2 | ≈230 h | 파일명 시각 검증 통과 시 | 원 split |
| AMI headset | EN | 3–5 | 171회의 98.1 h | N=4 lane 유지, S≤4 | 공식 |
| NOTSOFAR-1 close-talk | EN | 3–8 | 237회의 24.2 h | N 5–8, 재배정 2.3 %, S≥3 9 % | 공식 train/dev1/eval |
| ICSI headset | EN | 3–10 | 75회의 71.6 h | N≥6 재배정 주 공급원 | 회의 계열 hold-out |
| DiPCo · CHiME-6 | EN | 4 | 5 h · ≈40 h | 평가 · 누설 검사 후 보조 | 공식 |
| 2화자 대화 결합(stitching) | KO/EN | 4–8 | 생성 | KO 다자·재배정 사례. **EOT 마스크**, `synthetic=stitch` | — |

KO 에는 3인 이상 A등급 자료가 없다. KO 다자 성능은 합성 조건 결과로만 보고한다. 중복 제거·split·라이선스는 이전 정본 §5 와 [[output-phase2-training-db]]를 따른다.

**AI Hub 라벨 처리(2026-09-15, [[decision-aihub-transcript-policy]]).** (1) 모든 A등급 자료는 화자 채널 에너지 VAD + 정렬 토큰으로 발화 경계를 다시 잡는다(`p2_refine.py`; 71631 라벨 끝 +0.86 s 보정, 긴 발화는 0.25 s 이상 pause 에서 분할). (2) 전사 감독은 라벨·ASR 대조 CER < 0.2 인 발화만; 결손 의심(글자/초 < 1.5)·불일치 발화는 텍스트를 비운다. (3) 전사 없는 발화가 덮는 청크는 창을 유지한 채 payload·NEXT label 을 전부 마스크한다(활동 타깃 유지). (4) 71631 실내 원본·otoSpeech 가 전사 감독의 주력이고 실외 조각은 활동·턴 감독 위주다. 단일 ASR pseudo-label 은 쓰지 않는다.

## 9. 학습

- **단일 병렬 forward.** lane 라벨은 reference allocator(§3.1 을 정답 segment 에 적용)가 만들므로 입력열이 고정이다. 모델 출력이 decoder 입력으로 되먹임되지 않는다.
- **손실.** `L = L_AR(soft EOT 포함) + λ_act L_activity (+ Q3: λ_fut L_future + λ_haz L_hazard) (+ 조건부: λ_emb L_metric)`. 켜는 순서: D1 AR+활동 → Q3 헤드 → (조건부) 임베딩 헤드.
- **Trainer.** 이전 정본 §9 의 HF Trainer+Liger 경로 유지. 변경점은 collator 의 soft label 과 lane 3–6·활동 헤드뿐이다. 동결 E2 encoder 의 save/load parity 검사 유지.

## 10. 평가

- 인식: EN WER / KO CER(lane 태그 제거), speaker-attributed cpWER(외부 재매핑 뒤 regroup), lane 을 사람으로 채점한 값은 상한 참고.
- 화자: 재배정 세대 변경 검출 정확도(ICSI·NOTSOFAR N≥7), unresolved 비율, 최초 ID 확정 지연.
- EOT: 후보 위치 p(EOT) 의 AUC·calibration(양성 교대∪침묵, 음성 유지, 혼재 제외), 임계값별 precision/recall, **유지 구간 내 오방출률**(barge-in 위험), 지연은 offset+320 ms + 연산. TurnBench dev 는 이 P 출력으로 채점.
- 시스템: tick p99, RTF<1, backlog 비발산, 10–60 분 자유실행의 lane 부족률·ID switch.
- 세트: TurnBench dev, AMI test, ICSI hold-out, NOTSOFAR-1 dev1/eval, DiPCo, 71631 VS_02(E2 dev 재사용 명시), KO stitched(합성 표시).

## 11. 구현 지도

| 위치 | 책임 |
|---|---|
| `vapasr/data/dialogue_interleave.py`(신규, 이전 정본 §9) | 시각순 직렬화, EOT 후보 삽입 |
| `vapasr/data/lane_alloc.py`(신규) | reference allocator `never_free` / `lazy_free`, generation·episode·closed_at |
| `vapasr/data/eot_soft.py`(신규) | 구간 끝 결과 분류 → p_end, collator 의 `(label_alt, weight)` |
| `vapasr/hf/lane_state.py`(신규) | 추론 parser: §3.2 상태 전이, capacity 보고, episode 출력 |
| `vapasr/hf/modeling_vapasr.py` | soft CE, 활동 헤드, lane 3–6 registry·mask, `stream_decode` 의 parser 연결 |
| `vapasr/data/targets.py` | Q3 lane 행 미래 4-bin·hazard |
| `experiments/p2_speaker_probe.py`(신규) | D2 probe |
| `experiments/p2_external_remap.py`(신규) | lane run 임베딩·온라인 클러스터링·regroup 평가 |
| `tests/test_lane_protocol.py`(신규) | §12 fixture |

## 12. 필수 fixture

이전 정본 §4.4 의 16 사례를 EOT 즉시 규칙으로 갱신해 유지하고, 다음을 더한다.

17. 시작 무음은 AUDIO/NEXT 만.
18. 한 사람 발화→EOT→재개가 같은 lane·새 episode.
19. 2/3/4 명 겹침에 서로 다른 lane, audio 토큰은 하나.
20. N≤R 세션에서 `never_free` ≡ `lazy_free` 출력.
21. FREE 없음 → closed_at 최소 lane 재배정, generation+1, 이전 episode 보존.
22. OPEN lane 은 재배정 후보 제외 → `lane_capacity_exhausted`.
23. EOT 뒤 ONSET 없는 늦은 lexical 보존·지연 오류 계수.
24. soft target: p_end 별 두 점 label, 미관측 mask, 손실 오염 없음.
25. 동일 prefix·다른 미래 suffix 에서 현재 입력·출력 동일(정답만 달라짐).
26. checkpoint 재개·불균등 DDP·crop EOF carry parity.
27. legacy mono decoder 로 잘못 로드 시 명시적 실패.

## 13. 실행 단계와 관문

| 단계 | 작업 | 통과 기준 / 중단 |
|---|---|---|
| **D0** 프로토콜 | allocator 두 정책·lane parser·EOT 후보 직렬화·soft collator·fixture | 전부 통과, N≤R 동등성, 소유자 오류 0. GPU 불필요 |
| **D2** 표현 probe(D0 와 병행) | E2 체크포인트로 71631·AMI 혼합의 화자 임베딩 EER(비겹침/겹침, 증거 길이별, 60 s 재등장) | 비겹침 외부 인코더 EER ≤10 %. 겹침 결과로 구간 임베딩 헤드 채택 여부 결정 |
| **Q0** 데이터 | 정렬·QC·manifest(71631·134-1·otoSpeech·AMI·NOTSOFAR·ICSI), stitching 생성기, p_end 표 동결 | 이전 정본 Q0 관문 + NOTSOFAR/ICSI 정렬 coverage |
| **D1** 학습 | K=6·lazy_free·EOT soft·활동 헤드. 32창 overfit → 실학습 | 이전 정본 K=6 대비 WER/CER·cpWER 회귀 ≤5 %; 유지 구간 내 오방출률 ≤ Q0 동결 임계; p(EOT) calibration; lane 부족률 |
| **D3** 외부 재매핑 | lane run 임베딩·온라인 클러스터링·regroup 평가 | N≤6 에서 lane=사람 대비 회귀 ≤2 %, N≥7 세대 변경 검출 보고 |
| **Q3** 미래 헤드 | lane 행 4-bin·hazard | 이전 정본 Q3 기준 |
| **D5** 장문 | 60–120 s carry → 10–60 분 자유실행 | RTF<1, backlog 비발산, ID switch·lane 부족 보고 |

모델 내 화자 메모리(원안 D3/D4)는 범위 밖이며, D3 외부 재매핑이 N≥7 에서 실측으로 부족할 때만 재검토한다.

## 14. 불확실성

- 겹침 구간 화자 표현 분리는 미검증(D2 관문).
- soft EOT 의 정밀도는 관측 시점 단서가 허용하는 만큼이다. 부족하면 후보 위치를 offset+0.5–1.0 s 로 늦추는 변형을 같은 soft 규칙으로 비교한다.
- ICSI·NOTSOFAR 실측은 정렬 전 어노테이션 기준이다.
- CHiME-6 누설·71631 E2 노출 감사·Switchboard 시각 검증은 Q0 미완 항목이다.
- KO N>2 는 합성뿐이다.

## 근거

- [[decision-phase2-canonical-lane-plan]], [[decision-eot-immediate-soft-label]], [[decision-multi-speaker-scope]]
- [[output-phase2-dynamic-speaker-memory-plan-v2]] §3·§8·§10·§11 — 원안 검토·실측·검증
- [[output-phase2-lane-proposal-report]] — 그림 설명(아티팩트 포함)
- [[output-phase2-streaming-asr-diarization-plan]] — 유지되는 참조 절
- `raw/sources/experiments/2026-09-15-phase2-lane-sim/` — lane 시뮬레이션·비겹침 증거·구간 끝 결과 분류(AMI·ICSI·NOTSOFAR-1·71631)
