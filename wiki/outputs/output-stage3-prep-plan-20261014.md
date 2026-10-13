---
type: output
status: active
created: 2026-10-14
updated: 2026-10-14
summary: Stage 3(2 인 같은 언어 대화의 화자 귀속 + ASR + 턴 끝 EOT) 준비 계획 — Phase 2 정제 대화·정렬·혼합·창을 다시 직렬화하고, 독일어 KEC·ZTC 를 새로 붙이고, 회의 코퍼스는 2 화자만 골라 섞는다
sources:
  - '[[decision-cst-training-stages-and-speaker-slots]]'
  - '[[output-phase2-training-db]]'
  - '[[output-phase2-lane-plan]]'
related:
  - '[[decision-eot-immediate-soft-label]]'
  - '[[output-phase2-eval-plan]]'
  - '[[output-q06-s4-ast-eval-20261013]]'
raw_authors:
  - tskim
---

# Stage 3 준비 계획 (2026-10-14)

## 목표
- **시작점:** Stage 2 최종 체크포인트(q06-s4-ast/final)
- **학습 내용:** 2 인 같은 언어 대화의 화자 귀속 + ASR + 턴 끝 `<EOT>`
- **유지할 것:** Stage 2 단일 화자 번역 데이터를 섞어 번역 능력을 지킨다. 같은 시기에 Stage 4 용 대화 번역 타깃을 만든다.
- **근거:** [[decision-cst-training-stages-and-speaker-slots]]

## 형식
- **prefix:** `language <SPK_A>{언어} <SPK_B>{언어}<asr_text>`
  - Stage 3 는 두 슬롯이 같은 언어다. 그래서 먼저 말한 사람이 A 다.
  - Stage 4 에서 언어가 달라지면 언어로 슬롯을 고정한다.
- **출력:** 화자가 바뀔 때 `<SPK_x>` 를 내고, 그 화자의 턴이 끝나면 `<EOT>` 를 낸다.
  - 직렬화: `build_interleaved_events` 의 화자 태그 + 시점 고정 이벤트(`(tid, t, k_min)`)
- **`<EOT>` 규칙:** 구간 끝 즉시 내고, soft label 이다([[decision-eot-immediate-soft-label]], `vapasr/data/eot_soft.py`).
  - 교대 1.0 · 침묵 0.8 · 혼재 0.5 · 본인 1–3 s 뒤 재개 0.3 · 본인 1 s 안 재개 0.0
  - 지금 `modeling_vapasr.forward` 는 soft label 을 lane 모델(`config.lanes > 0`)에서만 받는다. lane 이 없을 때도 받도록 고친다.
- **쓰지 않는 것:** lane 토큰 `<SPK_3..6>`, `<ONSET>`, 활동 헤드

## 데이터
Phase 2 산출물은 mxc `VAPKT-data/data/phase2/` 에 있다. 정제 대화·단어 정렬(asr-tn-v1)·mono 혼합·split v1·20–40 s 창이 다 만들어져 있다.

| 언어 | 코퍼스 | 대화·시간 | 창(20–40 s, hop 10 s) | 상태 | 할 일 |
|---|---|---|---|---|---|
| KO | 71631 실내 stereo | 757 · 196.6 h | 22,426 · 187 h (2 인 94 %) | 준비됨 | 새 형식으로 다시 직렬화 |
| KO | 134-1 성인 실외 조각 | 1,492 · 344 h | 29,528 · 246 h | 준비됨 | 같음 |
| KO | 134-2 청소년 | 1,649 · 400 h | 49,911 · 419 h | 준비됨(마스크 평균 32 %) | 같음 |
| EN | otoSpeech | 420 · 105 h | 12,522 · 105 h | 준비됨 | 같음 |
| EN | AMI·ICSI·NOTSOFAR | 171·75·237 회의 · 196 h | 3 인 이상 창이 대부분 | 정제·정렬됨 | **두 화자만 골라 그 둘의 채널만 섞는 창 생성기**, 채널 누설 검사 |
| DE | KEC(남부 독일어 자유 대화) | 79 명 · 40 h | – | 원자료만(44.1 kHz, 화자별 wav + TextGrid) | **어댑터** → 정제 → 정렬 → 혼합 → 창 |
| DE | ZTC(스위스식 표준 독일어 과제 대화) | 29 쌍 × 3 세션 · 42 h | – | 원자료만(화자별 wav + TextGrid + MAuS 정렬) | 같음 |
| KO·EN·DE | 이어 붙인 합성 2 인 대화 | 생성 | – | `dialogue_stitch.py` 있음 | 단일 화자 Stage 2 데이터로 생성. 독일어 대화가 적은 것을 메운다. Stage 4 교차 언어에도 쓴다 |

- **NSC(독일어, 155 h)는 뒤로 미룬다.**
  - 대화는 전화 문의를 흉내 낸 것이고, 상대는 거의 한 사람(279 세션)이다.
  - 두 사람 채널이 다 있는 부분(Dialog 5–8)에는 전사가 없다.
  - 쓰려면 교사 ASR 로 의사 전사를 만들어야 한다.
- **언어 불균형:** 실제 대화는 한국어 약 850 h, 영어 약 105 h + 회의, 독일어 약 80 h 다.
  - 합성 대화와 샘플링 비율로 맞춘다.
  - 비율은 파일럿에서 정한다.
- **Replay:** Stage 2 단일 화자 데이터(ASR + 번역 모드)를 섞는다. 비율은 파일럿에서 정한다.

## 평가
- **held-out(split v1, 학습 제외):**

  | 코퍼스 | 기준 | 규모 |
  |---|---|---|
  | 71631 | 화자 5 % 가 낀 대화 | 59 대화 · 14.3 h |
  | 134-1 | 같음 | 155 대화 · 35.5 h |
  | 134-2 | 같음 | 223 대화 · 50.8 h |
  | otoSpeech | 배우 5 % | 38 대화 · 9.8 h |
  | AMI | 공식 test | 20 회의 · 11.5 h |
  | NOTSOFAR | dev | 36 · 3.7 h |
  | ICSI | 표준 eval | 6 회의 · 5.8 h |

  - KEC·ZTC 도 화자(쌍) 기준으로 같은 방식으로 뗀다.
  - 학습에 쓰지 않은 NIKL 2020 은 한국어 외부 평가로 쓴다.
- **지표([[output-phase2-eval-plan]] 의 4 축을 2 인으로 줄인 것):**
  - 인식: 화자와 무관한 ORC-WER
  - 화자 귀속: cpWER, 태그 혼동률
  - 턴: `<EOT>` 정밀도·재현율·지연, p(EOT) 보정
  - 퇴행 검사: Stage 2 평가 묶음(단일 화자 ASR·번역)을 같은 방식으로 다시 잰다
- `experiments/p2_eval_sessions.py` 를 새 형식에 맞춰 고쳐 쓴다.

## 작업 순서와 관문
| 단계 | 작업 | 통과 기준 |
|---|---|---|
| P0 점검 | Phase 2 산출물 재확인(창·마스크·오디오 조립을 로더 경로로 표본 점검), held-out 대화가 Stage 1·2 학습 목록에 없는지 대조(71631 발화 crop 이 Stage 1·2 에 들어가 있다) | 누설 0, 조립 오류 0 |
| P1 독일어 | KEC·ZTC 어댑터(TextGrid → dialogues.jsonl), 정제·정렬(ZTC 는 MAuS 활용)·혼합·창, 화자 기준 split | 표본 청취·정렬 검사 통과 |
| P2 회의 2 화자 | 화자 쌍 선택 창 생성기, 채널 누설 검사(다른 화자 구간의 채널 에너지) | 누설 기준 통과한 쌍만 사용 |
| P3 직렬화 | 2 슬롯 prefix + 화자 태그 + soft `<EOT>` 데이터셋, fixture 테스트, 모델 forward 의 soft label 일반화, 전 코퍼스 dry-run | 불일치 0, 형식 테스트 통과 |
| P4 평가 | held-out 묶음·지표 도구(ORC-WER·cpWER·EOT) | 오라클 입력에서 지표 sanity |
| P5 파일럿 | 소규모 학습(32 창 overfit → 수천 step), replay·언어 비율 정하기 | 단일 화자 퇴행 없음, cpWER·EOT 학습 곡선 |
| P6 본 학습 | 사용자 제출 | – |
| 병행 | Stage 4 용 대화 번역 타깃(턴 단위 교사 번역, 간투사·말 끊김 규칙) | 교사 품질 검사 |

## 열린 질문
- `<EOT>` 를 soft 로 할지, 처음에는 hard(p_end ≥ 0.5)로 단순하게 시작할지(P5 에서 비교)
- 맞장구 처리: CST 첫 논문 범위에서는 사건으로 다루지 않는다. 전사로는 그대로 내고 `<EOT>` 대상에서는 뺄지 정해야 한다.
- 창 길이(20–40 s)를 그대로 쓸지, Stage 2 의 30 s 상한과 맞출지
- 134-2 의 높은 마스크 비율(평균 32 %)을 어떻게 다룰지(결손 구간 학습 제외 규칙은 Phase 2 그대로)
