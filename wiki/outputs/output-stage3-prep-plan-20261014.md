---
type: output
status: active
created: 2026-10-14
updated: 2026-10-15
summary: Stage 3(2 인 같은 언어 대화의 화자 귀속 + ASR + 턴 끝 EOT) 준비 계획 — Phase 2 정제 대화·정렬·혼합·창을 다시 직렬화하고, 독일어 KEC·ZTC·NSC 를 새로 붙이고(P1 완료: 106 h, 전사 감독 62 h, 창 12,582 개), 회의 코퍼스는 2 화자만 골라 섞는다
sources:
  - '[[decision-cst-training-stages-and-speaker-slots]]'
  - '[[output-phase2-training-db]]'
  - '[[output-phase2-lane-plan]]'
  - '[[2026-10-14-stage3-german-data]]'
  - '[[2026-10-14-stage3-p3-serialization]]'
  - '[[2026-10-15-stage3-train-smoke]]'
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
| DE | KEC(남부 독일어 자유 대화) | 171 대화 · 29.2 h(두 화자가 다 있는 파트) | 2,444 · 20.3 h | **준비됨(P1)**. 축약 표기 → 표준 철자, 전사 감독 21.8 h | 새 형식으로 직렬화 |
| DE | ZTC(스위스식 표준 독일어 과제 대화) | 373 대화 · 41.9 h | 5,877 · 48.8 h | **준비됨(P1)**. 스위스식 ss → ß, 전사 감독 24.1 h | 같음 |
| DE | NSC(전화 문의형 반자연 대화, Dialog 5–8) | 276 세션 · 34.8 h | 4,261 · 35.7 h | **준비됨(P1)**. 교사 의사 전사(두 모델 일치 67.6 %), 전사 감독 16.4 h, 마스크 평균 23 % | 같음 |
| KO·EN·DE | 이어 붙인 합성 2 인 대화 | 생성 | – | `dialogue_stitch.py` 있음 | 단일 화자 Stage 2 데이터로 생성. 독일어 대화가 적은 것을 메운다. Stage 4 교차 언어에도 쓴다 |

- **NSC 는 Dialog 5–8 의 2 채널 대화를 쓴다(사용자 확인, 2026-10-14).**
  - 파일: `*_semispontaneousdialogs_{speaker,interlocutor}.wav`(276 세션). `interaction_##` 파일(267 세션, 3,375 쌍)은 같은 대화를 상호작용 단위로 자른 것이라 따로 세지 않는다.
  - 대본 대화(Dialog 1–4)는 화자 쪽 조각만 있어 2 인 대화로는 못 쓴다.
  - 전사가 없으므로 교사 ASR 로 의사 전사를 만들고, 두 모델 일치로 선별한다(YODAS-de 와 같은 방식).
  - 상대는 거의 한 사람(amsterdam 279 세션, aden 일부)이다. 같은 목소리에 치우치지 않게 비중을 조절하고, aden 세션을 일반화 확인용으로 뗀다.
  - 독일어 실제 대화는 KEC 29.2 h + ZTC 41.9 h + NSC 34.8 h = 106 h 다. 전사 감독이 걸린 발화 시간은 62 h 다(P1 결과).
- **언어 불균형:** 실제 대화는 한국어 약 850 h, 영어 약 105 h + 회의, 독일어 106 h 다.
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

  - 독일어는 split v2(`splits/v2.json` = v1 + 독일어)에 있다.
    - KEC: 녹음 쌍 4/39, 21 대화 · 3.5 h
    - ZTC: 화자를 공유하는 dyad 덩어리 3/27, 39 대화 · 4.5 h
    - NSC: 상대역이 aden 인 22 세션 · 3.3 h. 정답이 교사 전사라 전사 지표는 참고용이다.
  - 학습에 쓰지 않은 NIKL 2020 은 한국어 외부 평가 후보다(Stage 1 NIKL 표본과의 중복 확인 필요).
  - 71631 held-out 은 Stage 1·2 에 노출됐다(아래 P0).
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
| P1 독일어 | KEC·ZTC 어댑터(TextGrid → dialogues.jsonl), NSC 의사 전사(교사 ASR 두 모델 일치), 정제·정렬·혼합·창, 화자 기준 split | 표본 청취·정렬 검사 통과 — **2026-10-14 완료**(정렬·로더 점검 통과. 정렬은 세 코퍼스 모두 Qwen3-ForcedAligner 로 다시 했고 ZTC 의 MAuS 는 쓰지 않았다. 청취는 사용자 확인 필요) |
| P2 회의 2 화자 | 화자 쌍 선택 창 생성기, 채널 누설 검사(다른 화자 구간의 채널 에너지) | 누설 기준 통과한 쌍만 사용 |
| P3 직렬화 | 2 슬롯 prefix + 화자 태그 + soft `<EOT>` 데이터셋, fixture 테스트, 모델 forward 의 soft label 일반화, 전 코퍼스 dry-run | 불일치 0, 형식 테스트 통과 — **2026-10-14 완료**(113,389 창 불일치 0, GPU 스모크 통과. 학습 스크립트 연결은 P5) |
| P4 평가 | held-out 묶음·지표 도구(ORC-WER·cpWER·EOT) | 오라클 입력에서 지표 sanity — **2026-10-15 통과**(slot·cp 0, EOT 재현율 1) |
| P5 파일럿 | 소규모 학습(32 창 overfit → 수천 step), replay·언어 비율 정하기 | 단일 화자 퇴행 없음, cpWER·EOT 학습 곡선 — **2026-10-15 준비됨**(학습·평가 스모크 통과, 제출 대기) |
| P6 본 학습 | 사용자 제출 | – |
| 병행 | Stage 4 용 대화 번역 타깃(턴 단위 교사 번역, 간투사·말 끊김 규칙) | 교사 품질 검사 |

## P0 점검 결과 (2026-10-14)
- **71631 실내 757 대화는 모두 Stage 1·2 학습에 들어갔다.** split v1 의 held-out 59 대화도 포함이다.
  - Stage 1·2 의 `asr-aihub71631-train` 조각 228,264 개가 `Training/TS_01.실내_5` 757 대화 전부에서 나왔다.
  - 단일 발화 조각이고 2 인 혼합은 아니지만, 같은 음성과 전사를 이미 봤다.
  - → 71631 은 Stage 3 학습에만 쓴다. held-out 은 '본 음성' 진단용으로만 쓰고, 미노출 평가에서는 뺀다.
- **134-1·134-2 는 Stage 1·2 학습과 대화가 겹치지 않는다.**
  - 134-1 held-out 155 대화의 화자 96 명 중 10 명은 71631 실내에도 나온다(대화는 다르다). 미노출 평가에서는 이 화자가 낀 대화를 뺀다.
  - 134-2 held-out 223 대화(화자 130 명)는 71631 과 화자가 겹치지 않는다.
- **미노출 한국어 평가는 134-2 held-out(50.8 h)을 주로 쓰고, 134-1 held-out(겹치는 화자 제외)을 보조로 쓴다.**
  - NIKL 2020 은 Stage 1 의 NIKL 표본과 겹치는지 따로 확인한다.

## P1 독일어 결과 (2026-10-14)
- 원자료와 단계별 수치: [[2026-10-14-stage3-german-data]]
- **KEC 는 축약 표기로 전사돼 있어 표준 철자로 바꿨다.**
  - KEC 전사에는 nich·is·hab·des·n 같은 축약형이 그대로 있다(강제 정렬기 때문). Stage 1·2 독일어 타깃은 표준 철자다.
  - 바꾸는 방식(`dialogue_corpora.canon_de`):
    - 모호하지 않은 것은 고정 대응으로 바꾼다(nich→nicht, is→ist).
    - 모호한 것(des·n·ne·s·mer, das/dass)은 같은 자리의 ASR 단어가 허용된 표준형일 때만 바꾼다.
    - ZTC 의 스위스식 ss 는 ß 로 바꾼다.
  - ASR 대조 오류율 중앙값: KEC 0.25 → 0.083, ZTC 0.111 → 0.0
  - 축약 표기 원본은 `*.verbatim.dialogues.jsonl` 로 남겼다.
- **KEC 의 `NANA` 는 익명화된 이름이다.** 그 구간 오디오는 −20 dB 일정 음이라 전사에서 뺐다.
- **사람 전사 코퍼스(KEC·ZTC)는 ASR 불일치 0.5 이상만 전사 감독에서 뺀다.**
  - 남은 불일치는 대부분 구어·스위스 억양에서 생긴 ASR 오류이고, 사람 전사는 맞았다.
  - 한국어 AI Hub(군중 전사)는 0.2 에서 뺐다.
  - 한 화자가 통째로 어긋난 채널(KEC `rec_026_p4` IS_051)은 화자 단위로 뺀다(`p2_refine --asr-spk-thr 0.8`).
- **NSC 는 Qwen3-ASR-1.7B 와 Whisper large-v3 가 일치한 67.6 % 만 전사 감독에 쓴다.**
  - 일치 기준: WER ≤ 0.2, 간투사 무시, Whisper 숫자는 수사로 편다.
  - 불일치 구간은 시각·화자 정보만 남긴다.
  - 번들 Whisper 가 large-v3 에서 영어 번역을 내는 결함(언어 수 99 가정)을 찾아 고쳤다.
- **로더 경로 표본 점검을 통과했다.**
  - 정렬 토큰이 발화 구간 ±0.3 s 안에 드는 비율: KEC 99.7 %, ZTC 98.7 %, NSC 96.8 %
  - 40 창 조립 오류 0
  - 청취 점검은 하지 않았다.

## P3 직렬화 결과 (2026-10-14)
- 원자료: [[2026-10-14-stage3-p3-serialization]]
- 코드:
  - `vapasr/data/cst_dialogue.py`(데이터셋·collate)
  - 모델 `forward(sem_only_mask=)`·`add_semantic_tokens(turn=True, reinit_turn=True)`
  - trainer·packing 이 soft label 키를 넘긴다.
- **위 '형식' 절에 더한 구현 규칙:**
  - EOT 결과(p_end)는 창이 아니라 대화 전체 시간축으로 정한다. Phase 2 는 창 안 episode 만 봐서 창 끝 근처 교대를 놓쳤다.
  - 다른 화자가 먼저 말해 EOT 앞에 화자 태그를 새로 열어야 하면 soft 결정은 그 태그 위치에 둔다(soft 결정의 0.1–0.5 %).
  - 창 끝을 넘는 episode 와 창 안 토큰이 없는 episode 는 EOT 를 내지 않는다.
  - flush(창 끝 뒤 방출)에도 화자 태그를 넣는다. 기존 래퍼는 flush 에 태그를 넣지 않았다.
  - `<SEM_END>` 는 대화 행에서 중립(`sem_only_mask`)이다. replay(단일 화자) 행은 기존 `sem_mask` 로 SEM·EOT 둘 다 중립이다.
  - Stage 2 체크포인트의 `<EOT>` 행은 예약만 된 행이라 켤 때 다시 초기화한다.
- **전 코퍼스 dry-run:**
  - 2 인 대화 7 코퍼스(split v2 held-out 제외) 113,389 창에서 불일치 0 이다.
  - 시퀀스 길이 p50 830–898, 최대 1,547 토큰이다.
  - 창당 EOT 5.2–12.3 개, 한 화자만 말하는 창 0.9–12.3 % 다.
- **GPU 스모크(s4 final):**
  - Stage 3 배치 forward·backward 가 유한하다. `<SEM_END>` 행 grad 는 0 이다.
  - 패딩과 packing 의 eval 손실 차이는 0.0014 이하다.
  - 처음 loss_turn 은 6–7 이다(`<EOT>` 미학습).
- **맞장구 영향:**
  - 지금 규칙(Phase 2)에서는 상대 맞장구 때문에 결과의 24–50 % 가 both(0.5)다.
  - 1 s·3 토큰 이하를 맞장구로 빼면 11–32 % 로 줄고, 대부분 hold 로 옮겨 간다.
  - `backchannel=(1.0, 3)` 선택으로 비교할 수 있게 했다(기본 끔).

## P4·P5 준비 (2026-10-15)
- 원자료: [[2026-10-15-stage3-train-smoke]]
- **사용자 결정(2026-10-15):**
  - Stage 3 학습은 별도 스크립트로 한다.
  - 섞는 비율과 EOT 규칙은 파일럿에서 비교해 정한다.
- **학습:** `experiments/stage3_train.py`
  - 셋:
    - 대화: `dlg_ko`·`dlg_en`·`dlg_de`
    - Stage 2 replay: `rep_ko`·`rep_en`·`rep_de`. `<SEM_END>`·`<EOT>` 중립
  - step 마다 셋을 가중치로 고른다(`--weights`, `vapasr/hf/stage3.WeightedRoundRobinLoader`).
  - 레시피는 q06-s4-ast 와 같다.
  - 제출: `slurm/submit-stage3-train.sh`(산출 `/soundai/Model/VAPASR/stage3-<run>`)
- **스모크(단일 GPU 20 step):**
  - 대화 `<EOT>` 손실 4.97 → 1.2–1.7
  - final 재로드 parity OK
  - 실데이터 dry-run 에서 대화 창 KO 90,799 · EN 11,414 · DE 11,195, 6 셋 불변식 0
- **평가:** `experiments/cst_stage3_eval.py`
  - held-out 창을 두 슬롯 스트리밍 디코드로 잰다.
  - 지표: slot·cp(화자 귀속)·agn(화자 무시) 오류율, 슬롯 뒤집힘, EOT 재현율·정밀도·보류 오방출·지연
  - 참조와 가설 모두 전사 없는 음성 청크를 뺀다.
  - 오라클 점검에서 slot·cp 0, EOT 재현율 1 이다.
- **파일럿 비교안(각 1 노드, 2,000 step):**
  - 추정: step 당 대화 창 약 110 개(GPU 당 16,384 토큰 ÷ 창 약 900 토큰 × 8 GPU). 대화 step 은 약 1,330 회다.
  - 언어 1:1:1 이면 언어마다 약 5 만 창이다. KO 는 약 0.55 epoch, EN·DE 는 약 4.4 epoch 반복이다(그래서 p4 를 비교한다).

  | run | 바꾸는 것 | 보는 것 |
  |---|---|---|
  | p1 기준 | 대화 KO·EN·DE 1:1:1, replay 각 0.5(대화 2/3·replay 1/3), soft EOT, 맞장구 규칙 끔, 무음 격자 창 | 모든 비교의 기준 |
  | p2 | `--eot-mode hard` | EOT 정밀도·재현율·보류 오방출 |
  | p3 | `--backchannel 1.0,3` | EOT 보류 오방출, 맞장구 전사 |
  | p4 | 언어 비율 KO 2 · EN 0.7 · DE 0.7(창 수의 제곱근 비례) | 언어별 cp·agn |
  | p5 | replay 각 0.25(replay 약 1/5) | Stage 2 퇴행 vs 대화 성능 |

  - p1 을 먼저 내 전체 규모의 로딩 시간과 s/step 을 확인한 뒤 나머지를 낸다.
  - 평가는 `cst_stage3_eval.py` 로 held-out 을 재고, Stage 2 평가 묶음으로 단일 화자 퇴행을 잰다.

## 열린 질문
- `<EOT>` 를 soft 로 할지, 처음에는 hard(p_end ≥ 0.5)로 단순하게 시작할지(P5 에서 비교)
- 맞장구 처리: CST 첫 논문 범위에서는 사건으로 다루지 않는다. 전사로는 그대로 내고 `<EOT>` 대상에서는 뺄지 정해야 한다.
  - P3 측정: 맞장구를 빼면 both(0.5) 결과가 24–50 % → 11–32 % 로 준다. `CstDialogueDataset(backchannel=)` 로 P5 에서 비교한다.
- (해결 2026-10-15) Stage 3 학습 스크립트는 별도로 만들었다(`experiments/stage3_train.py`).
- 창 길이(20–40 s)를 그대로 쓸지, Stage 2 의 30 s 상한과 맞출지
- 134-2 의 높은 마스크 비율(평균 32 %)을 어떻게 다룰지(결손 구간 학습 제외 규칙은 Phase 2 그대로)
- 독일어 간투사: KEC·ZTC 는 사람 전사의 "äh"·"ähm" 을 남기고, NSC(교사 전사)와 Stage 1 독일어는 거의 내지 않는다. 같은 언어 안에서 간투사 표기가 섞인다.
- NSC 상대역 편중(amsterdam 254 세션): 샘플링 가중치를 낮출지 P5 에서 정한다.
