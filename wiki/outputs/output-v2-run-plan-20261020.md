---
type: output
status: active
created: 2026-10-20
updated: 2026-10-20
summary: V2 첫 학습(v2-p1) 계획·결과 — 받아쓰기는 크게 올랐고(회의·AMI −6–10 pt) 화자 귀속은 그대로, TAXI 는 5 pt 나쁨(탭 탓 아님)
sources:
  - '[[2026-10-20-v2-multispeaker]]'
  - '[[decision-v2-run-scope-20261020]]'
  - '[[output-v2-speaker-memory-plan-20261020]]'
  - '[[output-stage4-prep-plan-20261019]]'
  - '[[2026-10-20-v2a-dual-stream]]'
raw_authors:
  - tskim
---

# V2 첫 학습 준비 (2026-10-20)

결정은 [[decision-v2-run-scope-20261020]] 에 있다. 이 페이지는 작업과 설계 선택을 적는다.

## 데이터 현황
| 출처 | 언어 | 시간 | 화자 수 | 감독 비율(시간) | held-out(split v3) |
|---|---|---:|---|---:|---:|
| AMI | 영 | 100 h | 회의마다 3–5(대부분 4) | 0.97 | 20 회의 |
| ICSI | 영 | 72 h | 3–7(대부분 5–7) | 0.96 | 6 |
| NOTSOFAR | 영 | 25 h | 3–7 | 0.99 | 36 |
| CallHome 독일어 | 독 | 18 h | 2(104)·3(12)·4(4) | 0.15 | 12 |
| 기존 2 인 대화(AI Hub·otoSpeech·KEC·ZTC·NSC) | 한·영·독 | – | 2 | 0.57–0.98 | 기존 |

- AMI 창별 화자 수(2026-10-19): 40 s 창의 33 % 가 3 명, 42–49 % 가 4 명이다.
- 한국어·독일어 3–4 인 실제 대화는 사실상 없다. 그래서 합성으로 만든다.

## 설계 선택
- **슬롯:** A·B·C·D 는 창 안에서 먼저 말한 순서다(지금 규칙의 확장).
  - C·D 토큰은 `<SPK_3>`·`<SPK_4>`(예약 행을 평균 + 잡음으로 다시 초기화하고 차단을 푼다).
  - 화자 활동 헤드는 2 → 4 출력이다. 기존 A·B 행은 그대로 두고 C·D 행만 새로 만든다.
- **prefix 의 슬롯 선언:** 학습 창의 절반은 창 화자 수만큼(최소 2), 나머지 절반은 4 슬롯을 모두 선언한다.
  - 추론 때 화자 수를 알면(전화 2 명) 그만큼, 모르면(회의) 4 를 선언하면 된다. 선언한 슬롯이 비어 있어도 되는 것을 배운다.
- **교차 언어 합성:** 2 인 그대로 둔다(슬롯 언어 고정).
- **창(A + B):**
  - epoch 마다 다른 "draw" 를 쓴다. draw 하나는 대화를 길이 U(20, 90) s 창으로 이어 덮는다. 시작은 무작위 오프셋에서 무음 지점으로 맞춘다.
  - 데이터셋은 draw R 개를 미리 만들어 두고, sampler 가 epoch e 에 draw e mod R 을 쓴다. 워커 복사본과 길이 추정은 한 번만 만든다.
  - 지금은 hop 10 s 격자라 한 epoch 에 같은 말을 약 3 번 본다. 이어 덮기는 한 번이다. 그래서 epoch 당 step 이 준다.
  - 창 화자가 4 명을 넘으면 4 명 이하가 되는 가장 긴 길이(≥ 20 s)로 줄인다. 그래도 넘으면 버린다(ICSI·NOTSOFAR 의 5–7 인 회의).
- **정체 회복 학습(recover):** 2 슬롯 창에서만 쓴다(지금 꺼져 있다).
- **평가:**
  - V1 비교용으로 기존 셋(20–40 s 고정 창, Stage 3 회귀·대화 번역·영↔독 합성·TAXI)을 그대로 쓴다.
  - 새로 더하는 셋: 긴 창(60–90 s) held-out 대화, 회의 held-out(≤ 4 화자), 3–4 인 합성 held-out(한·독), TAXI 세션 통째(`--max-window 90`)
  - cpWER 은 N 슬롯 최적 짝이다.

## 작업과 상태
| # | 작업 | 상태 |
|---|---|---|
| V1 | 모델: 슬롯 C·D 토큰 켜기(`enable_slots`), 화자 활동 헤드 2 → 4(A·B 행 유지) | **완료**(단위 시험) |
| V2 | 시퀀스: `cst_dialogue` N 슬롯(전사·번역 흐름 2N, 태그, 다시 읽기, 점검, 활동 라벨, 슬롯 선언 `declare_all_p`·`declare`) | **완료**(단위 시험, 2 화자는 예전과 같음) |
| V3 | 창: draw 이어 덮기(`window_draws`)·20–90 s·화자 상한 줄이기(`shrink_speakers`), sampler 가 epoch e 에 draw e mod R | **완료**(단위 시험) |
| V4 | 평가: N 슬롯 채점(cp 순열)·오류 분해·prefix 별 배치, 창 인자(`--max-speakers`·`--window`·`--window-draws`·`--declare`), AMI 공식 dev·eval(IHM-mix·SDM) | **완료**(오라클 통과) |
| V5 | 합성: `synth_session` N 화자(`p_back` 0.2, 2 인은 같은 난수열), `--speakers 2:0.2,3:0.4,4:0.4`, 3–4 인 합성 셋(학습 언어마다 1 만·평가 300) | **완료** |
| V6 | 회의 번역 타깃: 원문 단위 `stage4-dlg-units-meet-v1`(AMI·ICSI·NOTSOFAR 262,380 단위·2.04 M 단어·227,935 행) → 생성(en→de·ko)·수리는 apex 제출(사용자) | **완료**(93807·93808) |
| V7 | 학습 스크립트: `--max-speakers`·`--declare-all-p`·`--window`·`--window-draws`·`--dlg-corpora`(회의 = `dlg_mt`)·지문, V2-a 탭, 전화 대역 증강 | **완료**(코드) |
| V8 | dry-run → 스모크 → 학습 전 점검 → 제출 명령 | **완료** — v2-p1(apex 93883) 3,000 step·PARITY OK, 결과는 아래 "v2-p1 결과(중간)" |

## 평가 분할 정리 — split v4 (2026-10-20)
- 사용자 질문: "AMI-IHM 과 AMI-SDM 으로도 평가 진행하려는 데, 학습 셋과 겹치지는 않지?"
- 확인 결과(V1 은 회의를 학습하지 않아 해당 없음, V2 부터 회의를 학습):
  - AMI 공식 eval 16 회의(9.06 h)는 v2 부터 모두 held-out 이다.
  - AMI 공식 dev 18 회의(full-corpus ASR 분할, 9.67 h)는 학습 쪽이었다.
  - **NOTSOFAR 원본 eval_set 129 회의가 학습 쪽이었다**(held-out 36 = 원본 dev_set, 나머지 72 = train_set).
- **V2 는 split v4 로 학습한다**(`make_split_v4.py`, `phase2/splits/v4.json` 새 파일):
  - AMI: 38 회의 held-out(21.1 h) — 공식 eval·dev + ES2014. 학습은 133 회의(78.8 h).
  - NOTSOFAR: 165 회의 held-out(17.1 h) — 원본 dev·eval. 학습은 train_set 72 회의(7.4 h).
  - 다른 코퍼스는 v2 와 같다(V1 평가 창 그대로).
- mxc 의 AMI 는 헤드셋 오디오(Headset-0..4·Mix-Headset)만 있다. AMI-SDM(원거리 마이크 Array1)은 따로 받아야 한다.

## 학습 설정 (사용자 승인 2026-10-20 "그래 그 비율로 진행해.")
- init: Stage 3 최종 p11 ck3000(st4 run 들과 같다). 설정은 st4-m2a(전화 대역 증강 0.3)와 같고 V2 변경만 더한다. split v4.
  - V2 변경: `--enc-tap-layer 0`, `--max-speakers 4 --declare-all-p 0.5`, `--window 20,90 --window-draws 6`, 회의 코퍼스 + 회의 번역 타깃, 합성 v3(3–4 인)
- 섞는 비율: `dlg_ko:2,dlg_en:0.7,dlg_de:0.7,dlg_mt:1.0,sim_ko:0.8,sim_en:0.8,sim_de:0.8,sim_en-de:1.0,sim_en-ko:1.0,rep_ko:1.2,rep_en:1.2,rep_de:1.2`
  - step 비중: 실제 대화 35.5 %(회의 8.1 %), 3–4 인 합성 19.4 %, 교차 언어 합성 16.1 %, 단일 화자 29.0 %
  - m2a 대비: 회의 추가, sim_ko 0.5 → 0.8, 같은 언어 합성을 v3(3–4 인)으로 교체. CallHome 독일어(`rep_chde`)는 m2b 에서 TAXI 가 나빠져 넣지 않는다.
- step: 3,000(m2a 2,500 — 새로 배울 것이 많고 창이 길어 step 당 창 수가 준다). lr 4e-5 cosine, warmup 300, 500 step 마다 저장, 2 노드.
  - m1 은 step 이 늘수록 번역 반복 생성이 다시 늘었다. 그래서 ck1500·2000·2500·final 을 모두 재 고른다.
- 예상: m2a 1.0 s/step → V2 1.2–1.5 s/step(긴 창), 약 1–1.3 h

## V2 학습 전 점검 결과 (2026-10-20, `v2_check.sh` — mxc 개발 사본에서, 본 checkout 반영 뒤 단위 시험 95 통과)
- 입력: 3–4 인 합성 v3(한 1 만·영 9,998·독 9,994 세션, 각 250–263 h), 회의 번역 타깃(영→독 98.7 %·영→한 96.5 % 행 유지), split v4
- dry-run(CPU): 12 셋 × 32 항목 × δ 7 개 불일치 0
  - 창(6 draw 합): dlg_ko 285,136 · dlg_en 38,108 · dlg_de 37,429 · dlg_mt 55,034 · sim_ko·en·de 99–102 k · 교차 언어 97–99 k
  - 회의 창 중 4 명이 넘는 창은 4,115 개를 줄이고 3,335 개를 버렸다(5–7 인 ICSI·NOTSOFAR). 회의 번역 타깃은 학습 쪽 행의 99.7 % 에 붙었다.
  - 지문: max_speakers 4 · declare_all_p 0.5 · window_draws 6 · window [20, 90] · enc_tap L00 · 전화 대역 0.3
- 스모크 20 step(GPU 하나, Stage 3 최종에서): 손실 정상, 끝 재로드 PARITY OK
  - slot_tokens A·B·`<SPK_3>`·`<SPK_4>`, 차단 해제, 화자 헤드 4 출력, `adapter_tap` 저장
- 3,000 step 동안 셋별 회전 수(추정): dlg_ko 1.6 · dlg_en/de/mt 약 4 · 합성 약 1.8–2.3 — 바퀴마다 다른 draw

## V2-a 학습 전 점검 결과 (2026-10-20) → [[2026-10-20-v2a-dual-stream]]
- 단위 시험 mxc 54 통과, dry-run 11 셋 불일치 0, 스모크 20 step PARITY OK(`adapter_tap` 저장·재로드, 0 초기화 마지막 층이 20 step 만에 |w| 0.0023 으로 학습)
- 평가 경로: Stage 3 평가 직렬 = 배치(한·영), 영↔독 합성 평가 실행
- 실제 모델: 탭 = 층 출력(차이 0), 0 초기화 LLM 입력 = V1(차이 0)·디코드 12 창 모두 같음

## v2-p1 결과(중간, 2026-10-20 23:40) → [[2026-10-20-v2-multispeaker]]
- 학습: 첫 step 의 `slot_activity` IndexError(화자 상한 줄이기 반올림, 고침 b844d9a) 뒤 restart 2/5 부터 정상. 3,000 step(55 분), PARITY OK.
- V1(st4-m2a ck2000) 대비, δ10:
  - 2 인 같은 언어 대화(V2 ck2000): cp 25.59 → 24.46, agn 18.18 → 17.03. 대화 짝 bootstrap 구간이 0 아래다(cp −1.33 [−2.43, −0.28]).
    - `<EOT>` 재현율 0.72 → 0.79, 대화 번역 네 방향 +0.9–1.5 BLEU
  - 2 인 창의 화자 귀속 몫은 7.4 그대로다(−0.11 [−1.07, +0.86]).
  - 단일 화자 ±0.5 pt, CallHome 독일어 21.6 → 19.7
  - **TAXI(실제 독↔영 전화) cpWER 28.1 → 33.1–36.6**(체크포인트 넷 모두 나쁨)
    - 대부분 두 화자 말이 한 언어 슬롯으로 몰린 짧은 세션 탓이다.
    - C·D 슬롯 새기는 0 이다. 탭을 끄면 37.3 으로 더 나빠진다.
    - 남은 후보는 데이터·창 변경(합성 2 인 → 3–4 인 위주, 회의, 긴 창, 4 슬롯 선언)이다.
  - 영↔독 합성 δ10 영어 반복 생성이 ck2500·final 에서 늘었다.
- 진행 중: 긴 창·회의·3–4 인 합성·AMI(IHM-mix·SDM)·회의 번역. V2 의 목적(다화자 귀속)은 이 결과로 판단한다.

## v2-p1 최종·탭 뺀 대조 (2026-10-21) → [[2026-10-20-v2-multispeaker]]
- 받아쓰기: 회의 agn −6–9 pt, AMI 공식 eval cp IHM-mix 54.0 → 47.9 · SDM 60.9 → 53.2, 회의 번역 ICSI +2–5 BLEU.
- 화자 귀속: 회의 19–40 pt, 3–4 인 합성 44–47 pt 로 V1 과 같거나 크다. 긴 창만 16.1 → 14.9.
- 탭 뺀 대조(v2-p1-notap, 같은 데이터·step): 탭은 받아쓰기 이득(agn −2.0)이고 귀속은 그대로다. TAXI 하락은 탭이 아니다(notap 32.1).
- 결론: V2 의 목적(다화자 귀속)은 이루지 못했다. TAXI 귀속은 V3(턴 시작 교대 결정)에서, 다화자 귀속은 화자 목적함수가 있는 표현·기억으로 다룬다 → [[output-v3-attribution-plan-20261021]]
