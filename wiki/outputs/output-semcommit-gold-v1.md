---
title: SEM_END 골드셋 v1 — 전수 주석 599 스트림(영어 낭독·영어 자발·한국어 짧은/긴 발화)과 v0.2 교사·모델의 골드 대조
summary: "모든 단어 경계를 COMMIT/AMBIG/NO 로 판정한 평가 골드셋(Claude 두 관점 독립 주석 + 판정, κ 0.84–0.97). v0.2 교사 A 라벨은 정밀도 0.78–0.92 지만 실제 확정 자리의 14–31 % 만 잡고 Stage A 후보가 21–32 % 를 놓치며 N 은 42–68 % 만 맞다. 모델은 한국어에서 교사를 넘는다(P 0.95 / R 0.50) — 라벨 기준 정밀도 0.46 은 과소평가였다. 대화체 영어에선 거의 확정하지 못한다."
type: output
created: 2026-09-24
updated: 2026-09-24
sources: [raw/sources/experiments/2026-09-24-semcommit-gold-v1/, raw/inbox/streaming_asr_semantic_commit_plan.md]
related: [output-semcommit-v0.2-rack4-small, decision-semcommit-turn-eot-scope, output-semcommit-recipe-v0.3]
---

# SEM_END 골드셋 v1

**라벨링 레시피 v0.3(2026-09-24 추가)**: 아래 §4 의 교사 관문 목표(A 정밀도 ≥ 0.9 를 지키며 재현율 상향)에 맞춰 레시피를 고쳤다 — 같은 8B 교사로 A 재현율 0.70–0.82, 정밀도 0.89–0.99(v0.3.2), 교사 관문 스크립트도 v0.3 기본으로 바뀌었다: [[output-semcommit-recipe-v0.3]].

## 질문
v0.2 실험([[output-semcommit-v0.2-rack4-small]])의 평가 정답은 교사 라벨(A)이라, 교사가 틀리면 모델 점수도 같이 틀린다. 교사와 무관한 기준으로
(1) 지금 교사가 실제 확정 자리를 얼마나 잡는지, (2) 모델이 실제로 얼마나 맞게 확정하는지를 재고, mxc 본 실행 전 교사 관문의 기준을 만든다.

## 요약
- **골드셋 v1**: 599 스트림, 단어 경계 14,545 개 전수 판정 → COMMIT 1,099 · AMBIG 313. 영어 낭독(LibriSpeech test-clean 100) · 영어 자발(GigaSpeech test podcast/YouTube 79) · 한국어 짧은 발화(Kspon eval_clean 300) · 한국어 긴 발화(Kspon eval 18 어절+ 120).
  Claude 두 관점(규칙 적용 / 뒷단 에이전트) 독립 주석 → 불일치 197 경계만 판정. 주석자 일치 κ 0.84–0.97, COMMIT F1 0.95–0.99.
- **v0.2 교사 라벨은 실제 확정 자리의 14–31 % 만 잡는다**(A 재현율). A 정밀도는 0.78–0.92 로 괜찮다. Stage A 후보가 실제 확정 자리의 21–32 % 를 아예 놓치고, 가린 B 의 31–68 % 가 실제 확정 자리, N 은 42–68 % 만 맞다.
- **모델은 한국어에서 교사보다 낫다**: r1(δ4) bias 0 이 짧은 발화 P 0.95 / R 0.50(교사 A 자체 0.92 / 0.31), bias +2 면 0.84 / 0.67. 긴 발화 0.82 / 0.48 → +2 에서 0.68 / 0.74.
  v0.2 보고서의 한국어 정밀도 0.46(교사 A 기준)은 모델의 맞는 확정을 틀렸다고 센 과소평가였다.
- **영어는 약하다**: 낭독 P 0.81 / R 0.09(bias +2 에서 0.70 / 0.24), **대화체(GigaSpeech)는 79 스트림에서 확정 1 개**(bias +2 도 14 개, R 0.06) — 낭독체만으로 배운 확정이 자발 발화로 옮겨 가지 않는다.
- 음성인식은 새 셋에서도 유지: GigaSpeech WER 14.92 → 14.17 %(r1), 긴 한국어 CER 10.96 → 10.82 %.

## 1. 구성과 절차
| 파트 | 원천 | 스트림 | 경계 | COMMIT | AMBIG | κ(3 분류) | 두 주석자 COMMIT F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ls-test | LibriSpeech test-clean(~30 s 연속 문장) | 100 | 7,044 | 503 | 140 | 0.844 / 0.933 | 0.951 / 0.990 |
| gs-test | GigaSpeech test podcast·YouTube 한 화자 구간(8–25 s) | 79 | 2,674 | 192 | 54 | 0.938 | 0.982 |
| ks-eval | KsponSpeech eval_clean 발화 | 300 | 2,153 | 247 | 65 | 0.967 | 0.973 |
| ks-long | KsponSpeech eval 18 어절+ 발화(ks-eval 과 안 겹침) | 120 | 2,674 | 157 | 54 | 0.947 | 0.977 |

- 지침 `GUIDELINE.md`: 계획서 §3·§5 를 주석용으로(COMMIT = 완결 ∧ 안정, 연결어미·같은 절 이어짐·담화 표지·스스로 고침·반복·인용 내부는 NO, 확신 없으면 AMBIG).
- 주석자는 등급·교사 라벨을 보지 않았다. 참조 전사(영어 구두점 전사, 한국어 Kspon 원 전사)는 구조 파악용. 오디오는 쓰지 않았다(억양·쉼 미반영).
- gs-test 는 이번에 새로 만든 대화체 영어 몫(`experiments/semcommit_build_gigaspeech.py`: 선택 → wav 추출 → Qwen3-ForcedAligner → words.jsonl, 구두점 전사 = pnc_text).
- 도구: `experiments/semcommit_gold.py`(packet · merge · score-labels · score-eval), 지표 `commit_metrics.gold_*`.
  P = 골드 COMMIT 에 맞은 확정 ÷ (확정 − 골드 AMBIG 자리 확정); 단어 중간·첫 단어 전·중복 확정은 오류.

## 2. v0.2 교사 라벨 대 골드
| 파트 | 후보 재현(Stage A) | A 정밀도 | A 재현율 | N 정밀도 | B 중 실제 확정 |
|---|---:|---:|---:|---:|---:|
| ls-test | 0.68 | 0.78 | 0.14 | 0.48 | 0.56 |
| gs-test | 0.77 | 0.89 | 0.16 | 0.42 | 0.68 |
| ks-eval | 0.79 | 0.92 | 0.31 | 0.60 | 0.45 |
| ks-long | 0.74 | 0.86 | 0.27 | 0.68 | 0.31 |

v0.2 교사: Qwen3-8B(A·B·C) + EXAONE-3.5-7.8B(B), gs-test·ks-long 은 같은 레시피로 새로 라벨링(`labels/v0.2-gold`). 교사 일치도는 새 셋에서도 κ 0.117(gs) · 0.183(ks-long).

## 3. 모델 대 골드 (δ4)
| 파트 | r1 bias 0 P / R / F1 | r1 bias +2 | r2 bias 0 | r2 bias +2 | 교사 A 자체(오라클) |
|---|---|---|---|---|---|
| ls-test | 0.81 / 0.09 / 0.16 | 0.70 / 0.24 / 0.35 | 0.82 / 0.11 / 0.19 | 0.72 / 0.29 / 0.41 | 0.78 / 0.14 / 0.24 |
| gs-test | 1.00 / 0.01(확정 1) | 0.79 / 0.06 / 0.11 | 1.00 / 0.01 | 0.83 / 0.10 / 0.18 | – |
| ks-eval | 0.95 / 0.50 / 0.65 | 0.84 / 0.67 / 0.75 | 0.93 / 0.52 / 0.67 | 0.78 / 0.68 / 0.73 | 0.92 / 0.31 / 0.47 |
| ks-long | 0.82 / 0.48 / 0.60 | 0.68 / 0.74 / 0.71 | 0.80 / 0.51 / 0.62 | 0.62 / 0.75 / 0.68 | – |

- 지연(맞힌 확정, 단어 끝 기준) p50 0.32–0.36 s(δ4), δ2 는 0.18 s(ks-eval r1 δ2 bias 0: P 0.95 / R 0.42).
- r2(N 가중 끔)는 재현율이 조금 높고 정밀도가 조금 낮다 — 차이는 작다.

## 4. 해석
1. **교사 관문의 목표가 숫자로 정해졌다**: 지금 교사는 정밀도(0.78–0.92)는 되지만 재현율(0.14–0.31)이 문제이고, 그 원인은 Stage A 후보 누락(21–32 %)과 B 로 가려지는 실제 확정 자리(31–68 %)다.
   큰 교사·새 레시피는 “A 정밀도 ≥ 0.9 를 지키면서 A 재현율을 몇 배로” 가 기준이다(`experiments/semcommit_teacher_gate.sh` 로 네 파트 한 번에).
2. **N(금지) 라벨은 교사 품질로는 쓸 수 없다**(정밀도 0.42–0.68). 사람 전사 표지·고확신 스스로 고침만 남기거나 끈다.
3. **모델은 라벨보다 일반화한다**(한국어 R 0.50 > 교사 0.31, 같은 정밀도대). 라벨 재현율만 올라가도 모델 재현율이 크게 오를 여지가 있다.
4. **대화체 영어는 데이터가 없으면 못 배운다** — 본 학습에 대화체 포함([[decision-semcommit-turn-eot-scope]])이 필수라는 근거.
5. 모델 평가는 이제 골드 기준으로 한다(교사 A 기준 지표는 교사 오류를 모델 오류로 센다).

## 불확실성
- 골드는 Claude 가 만들었다(사람 검수 없음). 두 관점 일치는 같은 모델 계열의 일치라 신뢰도의 상한이다. 대화체 영어는 GigaSpeech 대리(본 학습 Switchboard 와 다름), 한국어는 Kspon 뿐.
- ls-test·ks-eval 은 v0.2 교사 라벨로 이미 평가에 쓰던 셋이다(학습에는 안 씀).
- 모델 r1/r2 는 7 시간·5 분 학습의 소규모 모델이라 절대 수치보다 교사 대비 방향을 본다.

## 근거
- 원자료: `raw/sources/experiments/2026-09-24-semcommit-gold-v1/`(README·GUIDELINE·gold-*.jsonl·ann·adj·merge 통계·scores·SHA256SUMS).
- 전체 묶음(words·wav 포함): rack4 `/data4/tskim/semcommit/gold/v1`, 로컬 T5 `/Volumes/Samsung_T5/VAPKT-DB/semcommit-gold-v1`(mxc 업로드용).
- 코드: `experiments/semcommit_gold.py`, `experiments/semcommit_build_gigaspeech.py`, `experiments/semcommit_teacher_gate.sh`, `vapasr/hf/commit_metrics.py`(gold_*).
