---
title: Semantic commit v0.2 — rack4 소규모 구현·라벨링·학습·평가 (작은 오픈 LLM 교사)
summary: "<SEM_END>/<TURN_END> 파이프라인을 LibriSpeech 3.3 h + KsponSpeech 3.9 h 로 끝까지 실행. ASR 가드레일은 통과(EN WER 5.27→4.68 %, KO CER 12.72→12.80 %). SEM F1 은 KO 0.56, EN 0.17–0.24. 병목은 모델보다 라벨: 작은 교사 간 κ 0.12–0.21 이라 EN 문장 끝의 18 % 만 A 로 남고, N 은 절반이 오판, TURN 은 스트림 길이 지름길을 학습."
type: output
created: 2026-09-24
updated: 2026-09-24
sources: [raw/sources/experiments/2026-09-24-semcommit-v0.2-rack4/, raw/inbox/streaming_asr_semantic_commit_plan.md]
related: [output-phase1-report]
---

# Semantic commit v0.2 — rack4 소규모 실행

## 질문
계획서(`raw/inbox/streaming_asr_semantic_commit_plan.md`)의 semantic commit 시퀀스(`<SEM_END>`·`<TURN_END>` 를 `<NEXT_AUDIO>` 와 경쟁하는 순수 토큰으로)와
3 단계 LLM 재라벨링(A 후보·B 인과 판정·C 미래 안정성)을, 작은 데이터와 작은 오픈 모델로 rack4 GPU 1 장에서 끝까지 돌리면 무엇이 되고 무엇이 막히는가.
본 학습·대규모 데이터 준비는 mxc 에서 한다(이 실행은 컴파일·검증용).

## 요약
- **파이프라인은 끝까지 돈다**: 단어 색인 → Stage A/B/C → 등급(A/B/N) → E2 에서 학습(5.3 분, save/load parity OK) → 스트리밍 디코드 평가(oracle 상한 1.0, `--verify` 로 배치 디코드 = `stream_decode` 토큰 일치).
- **ASR 가드레일 통과**: E2+토큰(학습 없음) 대비 EN WER 5.27 → 4.68 %(r1)·4.90 %(r2), KO CER 12.72 → 12.80 %(r1)·12.72 %(r2) (δ4).
- **SEM 성능**: KO F1 0.56(P 0.46, B 제외 0.66, R 0.71, 지연 p50 0.35 s), EN F1 0.17(r1)·0.23(r2). KO 는 참조 A 의 90 % 가 발화 끝이라 사실상 '발화 끝 commit' 성능이다(중간 A 재현 2/9).
- **병목은 라벨**: 작은 교사 둘의 B 판정 일치도 κ = 0.12–0.21. EN 에서 LibriSpeech-PC 문장 끝 353 개 중 **A 는 63(18 %)**, B(마스크) 201 개. 모델은 라벨 분포를 그대로 재현한다(문장 끝 비율 P_pc: 모델 0.62–0.67 ≈ A 라벨 자체 0.68).
- **추가로 드러난 문제**: (1) N(hard negative) 은 블라인드 점검에서 절반이 좋은 경계, (2) Stage A 가 짧은 KO 발화의 끝(명백히 완결)을 후보로 안 냄, (3) `<TURN_END>` 가 EN 학습 스트림 길이(24–34 s)를 외워 22–26 s 에 조기 발사, (4) gpt-oss-20b 는 final 채널 로그확률 판정에서 97 % WAIT — EN 판정자로 못 씀.

## 1. 설정
- **데이터**(`experiments/semcommit_build_words.py`, forced-align manifest 로 단어 시각):

  | 셋 | 원천 | 스트림 | 단어 | 길이 |
  |---|---|---:|---:|---|
  | ls-train | LibriSpeech train-clean-100 일부 | 400 (라벨 398) | 29,983 | 3.3 h, 스트림 24–34 s(p10–p90) |
  | ls-test | test-clean | 100 | 7,044 | 48 분 |
  | ks-train | KsponSpeech_01 | 2,000 (라벨 1,977) | 20,996 | 3.9 h, 2.9–13 s |
  | ks-eval | KsponSpeech eval_clean | 300 (라벨 299) | 2,153 | – |

  EN 은 LibriSpeech-PC 구두점 전사(`pnc_text`)를 힌트·독립 참조로, KO 는 Kspon 전사 표지(`/` 간투사, `+` 반복, `*` 불명확)를 단어 태그로 둔다.
- **교사**(계획의 EXAONE-32B·gpt-oss-120b 대신 작은 오픈 모델, rack4 A100-40GB GPU 0 에서 순차):
  Qwen3-8B(bf16, thinking 끔) = Stage A(JSON 생성)·B·C, EXAONE-3.5-7.8B-Instruct(원격 코드 rev 0ff6b5e — 최신 코드는 transformers v5 전용) = Stage B(KO·EN),
  gpt-oss-20b(MXFP4→bf16, 앞 18 층 GPU) = KO tie-break(진단용, strict 등급이라 라벨 영향 없음). Stage B/C 는 라벨 토큰 로그확률로 결정적 채점.
  프롬프트 v0.2(`semcommit-prompt-v0.2+ab90c940`): Stage A 경계를 `[번호, 단어]` 쌍으로 받아 ±4 보정, Stage C REVISION 을 좁힘.
- **등급**: A = 두 판정자 SAFE ∧ C 가 REVISION 아님 → `<SEM_END>`, B = 불일치 등 → 결정 위치 마스크, N = C REVISION(SELF_REPAIR·RESTART)·전원 WAIT → 결정 위치 가중 1.0.
  C 의 QUALIFICATION·CONTINUATION 판정은 마스크(`--c-mask-types`) — Qwen3 가 QUALIFICATION 을 과다 판정(ls-test 236/587, 검토상 대부분 오판).
- **학습**(`experiments/semcommit_train.py`): E2 초기화, 인코더 동결(저장), lr thinker 3e-5·adapter 1e-4, 3 epoch 567 step(EN 배치 6 / KO 16, 언어 균형), δ ∈ {2,3,4,6}, 5.3 분·GPU 23 GB.
  r1 = 기본(N 결정 위치 가중 1.0), r2 = `--hardneg-weight 0`(N 위치도 일반 NEXT 가중 0.3/0.15 — 마스크가 아니라 상향 가중만 끔).
  둘 다 parity OK(955 키 값 차이 0, 인코더 638 키 저장, 재로드 손실 동일).
- **평가**(`experiments/semcommit_eval.py`): 자유 디코드, bias(logit[SEM]+b)·threshold(p ≥ θ) 스윕, 지표는 A 참조 기준 시간 창 P/R/F1(late 2 청크)·텍스트 위치 PCR·지연,
  B 위 commit 은 '애매'(오류로 안 셈), TURN P/R·조기 발사, WER/CER. 신규 `experiments/semcommit_pc_eval.py` 로 EN 은 PC 문장 끝(. ? !) 대비 P_pc·R_pc 도 낸다.

## 2. 라벨
| 셋 | 후보 | A | (발화 끝 A) | B | N | 판정자 κ |
|---|---:|---:|---:|---:|---:|---:|
| ls-train | 2,661 | 367 | 115 | 1,895 | 399 | 0.12 (qwen3~exaone35) |
| ls-test | 587 | 98 | 37 | 405 | 84 | 0.16 |
| ks-train | 3,255 | 712 | 548 | 2,175 | 368 | 0.20 (exaone35~qwen3) |
| ks-eval | 357 | 89 | 80 | 241 | 27 | 0.21 |

- **gpt-oss-20b 탈락**: ls-test EN Stage B 에서 WAIT 570 / SAFE 17(Qwen3 와 둘 다 SAFE 15 → A ≈ 3). harmony final 채널을 채워 라벨 로그확률만 보는 방식에서 극단적 WAIT 편향.
  같은 후보에 EXAONE-3.5 는 SAFE 316 / UNCERTAIN 244 / WAIT 27, Qwen3 와 둘 다 SAFE 226 → EN 두 번째 판정자를 EXAONE-3.5 로 바꿨다(이 실행 한정).
- **블라인드 점검**(등급 숨긴 100 항목, Claude 판정자 두 관점 — 계획 §3·§5 규칙 문자 그대로 적용하는 annotator / 커밋 받은 downstream LLM 관점, 두 관점 일치 94/100):

  | 셋·등급 | n | 두 관점 모두 COMMIT_OK | NOT_OK | 갈림·UNSURE |
  |---|---:|---:|---:|---:|
  | ls-test A | 30 | 23 (77 %) — 발화 끝 11/11, 중간 12/19 | 3 | 4 |
  | ks-eval A | 30 | 26 (87 %) — 발화 끝 22/25, 중간 4/5 | 2 | 2 |
  | ls-test N | 10 | **6** | 3 | 1 |
  | ks-eval N | 10 | **4** | 4 | 2 |
  | B (두 셋) | 20 | 8 | 8 | 4 |

  A 오판은 열린 종속절("when i say that i am an italian ‖ he begins"), 같은 문장 등위("all was quiet ‖ and black"), 인용 뒤 발화 표지("… ‖ said sir ferdinando"),
  열린 연결어미("탔는데 ‖ 그 다른 애들도", "이미지 때문에" 로 끝난 발화). N 오판은 Stage C 가 다음 문장을 SELF_REPAIR 로 본 경우("you are giving me a chance ‖ yes alexander")와
  전원 WAIT 인 완결 문장("그 근데 가는 데 시간이 너무 오래 걸려 ‖ 그 코엑스랑…").

## 3. 결과 (δ4; 괄호 없는 P/R 은 A 참조 시간 창 late 2)
### EN ls-test (100 스트림, A 98)
| 모델 | 설정 | commit | P | P(B 제외) | R | F1 | 지연 p50 | PCR | P_pc / R_pc | TURN P/R (조기) | WER |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E2+토큰(학습 없음) | – | 0 | – | – | 0 | – | – | – | – / 0 | – / 0 | 5.27 % |
| r1 | bias 0 | 56 | 0.23 | 0.41 | 0.13 | 0.17 | 0.33 s | 0.34 | 0.62 / 0.09 | 0.65 / 0.50 (27) | 4.68 % |
| r1 | bias +2 | 185 | 0.15 | 0.27 | 0.29 | 0.20 | 0.35 s | 0.42 | 0.52 / 0.25 | 0.57 / 0.55 (41) | 4.66 % |
| r2 | bias 0 | 67 | 0.28 | 0.46 | 0.19 | 0.23 | 0.35 s | 0.31 | 0.67 / 0.11 | 0.62 / 0.55 (32) | 4.90 % |
| r2 | bias +2 | 215 | 0.17 | 0.30 | 0.38 | 0.24 | 0.35 s | 0.41 | 0.54 / 0.30 | 0.53 / 0.59 (52) | 4.90 % |
| r1 δ2 | bias 0 | 51 | 0.24 | 0.43 | 0.12 | 0.16 | 0.17 s | 0.31 | 0.64 / 0.08 | 0.65 / 0.54 (29) | 7.77 % |
| 참조 직렬화(oracle) | – | 98 | 1 | 1 | 1 | 1 | 0.37 s | 0 | 0.68 / 0.18 | 1 / 1 | 0 |

### KO ks-eval (300 스트림, A 89 중 80 이 발화 끝)
| 모델 | 설정 | commit | P | P(B 제외) | R | F1 | 지연 p50 | PCR | TURN P/R (조기) | CER(공백 무시) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E2+토큰 | – | 0 | – | – | 0 | – | – | – | – / 0 | 12.72 % |
| r1 | bias 0 | 137 | 0.46 | 0.66 | 0.71 | 0.56 | 0.35 s | 0.21 | 0.74 / 0.85 (68) | 12.80 % |
| r1 | bias −2 | 89 | 0.54 | 0.74 | 0.54 | 0.54 | 0.35 s | 0.17 | 0.72 / 0.82 (67) | 12.80 % |
| r2 | bias 0 | 149 | 0.45 | 0.64 | 0.75 | 0.56 | 0.34 s | 0.23 | 0.71 / 0.83 (74) | 12.72 % |
| r2 | θ 0.5 | 142 | 0.46 | 0.65 | 0.74 | 0.57 | 0.35 s | 0.22 | 0.70 / 0.82 (74) | 12.72 % |
| r1 δ2 | bias 0 | 115 | 0.45 | 0.62 | 0.58 | 0.51 | 0.20 s | 0.24 | 0.77 / 0.85 (60) | 15.98 % |

- KO r1 bias 0 재현율: 발화 끝 A 64/80(0.80), 중간 A 2/9. EN r1 bias 0: 발화 끝 10/37, 중간 3/61.
- 조기 commit(PCR) 범주: EN 19 중 14, KO 29 중 25 가 '후보 아닌 위치(other_inside)'. KO 의 25 중 **11 은 Stage A 가 후보를 안 낸 발화 끝**("아 어떡하지", "진짜 오타쿠였구나")이라 사실상 라벨 누락.
- bias/θ 스윕이 EN 에서 정밀도를 거의 못 올린다(0.15–0.28): 모델 commit 의 대부분이 A 가 아닌 B(마스크된 문장 끝)나 쉼표 위에 있고, 라벨이 그 둘을 구분해 주지 않는다.

## 4. 해석
1. **라벨 재현율·일관성이 상한을 정한다.** κ 0.12–0.21 인 두 작은 판정자의 AND 는 EN 문장 끝의 57 % 를 B(학습에서 마스크)로 보낸다. 모델은 A 라벨 분포를 그대로 배워 P_pc 가 A 라벨 자체(0.68)에 붙는다.
   즉 모델·손실보다 교사 품질이 먼저다 — 계획의 큰 교사(EXAONE-32B·gpt-oss-120b)로 가기 전에 **같은 ls-test/ks-eval 후보에서 κ 와 A 비율을 먼저 재는 것**이 싼 관문이다.
2. **N 은 지금 방식이면 해롭거나 무익.** 블라인드 점검에서 절반이 좋은 경계였고, 상향 가중을 끈 r2 가 EN 에서 P·R 모두 약간 낫다(bias 0 F1 0.17 → 0.23; 표본이 작아 방향만).
3. **Stage A 는 발화 끝을 놓친다.** 짧은 KO 발화(평균 7 단어)에서 명백히 완결된 끝을 후보로 안 내는 경우가 있어, 모델의 맞는 commit 이 조기 commit 으로 채점된다.
4. **TURN 은 길이 지름길.** EN 학습 스트림이 모두 24–34 s 이고 TURN 은 늘 끝에 있어, 모델이 22–26 s 부근에서 발화 중에도 TURN 을 낸다(조기 27–52/100). KO 도 조기 60–78/300.
5. **KO 평가셋은 발화 끝 commit 만 잰다**(ks-eval A 의 90 %). 중간 commit 을 재려면 같은 세션의 연속 발화를 이은 다문장 KO 스트림이 필요하다.
6. **지연–정확도**: δ4 commit 은 단어 끝 뒤 0.33–0.35 s, δ2 는 0.17–0.20 s 지만 ASR 이 나빠진다(EN WER 7.8 %, KO CER 16.0 %).

## 5. mxc 본 실행 제안
- 교사 관문: 큰 교사로 ls-test·ks-eval 후보만 먼저 Stage B 를 돌려 κ·A 비율·블라인드 점검 정밀도를 이 표와 비교(`experiments/semcommit_teacher.py`, 프롬프트 v0.2 그대로).
  gpt-oss 는 final 채널 로그확률 방식에서 WAIT 편향이 있었으므로 120b 도 분포부터 확인(짧은 analysis 허용 여부 포함).
- Stage A 후보에 **스트림 마지막 단어를 항상 추가**(B/C 가 판정) — 발화 끝 누락 제거.
- N 은 사람 전사 disfluency 표지와 고마진 SELF_REPAIR/RESTART 로만, 나머지는 B 로. 또는 `--hardneg-weight 0` 을 기본으로.
- 스트림 길이를 넓게 섞고(예: 5–40 s) 끝 무음 길이도 무작위화해 TURN 의 길이 지름길을 끊는다. KO 는 세션 연속 발화를 이은 다문장 스트림을 만든다.
- EN 평가에 `semcommit_pc_eval.py` 의 P_pc·R_pc 를 A 참조 지표와 함께 보고(라벨 품질과 모델 품질을 분리).

## 불확실성
- 블라인드 점검 판정자는 Claude(두 관점)이지 사람 주석이 아니다. 표본은 등급·셋당 10–30 개라 비율의 신뢰구간이 넓다.
- 학습 데이터가 작고(EN SEM 양성 367, KO 712) 시드 1 개라 r1↔r2 차이는 방향만 믿을 수 있다.
- PC 문장 끝은 SEM_END 의 정의가 아니다(계획 §3). 낭독체 EN 에서 교차 점검용일 뿐, 쉼표 뒤 완결 절처럼 문장 끝이 아닌 정당한 commit 도 있다.
- EN 두 번째 판정자 교체(gpt-oss → EXAONE-3.5)는 이 소규모 실행에서만의 결정이다. 코드 기본값(`PRIMARY_JUDGES`)은 계획대로 두었다.

## 근거
- 원자료: `raw/sources/experiments/2026-09-24-semcommit-v0.2-rack4/` — `labels/`(등급 통계), `eval/`(보고서·PC 대조 `pc-ls-test.json`·`--verify` 결과), `runs/`(parity·실행 기록), `spot/`(블라인드 표본·키·판정), `scripts/`(rack4 실행 스크립트).
- rack4 산출물: `/data4/tskim/semcommit/{data/v0, labels/v0.2, runs/v0.2-r1, runs/v0.2-r2, eval/v0.2-r1, eval/v0.2-r2}`, 로그 `/data3/tskim/logs/semcommit-*.log`.
- 코드: 브랜치 `tskim/query/semcommit-streaming-asr` — `vapasr/data/semcommit_*.py`, `vapasr/hf/commit_metrics.py`(PC 대조 `pc_punct_after`·`pc_commit_counts` 추가), `experiments/semcommit_{build_words,teacher,train,eval,pc_eval}.py`.
