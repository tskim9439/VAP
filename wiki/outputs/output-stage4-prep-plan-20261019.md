---
type: output
status: active
created: 2026-10-19
updated: 2026-10-19
summary: Stage 4(교차 언어 2 인 대화 + 번역) 준비 계획과 진행 — 대화 번역 시험 생성 완료(관문 96–99 %)·전체 생성 제출 대기, 네 흐름 형식 구현·dry-run 통과, 교차 언어 합성 대화 다음
sources:
  - '[[decision-stage4-prep-20261019]]'
  - '[[decision-cst-training-stages-and-speaker-slots]]'
  - '[[decision-ast-monotonic-data]]'
  - '[[output-ast-mono-generation-20261009]]'
  - '[[output-stage3-prep-plan-20261014]]'
  - '[[2026-10-19-stage4-dialogue-translation]]'
raw_authors:
  - tskim
---

# Stage 4 준비 계획 (2026-10-19)

## 목표와 결정
- **Stage 4:** 교차 언어 2 인 대화 + 번역. 같은 언어 대화의 번역도 섞는다. 주 언어쌍은 EN↔DE, 둘째는 KO↔EN 이다.
- **출력 형식:** 언어로 고정한 두 슬롯 + `<TRANS>` + 턴 끝 `<EOT>`([[decision-cst-training-stages-and-speaker-slots]])
  ```
  language <SPK_A>Korean <SPK_B>English
  translate <SPK_A>English <SPK_B>Korean<asr_text>
  ```
- **결정(2026-10-19, [[decision-stage4-prep-20261019]]):**
  - 실제 같은 언어 대화를 지금 번역한다.
  - 교차 언어는 이어 붙인 합성 대화만 쓰고, TTS 합성은 병행 준비한다.
  - `<EOT>` 는 말이 끝날 때 낸다. 지연은 δ4–10 이다.
  - 한국어–영어 평가셋은 미룬다.
- **시작점:** Stage 3 최종 p11 checkpoint-3000

## 작업과 상태

| # | 작업 | 상태 | 비고 |
|---|---|---|---|
| S1 | 실제 대화 번역 타깃 | **시험 생성 완료, 전체 생성 제출 대기** | 아래 절 |
| S2 | 네 흐름 시퀀스(두 화자 × 전사·번역), 테스트, dry-run | **완료** | 아래 절 |
| S3 | 교차 언어 합성 대화(이어 붙이기) | 다음 | `sim_dialogue` 를 언어가 다른 두 화자로 넓힌다. 번역은 Stage 2 타깃(단일 화자 행)·S1 타깃(대화 발화) |
| S4 | 대화 번역 평가 도구 | 대기 | `cst_stage3_eval` 에 번역 채점(BLEU·chrF·xCOMET), DE↔EN 은 TAXI(CST-Bench) |
| S5 | 학습 스크립트·섞기·스모크 | 일부 | 대화 번역 옵션 추가(`--dlg-trans`). 교차 언어 셋·스모크는 S3 뒤 |
| S6 | 파일럿 → 본 학습 | 대기 | `low_p_hpc`(지금 쓸 수 있는 파티션) |
| S7 | TTS 합성(병행) | 대기 | [[output-cst-bench-syn-final-plan-20261003]] 의 TTS 계획부터 다시 본다 |

## S1 실제 대화 번역 타깃
- **방식:** Stage 2 생성([[output-ast-mono-generation-20261009]])을 그대로 쓴다.
  - Qwen3.8-27B, 프롬프트 v3, 단조 번역 + 판정 + 원문 접두 번역 → 단어별 확정 시각(stable prefix)
  - 판정 v3 + 한 번 수리
  - 방향: KO→EN, DE→EN, EN→DE·KO
- **원문 단위(`experiments/stage4_dialogue_units.py`):** 정제 대화의 발화를 `ast_mono_generate` 의 단위 형식으로 쓴다.
  - 정제가 나눈 조각은 원 발화로 다시 묶는다(행).
  - 단어 시각은 정렬 토큰에서 온다(Qwen3-ASR 토크나이저로 다시 토큰화해 id 대조 → offset 으로 단어에 묶음).
  - 원문은 `raw`(대소문자·문장부호)를 단어 시각에 문자 정렬한다. 영어·NSC·KEC 철자가 살아난다.
  - 일반 단위 `stage4-dlg-units-v1`: 영어 105 만·한국어 258 만·독일어 80 만 단어, 다시 토큰화 실패 0
  - 되살린 격리 발화 단위 `stage4-dlg-units-revive-v1`: 한국어 169 만·독일어 12 만 단어. 원문이 의심스러워 그 번역은 입력 문맥으로만 쓴다(S2).
- **시험 생성(코퍼스당 3 대화):** 관문 통과는 수리 뒤 en→de 98.9 %, en→ko 98.1 %, ko→en 95.7 %, de→en 98.6 % 다. Stage 2 단일 화자(96.3·92.7·97.8·99.3 %)와 비슷하다. 행 유지는 95.6–98.8 % 다.
- **분량:** 단어 × 방향 약 730 만(일반 548 만 + 되살림 181 만) → Stage 2 기준(2.5 만 단어·방향 ≈ 8 GPU 분) 약 39 GPU 시간, 2 노드면 2.5 시간 남짓
- **남은 일:** 전체 생성은 사용자 제출이다. 두 루트 각각 생성 → 수리 → `ast_build_targets --fix-dir fix-gen-v3-bf16 --pack`
- 자세한 수치: [[2026-10-19-stage4-dialogue-translation]]

## S2 네 흐름 시퀀스 (`vapasr/data/cst_dialogue.py`)
- **형식:** 흐름 = 슬롯 A·B 전사 + 그 번역. prefix 에 translate 줄을 넣는다. 목표 언어는 한·독 → 영, 영 → 독·한(무작위)이다.
  - 학습 창은 `ast_frac` 확률로 번역 모드, 나머지는 Stage 3 형식이다.
  - 청크·태그 규칙은 Stage 2 와 같다. 구간이 바뀔 때만 `<SPK_x>`·`<TRANS>`·`<SPK_x><TRANS>` 를 단다.
- **`<EOT>`:** 청크는 Stage 3 규칙(말이 끝난 때) 그대로다. 그 episode 의 전사·번역 꼬리 중 나중 것 뒤에 온다.
- **입력만·가림:**
  - 되살린 행의 번역은 입력 문맥으로만 쓴다(label −100).
  - 번역이 없는 행(판정 탈락, 창 시작 전에 시작한 행)은 그 방출 청크의 태그·`<NEXT_AUDIO>`·`<EOT>` 를 가린다. 전사는 학습한다.
- **dry-run(시험 타깃, 7 코퍼스 386 창):** 문제 0, 길이 16,384 초과 0
  - 1 차에서 번역이 원문 전사보다 앞서는 경우와 `<EOT>` 가 마지막 전사 단어보다 앞서는 경우를 찾아 고쳤다. `<EOT>` 의 25–41 % 가 해당했다.
  - 번역 모드 시퀀스는 Stage 3 보다 4–26 % 길다(p50).

## S5 학습 스크립트 (`experiments/stage3_train.py`)
- `--dlg-trans`(대화 번역 타깃), `--dlg-trans-input`(되살린 행 번역), `--dlg-ast-frac`(기본 0.5)
- 학습 전 검사에 번역 모드를 더했다. 끄면 Stage 3 와 같고 재개 지문도 그대로다.
- 교차 언어 합성 셋(S3)과 섞는 비율·스모크는 S3 뒤에 정한다.
