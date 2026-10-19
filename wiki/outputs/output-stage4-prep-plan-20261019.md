---
type: output
status: active
created: 2026-10-19
updated: 2026-10-19
summary: Stage 4(교차 언어 2 인 대화 + 번역) 준비 — 대화 번역 시험 생성 완료·전체 생성 제출 대기, 네 흐름 형식·교차 언어 합성·번역 평가 도구 구현(오라클 만점), 학습 옵션
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
| S1 | 실제 대화 번역 타깃 | **전체 생성 중**(apex job 91797 → 91798 → 91799 → 91800) | 아래 절 |
| S2 | 네 흐름 시퀀스(두 화자 × 전사·번역), 테스트, dry-run | **완료** | 아래 절 |
| S3 | 교차 언어 합성 대화(이어 붙이기) | **구현·시험 완료**, 본 생성은 S1 뒤 | 아래 절 |
| S4 | 대화 번역 평가 도구 | **도구 완료(오라클 만점)·TAXI 어댑터** | 평가 셋은 S1 뒤. xCOMET 은 다음 |
| S5 | 학습 스크립트·섞기·스모크 | 옵션 완료 | `--dlg-trans`·교차 언어 셋 `sim_<l1>-<l2>`. 섞는 비율·스모크는 타깃이 나온 뒤 |
| S6 | 파일럿 → 본 학습 | 대기 | `low_p_hpc`(지금 쓸 수 있는 파티션) |
| S7 | TTS 합성(병행) | **결정 필요** | 아래 절 — 엔진·원천 대화·내려받기(사용자) |

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
- **전체 생성 제출(사용자, 2026-10-19):** 일반 생성 → 수리 → 되살림 생성 → 수리를 `afterany` 로 이었다(job 번호는 서브셸 안에서만 넘김).
  - 첫 제출(91684–91687)은 hpc-50·51 에서 env 복사 `srun` 이 munge 인증 오류(`/run/munge/munge.socket.2: Permission denied`)로 바로 실패했다. 남은 연쇄(91685–91687)는 취소를 요청했다.
  - 제출 스크립트에 `--exclude` 를 더하고 hpc-50·51 을 빼서 다시 냈다: 91697(hpc-101·102, 09:30 시작) → 91698 → 91699 → 91700
    - 91697 은 env 복사 단계(정상 20–30 s)에서 30 분 넘게 출력 없이 멈췄다. 취소를 요청했다.
  - hpc-50·51·101·102 를 빼고 세 번째로 냈다: **91797(hpc-20·92, 10:18 시작 — env 복사 21 s, 16 rank × 14 항목) → 91798 → 91799 → 91800**
- **남은 일:** 끝나면 두 루트 각각 `ast_build_targets --fix-dir fix-gen-v3-bf16 --pack`
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

## S3 교차 언어 합성 대화 (`sim_build_dialogues.py --lang2`)
- 화자 a·b 를 언어가 다른 두 풀에서 고르고, 발화의 번역(words 행 = Stage 2 타깃, 대화 발화 = S1 타깃)을 들고 다니게 했다.
- 데이터셋은 슬롯을 언어로 고정한다. 어느 언어가 A 인지는 무작위다. 각 화자의 번역은 상대 언어다.
- 시험(KO↔EN 24 세션, 시험 타깃)에서 행 무결성 문제를 찾아 고쳤다. 한국어는 행 끝에서 번역이 확정되므로, 행 중간 조각에서 세션이 시작하면 말하지 않은 내용의 번역이 붙었다.
- 고친 뒤 dry-run 50 창 문제 0, 발화 693 개 모두 번역 행이 있다.
- 본 생성은 S1 타깃이 나온 뒤다: 학습 셋 `sim_en-de`·`sim_ko-en`, EN↔DE held-out 평가 셋(`--heldout`). KO↔EN 평가 셋은 사용자 결정을 기다린다.

## S4 번역 평가 도구 (`experiments/cst_stage4_eval.py`)
- 같은 언어 held-out 대화(목표 언어마다)와 교차 언어 합성 평가 셋을 같은 도구로 잰다.
- 지표: 받아쓰기·귀속, `<EOT>`, 번역 BLEU·chrF(방향별), 학습 타깃 대비 번역 지연, 빈 번역 비율
- 오라클(정답을 출력으로): 번역 100, 지연 0, 귀속 오류 0
- 출발점: Stage 3 최종을 번역 prefix 로 그대로 쓰면 합성 KO↔EN 에서 형식을 따르지 못한다. 받아쓰기 오류가 100 % 안팎이고 번역 BLEU 는 9–17 이다.
- TAXI(실제 독↔영 전화 대화, 86 세션) 어댑터: 번역을 cstbench 가설로 써서 cstbench 채점기로 잰다.
  - 출발점(Stage 3 최종 그대로, δ10): 영→독 BLEU 18.0, 독→영 3.7, 받아쓰기 WER 100 % 안팎
  - 받아쓰기 prefix 로도 슬롯 순열 허용 WER 53 % 다. 8 kHz 전화 음성이 크다(전화 대역 증강 후보).
- 남은 것: xCOMET

## S5 학습 스크립트 (`experiments/stage3_train.py`)
- `--dlg-trans`(대화 번역 타깃), `--dlg-trans-input`(되살린 행 번역), `--dlg-ast-frac`(기본 0.5)
- 교차 언어 합성 셋: 가중치 키 `sim_<l1>-<l2>`, `--sim-trans-frac`(기본 0.8)
- 학습 전 검사에 번역 모드를 더했다. 끄면 Stage 3 와 같고 재개 지문도 그대로다.
- 섞는 비율·스모크는 타깃이 나온 뒤 정한다.

## S7 TTS 합성 (병행 준비 — 결정 필요)
- 계획 원천: [[output-cst-bench-syn-final-plan-20261003]](평가셋용). 학습 데이터로 옮기면 다음과 같다.
  - 원천 대화: XDailyDialog(일상, 영·독 전문 번역), BConTrasT(과제형, 영·독). 한국어판은 Qwen3.8-27B 번역(silver)
  - 엔진: 공개 TTS(Qwen3-TTS 또는 CosyVoice 3, Apache-2.0)를 mxc GPU 로. Azure 는 평가셋용이다.
  - 그 뒤는 지금 파이프라인과 같다: 턴 음성 → 정렬(단어 시각) → 이어 붙이기(sim_build 시간 구조) → 번역 확정 시각(Stage 2 방식)
- 필요한 결정(사용자):
  1. 엔진(Qwen3-TTS · CosyVoice 3)
  2. 원천 대화(XDailyDialog·BConTrasT, 또는 LLM 이 쓴 대화)
  3. 규모(언어쌍마다 시간)
- 원천 텍스트·TTS 가중치는 T5 로 내려받고, 서버 업로드는 사용자가 한다.
