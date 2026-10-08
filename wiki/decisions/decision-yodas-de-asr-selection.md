---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-11-08
created: 2026-10-08
updated: 2026-10-08
summary: YODAS-Granary 독일어 ASR 학습 데이터 — 원 전사↔Qwen3-ASR WER ≤ 3 % 채택, Qwen 원출력을 영어 규약으로 정규화(asr-tn-de-v1.1.0) 2,125 h, Whisper 재전사 없음
sources:
  - '[[output-yodas-granary-de-transcript-check-20261008]]'
  - '[[2026-10-08-yodas-de-align-smoke]]'
  - '[[2026-10-08-yodas-de-align-run]]'
  - '[[2026-10-08-stage1-trilingual-d10-prep]]'
related:
  - '[[decision-eval-train-decontamination]]'
  - '[[decision-asr-tn-v1-freeze]]'
  - '[[decision-stage1-lookahead-320-1040ms]]'
---

# YODAS-Granary 독일어 ASR 학습 데이터 선별 (2026-10-08)

## 맥락
- 독일어 실시간 ASR 을 함께 학습하려고 YODAS-Granary 독일어(약 7,925 h)를 전처리한다.
- 사용자 원안은 Whisper-large-v3·Qwen3-ASR-1.7B 두 모델로 다시 전사해 원 전사와 비교하는 것이었다.
- 점검 결과([[output-yodas-granary-de-transcript-check-20261008]]):
  - Granary 원 전사는 Whisper-large-v3 의사 라벨 그대로라, v3 재전사는 독립 검증이 되지 않는다.
  - 원 전사와 Qwen 이 맞는 비율은 시간 기준 절반 정도다. 원 전사는 말 그대로가 아니다(간투사·반복 삭제, 절 누락).

## 선택지
1. 원안: 원 전사가 Qwen·Whisper 와 모두 맞으면 원 전사, 아니면 두 모델끼리 맞으면 Qwen 전사
2. 원 전사↔Qwen 일치만으로 선별
3. ASR 필터 없이 원 전사 그대로

## 결정 (사용자 동의 2026-10-08)
- **선별**: 원 전사와 Qwen3-ASR-1.7B 전사의 WER(독일어 정규화 `norm_de`)이 **3 % 이하**인 발화만 쓴다. Whisper 재전사는 하지 않는다.
- **학습 타깃**: 일치한 발화도 **Qwen 원출력**(말 그대로)을 쓴다. 시각은 이 텍스트에 정렬한다. 원 전사는 `raw_text` 로 남긴다.
- **정규화(2026-10-08 추가)**: 영어 학습 타깃(asr-tn-v1)은 소문자·무구두점이고 한국어도 학습 때 구두점을 지운다. 그래서 Qwen 원출력을 같은 규칙으로 바꾼다 — `asr-tn-de-v1.1.0`(`vapasr/data/textnorm_de.py`).
  - 소문자로 바꾸고 구두점을 지운다. 하이픈은 띄어 쓴다("YouTube-Kanal" → "youtube kanal").
  - 영어처럼 숫자가 남은 행은 제외한다(연도·기수·서수를 오디오 없이 가를 수 없다).
  - 약어("Dr.", "bzw.", "z.B.")·기호(%, €, &)·비라틴 문자(θ, ş)가 있는 행도 제외한다(발음과 글자가 다르다).
  - 원출력은 `display_text` 에 남겨 나중에 표기형(구두점·대소문자) 타깃의 원천으로 쓴다.
  - v1.0.0(`tn1`)은 비라틴 문자를 지우기만 하던 버그가 있어 쓰지 않는다. v1.1.0 은 `tn2` 에 새로 만들었다.
- **시각**: Qwen3-ForcedAligner-0.6B. 클립 끝을 넘는 시각은 영·한과 같은 규칙으로 처리한다(80 ms 이하는 잘라 맞춤, 그 이상 제외).
- **길이**: 1–40 s. 30 s 초과 발화도 쓴다(사용자 요청 2026-10-08). 30–40 s 는 별도 처리(long pass)로 돌리고, 행은 자르지 않고 통째로 쓴다.
- **사전 제외**: 1 s 미만, 빈 전사, 환각 문구, 3-gram 반복, 발화 속도 이상, 비라틴 문자, 중복 구간, 평가셋 8-gram 겹침([[decision-eval-train-decontamination]]).
- **dev**: 영상 200 개(영상 ID 해시 순)를 통째로 떼어 둔다.
- `<SEM_END>` 등 의미 단위 표시는 이번 단계에서 넣지 않는다.

## 결과와 영향
- 본 처리(1–30 s) 실제 채택: 2,045.4 h(1,340,112 발화; dev 4,903 발화). 영·한 선별 데이터(약 2,360 h)와 비슷한 규모다.
- 30–40 s(다른 사유 없는 1,394.5 h) 실제 채택: 231.1 h(23,086 발화; dev 96 발화). 시간 기준 채택률은 17 % 로, 1–30 s(32 %)보다 낮다. Granary 원 전사가 긴 발화에서 덜 맞기 때문이다.
- 합계: **2,276.5 h**(1,363,198 발화; words train 1,358,199 행, dev 4,999 행).
- 정규화 뒤 학습 데이터: **2,125.3 h**(1,315,348 행). 제외는 숫자 128.8 h, 약어 13.3 h, 기호 0.6 h, 비라틴 문자 0.2 h 다. dev 는 4,861 행(7.87 h)이고, 독일어 평가셋(`single-turn-yodas-de-dev-v1`)으로도 쓴다.
- 학습에는 30 s 상한이 없다(토큰 예산 묶음). 다만 가장 긴 항목이 30 s → 40 s 로 늘어 묶음 예산·메모리를 한 번 확인해야 한다.
- Whisper-v2 를 세 번째 표로 넣지 않아 GPU 약 100 시간을 아끼고, 유지량은 약 10 % 더 많다.
- 대가: 판정 모델이 Qwen 하나라, Qwen 과 원 전사(Whisper-v3)가 함께 틀린 경우는 걸러지지 않는다.

## 구현
- `experiments/yodas_de_prepare.py`(inventory, prefilter), `experiments/yodas_de_align.py`(전사·선별·정렬·words 행·오디오 tar; 30–40 s 는 `--select-flags dur_long --max-dur 40`), `slurm/yodas-de-align.sbatch`, `slurm/submit-yodas-de-align.sh`(30–40 s 는 `--long`)
- 출력: mxc `VAPKT-data/data/yodas-de-v1/results/<shard>/words-{train,dev}.jsonl`, `audio.tar`(+ 색인). set 이름 `yodas-de`, 언어 `German`
- 정규화: `experiments/yodas_de_textnorm.py --tag tn2` → `yodas-de-v1/tn2/`, 학습 목록 `yodas-de-v1/lists/de-words-train-tn2.list`(파일 4,556 개)
- 시험 실행·검증: `raw/sources/experiments/2026-10-08-yodas-de-align-smoke/`, 전량 실행: `raw/sources/experiments/2026-10-08-yodas-de-align-run/`

## 재검토
- 2026-11-08 또는 독일어 학습 첫 결과가 나올 때. 독일어 평가(FLEURS·CoVoST2)에서 Granary 쪽 오류가 보이면 문턱을 다시 본다.
- FLEURS·CoVoST2 독일어를 받으면 평가셋 겹침 검사를 다시 돌린다.
