---
type: output
status: active
created: 2026-10-09
updated: 2026-10-09
summary: AST 단조 번역 전체 생성 준비 — Stage 1 학습 행 568 만 개를 번역 단위 733 만 개로 나누고, 4 방향·단어마다 stable prefix 를 vLLM SLURM job 으로 만든다(약 675 GPU 시간)
sources:
  - '[[2026-10-09-ast-mono-generate-prep]]'
  - '[[2026-10-09-ast-vllm-bench]]'
related:
  - '[[decision-ast-monotonic-data]]'
  - '[[output-monotonic-ast-translation-methods-20261008]]'
  - '[[output-stage1-trilingual-d10-prep-20261008]]'
raw_authors:
  - tskim
---

# AST 단조 번역 전체 생성 준비 (2026-10-09)

## 요청
- 사용자: 4 방향(영→독·영→한·독→영·한→영)을 한 번에 돌리고, stable prefix 는 가장 정확하게 원문 단어마다 만든다([[decision-ast-monotonic-data]]).

## 방법 (`experiments/ast_mono_generate.py`)
- **입력:** 0.6B Stage 1(q06-s1m3-d10) 학습 목록 그대로다. 영어·한국어는 speechlm main + ASR words, 한국어 짧은 발화는 speechlm short, 독일어는 YODAS-de tn2 다.
- **원문 텍스트:** 행 구간마다 쓸 수 있는 첫 텍스트를 고른다.
  - 순서: `pnc_text` → `display_text` → `raw_text` → 정렬 단어 자체
  - `raw_text` 는 주석을 지운다: `<LAUGH>`, `{laughing}`, KsponSpeech `o/ b/`, `(70%)/(칠십 퍼센트)` 는 앞쪽 표기를 남긴다.
  - 대문자뿐인 LibriSpeech 텍스트는 소문자로 바꾼다.
- **단어 시각:** 학습 타깃인 강제 정렬 단어의 끝 시각을 원문 토큰에 옮긴다.
  - 토큰 수가 같으면 1:1 로 옮긴다.
  - 다르면 글자(문자·숫자) 정렬로 옮긴다. 그래서 "1901" 은 "... nine hundred one" 의 끝 시각을 받는다.
  - 문자 일치가 85 % 미만이거나 정렬 단어 쪽 일치가 60 % 미만이면 다음 텍스트로 넘어간다.
- **번역 단위:**
  - 문장 끝(. ? ! …)에서 나눈다. 약어·독일어 서수·소문자로 이어지는 경우는 나누지 않는다.
  - 40 단어를 넘으면 가운데에 가까운 쉼표에서, 쉼표가 없으면 가장 긴 쉼 위치에서 나눈다.
  - 행 구간(여러 발화를 이은 스트림)은 따로 다룬다.
- **생성:** 파일럿 v2 프롬프트를 그대로 쓴다. 단위마다 다음을 만든다.
  - 단조 번역 1 개
  - 원문 단어 prefix 마다 번역 1 개
  - 판정 1 개
  - 일반 번역은 만들지 않는다.
  - 방출 시각은 stable prefix 로 정한다. 시각은 행(스트림) 기준이다.
- **실행:** `slurm/ast-mono-generate.sbatch`(제출은 `slurm/submit-ast-mono-generate.sh`)
  - 노드마다 vLLM env tar 를 풀고 C 컴파일러를 확인한 뒤, GPU 8 장에 엔진을 하나씩 띄운다.
  - 작업 항목(방향 × 샤드)은 순위별로 고정 배분한다. 끝난 항목은 건너뛰므로 선점되면 requeue 해 이어 간다.

## 원문 단위 (prepare, 호스트 CPU 12 분)

| 언어 | 행 | 시간 | 번역 단위 | 원문 단어 | 샤드(약 2.5 만 단어) |
|---|---:|---:|---:|---:|---:|
| 영어 | 456,893 | 2,143 h | 926,008 | 18,140,406 | 725 |
| 한국어 | 3,908,384 | 6,137 h | 4,713,360 | 30,834,926 | 1,234 |
| 독일어 | 1,315,348 | 2,125 h | 1,692,739 | 19,739,272 | 790 |

- 행 수와 시간은 학습 로그와 맞는다. 영어 456,893 은 같다. 한국어는 학습 로그의 main 2,956,210 + short 945,016 보다 7,158 행 많다(학습 로더가 거른 행으로 보임). 독일어 1,315,348 은 같다.
- 원문 출처:
  - YODAS·VoxPopuli 영어는 `pnc_text`, speechlm 한국어와 YODAS 독일어는 `display_text` 를 썼다.
  - LibriSpeech·Switchboard·AIHub·Kspon·NIKL 은 `raw_text` 를 썼다.
  - 정렬 단어로 내려간 구간은 세트마다 1 % 미만이다(`raw/sources/experiments/2026-10-09-ast-mono-generate-prep/manifest-stats.json`).
- 작업 항목은 3,474 개다(영어 샤드는 두 방향이 같이 쓴다). prefix 호출은 약 8,700 만 번이다.

## 실제 크기 시험 (호스트 H200, 항목 하나씩)

| 항목 | 단위 | 호출 | 시간 | 관문 통과 | AL |
|---|---:|---:|---:|---:|---:|
| en-de u-00000 (VoxPopuli·LibriSpeech-rg) | 1,024 | 26,046 | 770 s | 76 %(VoxPopuli 84 %, LibriSpeech 67 %) | 4.32 s |
| ko-en u-00000 (speechlm) | 4,518 | 29,518 | 617 s | 84 % | 2.68 s |

- **전체 추정:** 약 675 GPU 시간이다. H200 8 장 노드 4 개면 약 21 시간, 6 개면 14 시간, 8 개면 11 시간이다.
- **prefix cache 는 효과가 없다**(적중 0). linear attention 이 섞인 모델이라 캐시 블록이 프롬프트(약 265 토큰)보다 큰 것으로 보인다.
- **관문 탈락의 상당수는 원문 자체가 문장 조각인 경우다.** 원 데이터 구간이 문장 중간에서 잘린 것으로, 판정 LLM 이 조각 번역을 "불완전·문법 오류"로 본다. 소문자·무구두점 LibriSpeech 에서 탈락이 많다.
  - 판정은 번역과 따로 저장된다. 그래서 생성이 끝난 뒤 조각 원문을 인정하는 판정으로 다시 판정하거나(비용 약 3 %), 탈락 단위만 고쳐 번역할 수 있다. 앞부분 번역은 다시 만들 필요가 없다.

## 다음
- 사용자 제출: `bash slurm/submit-ast-mono-generate.sh --nodes N --partition P`
- 진행 확인: `python3 experiments/ast_mono_generate.py status --root /soundai/users/tskim/VAPKT-data/data/ast-mono-v1`
- 생성 뒤:
  - 판정 재검토(조각 원문)·탈락 단위 수리
  - 행 단위 AST 타깃 조립(⌊t/80 ms⌋ + δ)
  - 화자·언어 prefix 형식으로 학습 연결([[decision-stage1-lookahead-320-1040ms]])
