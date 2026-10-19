---
type: output
status: active
created: 2026-10-19
updated: 2026-10-19
summary: Stage 0–2(0.6B q06-s0 → s1m3-d10 → s2·s3·s4-ast) 학습 DB 목록 — 영 2,143 h·한 6,137 h·독 2,125 h + CV4 영 351 h, 번역 타깃 639 만 행, 잡음 뱅크
sources:
  - '[[output-staged-training-plan-20261001]]'
  - '[[output-stage1-trilingual-d10-prep-20261008]]'
  - '[[output-ast-mono-generation-20261009]]'
  - '[[output-q06-s4-ast-eval-20261013]]'
  - '[[2026-10-09-ast-mono-generate-prep]]'
raw_authors:
  - tskim
---

# Stage 0–2 학습 DB 목록 (0.6B, 2026-10-19 정리)

## 요청
- 사용자(2026-10-19): "Stage 2 까지 학습에 사용한 DB 목록을 작성해줘."
- 대상 모델 계열: `q06-s0`(Stage 0) → `q06-s1m3-d10`(Stage 1) → `q06-s2-ast` → `q06-s3-ast` → `q06-s4-ast`(Stage 2 최종, Stage 3 시작점)

## 시작 가중치(사전학습 모델 — 우리가 학습에 쓴 DB 가 아님)
- Qwen3-ASR-0.6B: 언어 모델(thinker)·어댑터 초기값
- Nemotron 스트리밍 FastConformer(.nemo): 음성 인코더 초기값

## 단계별로 쓴 데이터

| 단계 | 학습 대상 | 데이터 |
|---|---|---|
| Stage 0 `q06-s0` | 어댑터만(final 모드, 2,000 step) | 영·한 words 목록(아래 표의 영어·한국어 전부) |
| Stage 1 `q06-s1m3-d10` | 인코더·어댑터·언어 모델 전체(3 epoch) | 영·한·독 words 목록(아래 표 전부, CV4 제외) + 잡음 뱅크 |
| Stage 2 `q06-s2-ast` · `s3-ast` | 전체 | Stage 1 과 같은 목록 + 단조 번역 타깃(`ast-mono-v1`, 593 만 행) |
| Stage 2 `q06-s4-ast` | 전체(s3 에서 이어 2 epoch) | 위 + Common Voice 4 영어 + 그 번역 타깃(`ast-mono-cv4en-v1`) |

## 음성 DB (Stage 1 학습 목록, `ast-mono-v1` prepare 의 세트별 집계)

| 언어 | 세트 | 행(스트림) | 시간 | 내용 |
|---|---|---:|---:|---|
| 영어 | `librispeech-960` | 60,913 | 448.3 h | LibriSpeech 960 h 낭독 |
| 영어 | `librispeech-960-rg` | 126,952 | 576.8 h | 같은 코퍼스에서 기본 풀(8–30 s)에 못 든 스트림을 발화 경계로 자르거나 같은 화자·장의 짧은 발화를 이어 붙인 것(regroup — 중복 아님) |
| 영어 | `voxpopuli-train` | 121,151 | 477.4 h | VoxPopuli 영어(유럽 의회 연설) |
| 영어 | `voxpopuli-train-rg` | 14,763 | 73.1 h | 같은 방식 regroup |
| 영어 | `yodas-en129` | 45,948 | 188.3 h | YODAS 영어 en129 부분(의사 라벨) |
| 영어 | `yodas-en129-rg` | 16,191 | 92.2 h | 같은 방식 regroup |
| 영어 | `swbd-train` | 43,643 | 135.0 h | Switchboard 전화 대화(한쪽 채널) |
| 영어 | `swbd-train-rg` | 27,332 | 152.3 h | 같은 방식 regroup |
| **영어 합** | | **456,893** | **2,143 h** | |
| 한국어 | `speechlm-partial` | 1,424,279 | 2,613.6 h | 사내 speechlm 수집본(19 개 DB) 중 QC·교사 관문 통과분 |
| 한국어 | `nikl-1000` | 1,205,360 | 1,392.9 h | 국립국어원(NIKL) 음성 말뭉치 |
| 한국어 | `kspon-full` | 619,034 | 1,186.0 h | KsponSpeech 전체 |
| 한국어 | `aihub-bc-train` | 431,447 | 706.6 h | AI Hub 031/033 방송 통번역 |
| 한국어 | `aihub71631-train` | 228,264 | 238.2 h | AI Hub 71631 자유대화(스테레오, 화자 = 채널) |
| **한국어 합** | | **3,908,384** | **6,137 h** | 학습 로그는 main 2,956,210 + 짧은 발화 945,016 스트림(4,990 + 1,122 h) — 로더가 7,158 행을 더 걸렀다 |
| 독일어 | `yodas-de` | 1,315,348 | 2,125.3 h | YODAS 독일어(Qwen3-ASR-1.7B 의사 전사, `asr-tn-de-v1.1.0` 정규화 tn2) |
| 영어(Stage 2 s4) | `cv4-en` | 232,845 | 정렬 351 h 중 train | Common Voice 4 영어 낭독(CoVoST 2 test·dev 클립·화자 제외) |

- 목록 파일(mxc `VAPKT-data/data/`):
  - 영·한 main·short: `semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637/{main,short}-words.list`
  - 영·한 ASR words: `semcommit-work/asr-words-v1/lists/asr-words-20260929.list`
  - 독일어: `yodas-de-v1/lists/de-words-train-tn2.list`
  - CV4 영어: `cv4-en-v1/lists/cv4-en-words-train.list`
- 전사 타깃은 정렬 단어(Qwen3-ForcedAligner)의 끝 시각을 갖는다. 영어는 `asr-tn-v1`, 한국어는 학습 때 구두점 제거, 독일어는 `asr-tn-de-v1.1.0` 이다.
- 단어 시각 정렬·전사 정규화·교사 의사 라벨의 출처는 각 세트의 준비 기록([[output-stage1-trilingual-d10-prep-20261008]], [[output-speechlm-semcommit-approval-audit-20260925]])에 있다.

## 번역 타깃 (Stage 2, Qwen3.8-27B 단조 번역 + 판정 + 한 번 수리, 행 = 위 음성 행)

| 묶음 | 방향 | 보존 행 | 번역 토큰 |
|---|---|---:|---:|
| `ast-mono-v1` | 독→영 | 1,303,230 / 1,315,348 | 2,511 만 |
| `ast-mono-v1` | 영→독 | 424,117 / 456,893 | 2,845 만 |
| `ast-mono-v1` | 영→한 | 395,109 / 456,893 | 2,926 만 |
| `ast-mono-v1` | 한→영 | 3,805,073 / 3,908,384 | 6,235 만 |
| `ast-mono-cv4en-v1`(s4) | 영→독 | 230,083 | – |
| `ast-mono-cv4en-v1`(s4) | 영→한 | 228,474 | – |

- 합계 593 만 → 639 만 행(s4). 생성 방법은 [[output-ast-mono-generation-20261009]]

## 증강 데이터 (Stage 1 부터)
- 잡음 뱅크 `noise-bank-v1`(6.4 GB): MUSAN noise 1,059 파일(8.9 h)·music 660 파일(42.6 h), RIRS_NOISES 잔향 6,325 개(simulated 방마다 2,000 + real 325). MUSAN speech 는 쓰지 않았다.
- 그 밖: SpecAugment, 속도 변형 0.9·1.0·1.1, 영어 음량 증강(Stage 2 s3 부터)

## 학습에 쓰지 않은 것(평가 전용)
- 받아쓰기: `single-turn-vpaa-kspon-v1`(VoxPopuli-AA·Kspon eval), `single-turn-yodas-de-dev-v1`, LibriSpeech test-other
- 번역: FLEURS, CoVoST 2(평가셋 화자·클립은 CV4 학습에서 뺐다)
