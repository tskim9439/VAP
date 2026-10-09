---
type: output
status: active
created: 2026-10-08
updated: 2026-10-09
summary: 영·한·독 Stage 1 학습 준비 — 독일어 정규화 2,125 h, 학습·평가 스크립트 독일어 지원, 지연 δ4–10(<DELAY_9/10>), 0.6B run 82738 진행
sources:
  - '[[2026-10-08-stage1-trilingual-d10-prep]]'
related:
  - '[[decision-stage1-lookahead-320-1040ms]]'
  - '[[decision-yodas-de-asr-selection]]'
  - '[[output-yodas-granary-de-transcript-check-20261008]]'
  - '[[output-staged-training-plan-20261001]]'
  - '[[output-stage1-pilot-eval-20261001]]'
raw_authors:
  - tskim
---

# 영·한·독 Stage 1 학습 준비와 0.6B 학습 (2026-10-08)

## 요청
- 사용자: 영어·한국어·독일어를 함께 실시간 전사하도록 학습을 준비하자.
- 이어서 미리 듣기 320–1,040 ms(화자 구분 대비)와 0.6B thinker(실시간성)를 정했다 → [[decision-stage1-lookahead-320-1040ms]].

## 1. 독일어 학습 텍스트 정규화
- 영어 학습 타깃(asr-tn-v1)은 소문자·무구두점이고, 한국어도 학습 때 구두점을 지운다. 독일어 행은 Qwen 원출력이라 구두점·대소문자가 있었다.
- `asr-tn-de-v1.1.0`(`vapasr/data/textnorm_de.py`, 영어 규칙을 따른 별도 모듈 — asr-tn-v1 지문은 그대로):
  - 소문자, 구두점 제거, 하이픈 띄어쓰기
  - 숫자·약어·기호·비라틴 문자 행 제외
- 행 변환(`normalize_de_row`)은 한국어 `normalize_ko_row` 와 같은 방식이다.
  - 단어의 원래 토큰에서 구두점 토큰만 빼서 철자가 맞으면 그대로 쓰고, 아니면 다시 토큰화한다.
  - 다시 토큰화하면 원래 토큰 시각을 순서대로 나눠 받으므로, 단어 끝 청크가 유지된다.
- 변환 결과(`experiments/yodas_de_textnorm.py --tag tn2`, mxc 호스트 CPU 약 10 분):

| | 남은 행 | 시간 | 제외 |
|---|---:|---:|---|
| train | 1,315,348 | **2,125.3 h** | 숫자 128.8 h · 약어 13.3 h · 기호 0.6 h · 비라틴 0.2 h |
| dev | 4,861 | 7.87 h | 숫자 0.42 h · 약어 0.05 h |

- 모든 행이 학습 로더와 같은 tokenizer 재분할 검사를 통과했다(재분할 실패 0).
- v1.0.0(`tn1`)은 θ·ş 같은 문자를 행 제외 없이 지우기만 했다(38 행). v1.1.0 에서 고쳐 `tn2` 로 새로 만들었고, dev 는 두 판이 같다.

## 2. 학습 스크립트
- `experiments/semcommit_train.py`:
  - `--de-words`(독일어 셋 `de`, 라벨 없는 ASR 행, 다른 셋은 이 파일을 읽지 않음), `--max-items-de`
  - `--delays` 9·10 허용 → `<DELAY_9/10>` 을 붙인다
- `slurm/submit-semcommit-train-apex.sh`: `--de-list`, δ 1–10 허용
- `vapasr/data/semcommit_tokens.py`: `add_delay_ext_specials`·`add_delay_tokens`
- `vapasr/hf/__init__.py`: `load_tokenizer` 가 `<DELAY_9/10>` 을 복원한다
- `vapasr/data/semcommit_dataset.py`: `check_sem_ids` 가 SEM 뒤 지연 확장 id 를 허용한다
- NEXT 가중은 한국어만 0.15 이고 나머지(독일어 포함)는 0.3 이라 trainer 변경은 없다.
- 확인:
  - CPU dry-run(영·한·독 일부 목록): 독일어 셋 생성, tokenizer 검사 32 행, 시퀀스 검사 64 항목 × δ6 개 불일치 0. δ4–10 도 같다.
  - 테스트(mxc): `tests/test_textnorm_de.py`, `tests/test_semcommit_delay_ext.py`, 기존 `test_single_turn_eval.py`·`test_textnorm.py` 통과
- 끝 패딩(항목 평균, 같은 표본 목록):

| 셋 | δ 1–8 | δ 4–10 |
|---|---:|---:|
| 영어 | 0.20 s | 0.36 s |
| 한국어 | 0.14 s | 0.25 s |
| 한국어 짧은 발화 | 0.34 s | 0.48 s |
| 독일어 | 0.52 s | 0.66 s |

## 3. 독일어 평가셋
- `experiments/eval_single_turn_asr.py prepare --yodas-de-dev-list` → mxc `VAPKT-data/data/evaluation/single-turn-yodas-de-dev-v1`(4,861 발화, 7.87 h, 오디오 867 MB)
- 참조는 정규화한 Qwen3-ASR-1.7B 전사라 silver 다(같은 교사를 증류한 모델에 유리하다). FLEURS·CoVoST2 독일어를 받으면 바꾼다.
- 채점: `vapasr/data/single_turn_eval.py` 에 독일어 WER(`score_de`)을 더했다.

## 4. 학습 실행
| job | 모델 | 시작점 | 설정 | 상태 |
|---|---|---|---|---|
| 82730 | 1.7B | q17-s1m-r3 | 1 epoch 이어서 | 모델 로드 직후 사용자 취소. `<DELAY_9/10>` 이 실제 1.7B 에 붙는 것까지 확인 |
| **82738** | **0.6B** | q06-s0 | 3 epoch | 진행 중 |

- 82738 설정(`slurm/submit-q06-s1m3-d10.sh`):
  - 영·한·독, δ4–10, 인코더 [56,0] → [56,3] 고정 학습
  - SpecAugment·속도 변형·잡음, final 30 %, apex 2 노드(hpc-42·43)
- 데이터: 영어 456,893 스트림 2,143 h · 한국어 2,956,210 스트림 4,990 h · 한국어 짧은 발화 945,016 스트림 1,122 h · 독일어 1,315,348 스트림 2,125 h
- 진행:
  - epoch 당 5,391 step(3 epoch 약 16,170 step)
  - step 1,370(18:36)에서 손실 0.37, top1 0.87, 1.7 s/step → 10/9 01:40 쯤 완료 예상
- 경로 변수를 제출 셸에 남기지 않도록 래퍼 스크립트로 제출한다(`slurm/submit-q06-s1m3-d10.sh`, 1.7B 는 `submit-q17-s1m3-d10.sh`).

## 남은 일
- 82738 완료 뒤 평가: 영어·한국어(`single-turn-vpaa-kspon-v1`)·독일어(`single-turn-yodas-de-dev-v1`), δ4·6·8·10 + final. 비교 대상은 q06-s1·q17-s1m-r3.
  - 완료(2026-10-09): 학습은 82844 로 이어서 끝났다. 결과는 [[output-stage1-q06-d10-eval-20261009]] 에 있다.
- 다음 단계: 화자·언어 prefix(`<SPK_A><EN><SPK_B><DE>`, A/B 무작위), 언어 토큰 추가, 두 언어 혼합·2 화자 스트림([[decision-stage1-lookahead-320-1040ms]])
- `semcommit_train.py`·`semcommit_dataset.py` 의 변경은 Codex 미커밋 헝크와 같은 파일이라 커밋 방식이 미정이다(mxc 에는 배포됨).
