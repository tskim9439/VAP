---
author_id: tskim
created: 2026-10-11
source_type: source-drop
---

# q06-s2-ast(final) FLEURS 번역 평가, 영어 무음 출력 진단, 교사 상한 (mxc 호스트, 2026-10-11)

## 1. FLEURS 평가 (`fleurs-model-metrics.txt`)

- 체크포인트: `/soundai/Model/VAPASR/semcommit-q06-s2-ast/final` (job 83866 완료 07:56, parity 통과)
- 평가 방법:
  - 실행: 평가 대기열 `results/evalq-q06-s2-ast/driver.sh`. `eval_single_turn_asr.py run` 을 shard 단위로 돌리고, GPU 0·1·2·5 가 하나씩 가져간다.
  - 설정: δ4·6·8·10 + final, batch 64, --verify
- FLEURS 4 방향 2,382 행 → `results/ast-fleurs-q06-s2-ast/`
- 음량 보정 변형 1,138 행 → `results/ast-fleurs-en-peak05-q06-s2-ast/`
  - FLEURS 영어 음성(en-de·en-ko 행)만 최대 진폭 0.5 로 맞췄다(`make_peak05.py`).
  - 새 파일: `audio/fleurs-peak05/en_us/` 와 `manifest-max30s-fleurs-en-peak05.jsonl`. 원본 세트는 그대로다.
- 이 파일은 summarize 전에 예측 파일을 읽어 계산한 값이다(`peek_eval.py`, 읽기만 함).
  - 지표는 `vapasr/data/ast_eval.translation_report` 와 같다.

## 2. 영어 원문에서 출력이 통째로 비는 문제 (`fleurs-silent-counts.txt`, probe_*.py)

- 증상:
  - 영→독·영→한 스트리밍에서 전사도 번역도 없이 끝나는 발화가 있다. 영→독 δ4 기준 156/641 이다.
  - 독→영·한→영은 0 이고, Kspon 받아쓰기는 1–2/3,000 이다.
  - 짧을수록 심하다: 5 s 미만 16/28, 5–10 s 104/334, 10–15 s 35/228, 15 s 이상 1/51.
- 원인: FLEURS 영어 test 원본 음성 자체가 아주 작다(`probe_tar.py`).
  - 원본 tar(float wav) 앞쪽 300 개 중 64 % 가 최대 진폭 0.02 미만이다.
  - 비는 발화의 RMS 중앙값은 −63.5 dB 다. 독일어는 −29 dB, 한국어는 −44 dB 다.
- 확인한 것 (영→독 δ4 에서 빈 48 개, GPU 6):
  - 번역 없이 받아쓰기 prefix 만 줘도 40/48 이 빈다. SPK_B 로 바꾸면 46/48, 영→한 prefix 는 40/48 이다. → 번역 모드 문제가 아니다.
  - Stage 1(q06-s1m3-d10/final)도 δ4·6·10 에서 36·33·19/48 이 빈다. → 이번 학습에서 생긴 문제가 아니다.
  - s2 checkpoint-1000 은 19/48, checkpoint-7000 은 38/48 이 빈다.
  - 파형을 최대 진폭 0.1 또는 0.5 로 키우면 0/48 이 빈다. 전사와 번역이 모두 정상이다(`probe_gain.py`).

## 3. 교사 상한 (`teacher-ceiling-scores.txt`, `teacher_ceiling_gen.py`, `teacher_ceiling_score.py`)

- 방법: Qwen3.8-27B(vLLM 0.30, bf16, GPU 6, 메모리 60 %; GPU 6 에 다른 사용자의 1 GB 프로세스가 있었다)로 FLEURS 원문 텍스트를 번역했다. 조건은 넷이다:
  - mono-raw: v3 단조 프롬프트(학습 타깃을 만든 프롬프트), 정답 원문(대소문자·문장부호 있음)
  - mono-norm: v3 단조 프롬프트, 정규화된 정답 원문(우리 ASR 출력과 같은 형식)
  - free-raw: 일반 번역 프롬프트, 정답 원문
  - mono-asr: v3 단조 프롬프트, 우리 모델의 final 모드 전사(우리 ASR 오류 + 교사 번역)
- 규모와 속도: 9,528 프롬프트를 162.5 s 에 처리했다. prefix cache 적중은 92 %, 길이 초과는 1 건이다.
  - 출력 정리는 학습 데이터 생성과 같은 `clean()` 이다.
- `bleu-length-ratio.txt`: 조건별 BLEU 의 BP·길이비·1-gram/4-gram 정밀도

## 원 위치 (mxc)

- `results/ast-fleurs-teacher-ceiling-20261011/`
  - `translations.jsonl`(조건별 원 출력 포함), `scores.json`
  - 첫 실행 로그 `gen.log`: spawn 오류로 실패(main 보호가 없었음). 다시 실행한 로그는 `gen2.log`
