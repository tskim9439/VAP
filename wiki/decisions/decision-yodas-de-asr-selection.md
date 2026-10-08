---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-11-08
created: 2026-10-08
updated: 2026-10-08
summary: YODAS-Granary 독일어 ASR 학습 데이터 — 원 전사↔Qwen3-ASR WER ≤ 3 % 만 채택, 학습 타깃은 Qwen 원출력, Whisper 재전사는 쓰지 않음
sources:
  - '[[output-yodas-granary-de-transcript-check-20261008]]'
  - '[[2026-10-08-yodas-de-align-smoke]]'
related:
  - '[[decision-eval-train-decontamination]]'
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
- **학습 타깃**: 일치한 발화도 **Qwen 원출력**을 쓴다. 영·한 데이터와 같은 규약(말 그대로 + Qwen 구두점)이고, 시각을 정렬한 텍스트와 같다. 원 전사는 `raw_text` 로 남긴다.
- **시각**: Qwen3-ForcedAligner-0.6B. 클립 끝을 넘는 시각은 영·한과 같은 규칙으로 처리한다(80 ms 이하는 잘라 맞춤, 그 이상 제외).
- **사전 제외**: 1 s 미만·30 s 초과, 빈 전사, 환각 문구, 3-gram 반복, 발화 속도 이상, 비라틴 문자, 중복 구간, 평가셋 8-gram 겹침([[decision-eval-train-decontamination]]).
- **dev**: 영상 200 개(영상 ID 해시 순)를 통째로 떼어 둔다.
- `<SEM_END>` 등 의미 단위 표시는 이번 단계에서 넣지 않는다.

## 결과와 영향
- 예상 유지량: 약 2,400 h(표본 기준 추정). 영·한 선별 데이터(약 2,360 h)와 비슷한 규모다.
- 30 s 초과 발화 1,452 h 는 지금 제외된다. 필요하면 정렬 후 잘라 쓰는 방안을 따로 검토한다.
- Whisper-v2 를 세 번째 표로 넣지 않아 GPU 약 100 시간을 아끼고, 유지량은 약 10 % 더 많다.
- 대가: 판정 모델이 Qwen 하나라, Qwen 과 원 전사(Whisper-v3)가 함께 틀린 경우는 걸러지지 않는다.

## 구현
- `experiments/yodas_de_prepare.py`(inventory, prefilter), `experiments/yodas_de_align.py`(전사·선별·정렬·words 행·오디오 tar), `slurm/yodas-de-align.sbatch`, `slurm/submit-yodas-de-align.sh`
- 출력: mxc `VAPKT-data/data/yodas-de-v1/results/<shard>/words-{train,dev}.jsonl`, `audio.tar`(+ 색인). set 이름 `yodas-de`, 언어 `German`
- 시험 실행·검증: `raw/sources/experiments/2026-10-08-yodas-de-align-smoke/`

## 재검토
- 2026-11-08 또는 독일어 학습 첫 결과가 나올 때. 독일어 평가(FLEURS·CoVoST2)에서 Granary 쪽 오류가 보이면 문턱을 다시 본다.
- FLEURS·CoVoST2 독일어를 받으면 평가셋 겹침 검사를 다시 돌린다.
