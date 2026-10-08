---
type: output
status: active
created: 2026-10-08
updated: 2026-10-08
summary: YODAS-Granary 독일어 전사 점검·전처리 — 원 전사↔Qwen WER ≤ 3 % 선별, 본 처리(1–30 s) 2,045 h 채택, 30–40 s 추가 처리 준비
sources:
  - '[[2026-10-08-yodas-de-transcript-check]]'
  - '[[2026-10-08-yodas-de-align-smoke]]'
  - '[[2026-10-08-yodas-de-align-run]]'
related:
  - '[[decision-yodas-de-asr-selection]]'
  - '[[output-eval-set-plan-asr-simulst-20261007]]'
  - '[[decision-eval-train-decontamination]]'
---

# YODAS-Granary 독일어 전사 품질 점검 (2026-10-08)

## 질문
- 사용자가 독일어 실시간 ASR 학습용으로 YODAS-Granary 독일어를 mxc 에 받았다(`VAPKT-data/data/corpora/data/yodas_granary`, 2,966 parquet, 원본 목록과 일치, CC-BY-3.0).
- 전처리 원안: Whisper-large-v3·Qwen3-ASR-1.7B 로 전사 → 원 전사와 두 모델이 3 % 이내면 원 전사, 아니면 두 모델끼리 3 % 이내면 Qwen 전사, 둘 다 아니면 제외 → Qwen3-ForcedAligner 로 시각.
- Granary 원 전사가 Whisper-large-v3 의사 라벨(+ Qwen2.5-7B 구두점 복원, Koluguri et al., Interspeech 2025)임을 확인한 뒤, 사용자가 "소규모로 Qwen 전사를 뽑아 품질을 보고, 충분하면 ASR 필터를 빼자"고 했다.

## 데이터 규모

| 묶음 | 발화 | 시간 | 길이 중앙값 |
|---|---:|---:|---:|
| `asr_only`(전사만) | 415,260 | 496 h | 약 1.9 s |
| `ast`(전사 + 영어 기계 번역) | 3,335,156 | 7,429 h | 약 4.7–5.7 s |
| 합계 | 약 375 만 | 약 7,925 h | 최대 40 s |

## 방법
- 표본 1,983 발화(3.45 h): 하위 묶음 4 개 × `asr_only`/`ast` 마다 parquet 10 개 × 25 발화.
- Qwen3-ASR-1.7B, Whisper-large-v2, Whisper-large-v3 를 같은 표본에 돌렸다(언어 독일어, greedy).
- 비교 전용 독일어 정규화: 소문자, 구두점 제거, 숫자→단어(num2words), ß→ss, 하이픈 분리, 간투사 제거. 복합어 띄어쓰기 차이를 없애려고 공백을 뺀 문자 오류율(CER)도 쟀다.

## 결과

교사 간 일치(공백 제외 CER):

| 비교 | 중앙값 | ≤ 2 % | ≤ 5 % |
|---|---:|---:|---:|
| Granary ↔ Whisper-v3 재실행 | **0.000** | 69 % | 74 % |
| Granary ↔ Whisper-v2 | 0.027 | 48 % | 56 % |
| Granary ↔ Qwen | 0.044 | 43 % | 52 % |
| Qwen ↔ Whisper-v2 | 0.061 | 39 % | 47 % |

- **Granary 원 전사는 Whisper-v3 결과 그대로다**(재실행과 중앙값 0). 따라서 원안의 "원 전사 ↔ Whisper-v3" 비교는 독립 검증이 아니다.
- 그래도 v3 재실행과 26 % 는 5 % 넘게 다르다. 불일치 사례를 읽어 보면 Granary 전사가 **말 그대로가 아니다**:
  - 반복·말 고침·간투사가 지워져 있다
  - 절이 통째로 빠지거나 앞부분이 빠진 경우가 있다
  - 오디오와 텍스트가 어긋나므로 실시간 ASR·강제 정렬 학습에 그대로 쓰면 안 된다
- `asr_only` 의 긴 발화는 원 전사 자체가 깨진 경우가 많았다(소문자·무구두점·내용 붕괴). Qwen 이 더 그럴듯했다.
- Whisper 특유의 환각 문구("Untertitel …", "Danke fürs Zuschauen")는 1,983 개 중 5 개로 드물었다(Granary 환각 필터가 작동).

선별 규칙별 예상 유지 시간(전체 7,925 h 로 환산):

| 규칙 | WER ≤ 3 % | WER ≤ 5 % | CER ≤ 3 % | CER ≤ 5 % |
|---|---:|---:|---:|---:|
| 원 전사 ↔ Qwen 일치만 | 약 2,411 h | 약 3,305 h | 약 4,041 h | 약 4,869 h |
| 원안(원 전사가 Qwen·Whisper-v2 와 일치 → 원 전사, 아니면 Qwen↔Whisper-v2 일치 → Qwen) | 약 2,179 h | 약 3,007 h | 약 3,755 h | 약 4,636 h |

- `asr_only` 는 시간 기준 13–25 % 만 남고, `ast` 는 32–64 % 남는다.
- 교사끼리만 일치해 구제되는 비율(원안의 3 번 규칙)은 표본의 4–6 % 로 작다.
- 처리 속도(H200 1 장): Qwen 은 실시간 대비 약 94 배, Whisper 는 약 80 배 → 전량이면 Qwen 약 84 GPU 시간, Whisper 모델당 약 100 GPU 시간.

## 결론
- **ASR 필터는 빼면 안 된다.** 원 전사만 쓰면 시간 기준 절반 가까이가 오디오와 어긋난 텍스트로 들어간다.
- **Whisper-v3 재실행은 뺀다.** 원 전사와 같은 모델이라 정보가 없다.
- Whisper-v2 를 세 번째 표로 더하면 유지 시간이 약 10 % 줄고 GPU 약 100 시간이 더 든다. 데이터가 충분하므로 필수는 아니다.
- WER ≤ 3 % 로 원 전사 ↔ Qwen 일치만 써도 약 2,400 h 가 남아, 영·한 선별 데이터(약 2,360 h)와 비슷한 규모다.

## 결정 (사용자 동의 2026-10-08)
- 권장안대로 정했다: [[decision-yodas-de-asr-selection]]
  - 판정은 원 전사 ↔ Qwen 일치만(Whisper 재전사 없음)
  - 문턱 WER ≤ 3 %
  - 학습 타깃은 Qwen 원출력

## 전처리 파이프라인 (2026-10-08 구현·시험)
1. **목록·사전 점검**(`experiments/yodas_de_prepare.py`, CPU, 완료): 출력 mxc `VAPKT-data/data/yodas-de-v1/`

| 제외 사유 | 발화 | 시간 |
|---|---:|---:|
| 30 s 초과 | 143,916 | 1,451.9 h |
| 3-gram 반복 | 12,402 | 80.2 h |
| 중복 구간·같은 전사 | 64,592 | 37.6 h |
| 환각 문구 | 10,404 | 34.3 h |
| 1 s 미만 | 134,598 | 27.5 h |
| 발화 속도 이상 | 9,023 | 14.6 h |
| Europarl-ST 독일어 8-gram 겹침 | 2,086 | 4.0 h |

   - 남은 후보: train 3,391,377 발화(6,315.7 h), dev(영상 200 개) 12,692 발화(23.0 h)
2. **전사·선별·정렬·묶기**(`experiments/yodas_de_align.py`, GPU, parquet 1 개 = shard 1 개):
   - Qwen3-ASR-1.7B 전사 → 원 전사와 WER ≤ 3 % 만 남김
   - Qwen3-ForcedAligner 시각. 클립 끝을 넘는 시각은 80 ms 이하만 잘라 맞춘다(영·한과 같은 규칙)
   - 영·한 speechlm 후보와 같은 검사로 words 행을 만들고(set `yodas-de`, 언어 `German`), 남은 오디오는 shard 별 무압축 tar 에 담는다
3. **시험 실행**(shard 6 개, 후보 104 발화; `raw/sources/experiments/2026-10-08-yodas-de-align-smoke/`):
   - 첫 시험에서 정렬 실패 8 건(남긴 48 건 중 17 %)이 모두 정렬기 끝 시각이 클립 길이를 넘은 경우였다
   - 80 ms 보정을 넣은 뒤 7 건이 구제됐다. 남은 1 건은 300 ms 넘침이라 제외했다
   - 출력 검증 위반은 0 건이었다: tar 색인, 오디오 길이, 토큰 시각, tokenizer 재분할을 보고 `SemCommitDataset` 로더로 직접 읽었다
4. **본 처리 전량 실행**(1–30 s, job 82542, 사용자 제출 8 노드 hpc): 13:08–14:11(약 63 분), shard 2,966 개 전부 완료, shard 오류 0

| 상태 | 발화 | 시간 |
|---|---:|---:|
| 채택(정렬 완료) | 1,340,112 | **2,045.4 h** |
| 불일치(WER > 3 %) | 2,029,526 | 4,224.5 h |
| 정렬 실패 | 34,406 | 68.7 h |

   - words 행 train 1,335,209, dev 4,903. 출력 약 242 GB(`VAPKT-data/data/yodas-de-v1/results/`)
   - 정렬 실패 사유: 토큰 시각 역전 19,065, 클립 끝 넘침 80–320 ms 15,072, 320 ms 초과 269
   - 무작위 shard 12 개(5,445 행) 검증 위반 0, 로더 bad 0. 80 ms 보정 행은 18 %
5. **30–40 s 처리**(long pass, 사용자 요청 2026-10-08): 사전 점검에서 길이만 걸린 138,258 발화(1,394.5 h)
   - 시험 4 shard(77 발화): 채택 14(시간 18 %), 정렬 실패 0, 검증 위반 0
   - 제출: `bash slurm/submit-yodas-de-align.sh --long --nodes 4 --partition hpc`. 결과는 같은 `results/` 아래 `<shard>-dur_long` 폴더, 요약 `align-summary-dur_long.json`

## 남은 일
- 30–40 s 처리 실행(사용자 제출) 뒤 채택 시간 확인
- 학습 연결: `semcommit_train.py` 의 ASR 전용 셋 구성에 독일어 셋 추가(현재 en/ko 만 만든다). 40 s 항목이 들어오므로 묶음 토큰 예산 확인
- FLEURS·CoVoST2 독일어를 받으면 평가셋 겹침 검사 재실행
- 정렬 실패 중 토큰 시각 역전(19,065 발화)의 원인 확인(선택)
