---
type: output
status: active
created: 2026-10-08
updated: 2026-10-08
summary: YODAS-Granary 독일어(7,925 h) 전사 점검 — 원 전사는 Whisper-v3 결과라 v3 재실행은 무의미, Qwen 과 일치는 시간 기준 절반 수준이라 ASR 필터는 필요
sources:
  - '[[2026-10-08-yodas-de-transcript-check]]'
related:
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

## 남은 결정 (사용자)
- 판정에 Whisper-v2 를 추가할지(원안 유지) / 원 전사 ↔ Qwen 일치만 쓸지
- 문턱(WER 3 % / 5 %, 또는 공백 무시 CER)
- 학습 타깃 표기: 일치한 경우에도 Granary(LLM 구두점 복원, 정리된 표기) / Qwen 원출력(말 그대로, 영·한과 같은 규약)
