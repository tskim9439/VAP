---
type: output
status: active
created: 2026-10-05
updated: 2026-10-05
summary: TAXI 정답 분할 오프라인 상한(예비) — Qwen3-ASR WER 영 13.78·독 7.33 %, 정답 전사→Qwen3.8-27B 영→독 BLEU 48.31·독→영 30.60, ASR 경유 42.66·28.11
sources:
  - '[[2026-10-05-cst-taxi-offline-baselines-mxc]]'
related:
  - '[[cst-bench]]'
  - '[[task-cst-strong-baselines]]'
  - '[[output-paper-draft-cst-s2tt-icml-20261005]]'
---

# TAXI 오프라인 정답 분할 베이스라인 (예비, 2026-10-05)

## 설정
- 데이터: CST-Bench-TAXI natural(86 세션, 640 턴: 독일어 349 · 영어 291). 세션 mono 혼합에서 정답 턴 구간을 잘라 넣었다.
- 시스템(턴마다 독립 번역, 대화 이력 없음):
  - 정답 전사 → Qwen3.8-27B 번역
  - Qwen3-ASR-1.7B(언어 지정) → Qwen3.8-27B 번역
  - Whisper-large-v3 직접 번역(독→영만)
- 프롬프트: "Translate this {src} utterance from a phone conversation into {tgt}. Output only the {tgt} translation." 탐욕 디코딩, 최대 160 토큰.
- 채점: `cstbench eval`, 소문자·구두점 제거·숫자 풀어쓰기(TAXI 참조 표기에 맞춤), 턴 ID 로 바로 대응(정답 분할 모드).
- 원자료: `raw/sources/experiments/2026-10-05-cst-taxi-offline-baselines-mxc/`

## 결과

| 시스템 | 영→독 BLEU / chrF | 독→영 BLEU / chrF |
|---|---|---|
| 정답 전사 → Qwen3.8-27B | 48.31 / 71.72 | 30.60 / 58.05 |
| Qwen3-ASR-1.7B → Qwen3.8-27B | 42.66 / 67.97 | 28.11 / 56.61 |
| Whisper-large-v3 직접 번역 | – | 30.69 / 56.67 |

- Qwen3-ASR-1.7B WER(코퍼스 단위): 영어 13.78 %, 독일어 7.33 %. 턴 평균으로는 영어 13.9 %, 독일어 10.05 %.
- 지연(이상적 시계): StreamLAAL = 원천 턴 평균 길이(영어 턴 4.62 s, 독일어 턴 3.09 s), 턴 종료 후 지연 0. 계산 시간을 더하면 0.1–0.2 s(배치 평균이라 근사).

## 해석
- 8 kHz 전화 음성이라 ASR 오류가 크다. 그 영향으로 ASR 경유 번역은 영→독에서 BLEU 5.7, 독→영에서 2.5 떨어진다.
- 독→영은 정답 전사를 써도 BLEU 30 대다. TAXI 참조 번역의 문체(직역·구두점 없음)와 짧은 턴의 영향으로 보인다. COMET 이 들어오면 다시 본다.
- Whisper 직접 번역(독→영)이 캐스케이드와 BLEU 는 비슷하고 chrF 는 같다.

## 주의
- 예비 수치다. 최종 프로토콜(COMET, 세션 bootstrap CI, 배치 없는 계산 시간)로 다시 채점해야 한다.
- 오프라인 시스템은 턴이 끝난 뒤 번역하고 끝점도 정답을 받으므로, 실제 사용에서 도달할 수 없는 품질 상한이다.

## 다음
- COMET·세션 bootstrap 을 채점기에 추가하고 재채점([[task-cst-bench-v0-metrics]])
- SeamlessStreaming, 동시 캐스케이드, 실제 조건 래퍼(VAD + 언어 판별)([[task-cst-strong-baselines]])
- 논문 Table 3·5 에 예비 수치로 반영됨([[output-paper-draft-cst-s2tt-icml-20261005]])
