---
type: question
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: Nemotron과 Qwen3 AuT의 실효 lookahead는 몇 ms이며 VAP latency 비교가 성립하는가
sources:
  - [[source-qwen3-asr]]
  - [[source-nemotron-3-5-asr-streaming]]
---

# 두 backbone 의 실효 lookahead 는 얼마인가?

## 질문

streaming 모드에서 `h_t` 를 계산할 때 실제로 미래 오디오를 몇 ms 나 보는가?

## 왜 중요한가

**이 연구 전체의 선행 조건이다.** → [[streaming-causality-and-latency-budget]]

- lookahead 를 latency 에 합산하지 않으면 [[source-turnbench]] VAP(368 ms)와의
  비교가 무효다.
- offline 모드로 특징을 뽑으면 미래가 새어 들어가 **모든 VAP 결과가 무의미** 해진다.
- lookahead > 320 ms 면 [[decision-asr-backbone]] 을 뒤집어야 한다.

## 지금까지 아는 것

- [[source-qwen3-asr]]: 기술 보고서가 **causality 를 명시하지 않는다.**
  "dynamic flash attention window 1–8s" 는 window 크기이지 right-context 0 을
  뜻하지 않는다.
- [[source-nemotron-3-5-asr-streaming]]: cache-aware streaming 이고 chunk 가
  80/160/320/560/1120 ms. chunk 크기가 곧 알고리즘 지연의 하한이지만
  **추가 right-context 가 있는지는 설정을 봐야 안다.**

## 2026-09-03 진전 — Nemotron 은 문서로 절반 답이 나왔다

[[source-nemotron-3-5-asr-streaming]] 모델 카드: chunk = `att_context_size=[56, N]`,
우측 context N frame × 80 ms 가 명시적 lookahead. **80 ms chunk = 우측 0 = attention
수준 strictly causal.** 남은 확인은 conv subsampling 의 암묵적 lookahead 뿐이다.

Qwen3 AuT 는 여전히 미확인. 게다가 [[source-qwen3-asr]] 의 streaming 은 vLLM 전용이라
hidden state 를 streaming 으로 뽑으려면 AuT 를 직접 chunk 호출해야 한다.

## 답 (2026-09-03) → [[output-encoder-causality-audit]]

| encoder | lookahead (max) |
|---|---:|
| CPC / 원 VAP | 0 ms |
| Nemotron `[56,0]` / `[56,1]` | 80 / 160 ms |
| Nemotron `[56,3]` 이상 | ≥ 320 ms — 부적합 |
| Qwen AuT sdpa 기본 | 발화 전체 (비인과, 마스크 미적용 버그) |
| Qwen AuT 1 s 블록 | 0–800 ms, 평균 420 |

Nemotron 은 80/160 ms chunk 로 사용 가능. Qwen 은 [[decision-asr-backbone]] 조건 위반 →
[[task-qwen-aut-causal-adaptation]]. 남은 미확인 항목 없음.

## 답을 얻는 방법 (기록용)

절단 실험. 전체 오디오로 뽑은 `h_t` 와 시점 `t+Δ` 에서 자른 오디오로 뽑은 `h_t` 를
비교해, 일치하기 시작하는 최소 `Δ` 가 실효 lookahead 다. 코드 몇 줄이면 된다.

→ [[task-audit-encoder-causality-lookahead]]
