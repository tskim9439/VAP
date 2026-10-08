---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-11-08
created: 2026-10-08
updated: 2026-10-08
summary: AST 학습 데이터 — 전체 ASR 전사를 Qwen3.8-27B 로 단조(원문 어순) 번역만 만들어 쓰고, 번역 단어 시각은 stable prefix 로 정함, prefix 는 화자별 언어 → 화자별 목표 언어
sources:
  - '[[output-monotonic-ast-translation-methods-20261008]]'
  - https://arxiv.org/abs/2609.05799
related:
  - '[[decision-stage1-lookahead-320-1040ms]]'
  - '[[output-stage1-trilingual-d10-prep-20261008]]'
  - '[[decision-cst-paper-v1-scope]]'
---

# AST 학습 데이터: 단조 번역만, Qwen3.8-27B, stable prefix 시각 (2026-10-08)

## 맥락
- 기존 영·한·독 ASR 전사를 번역해 동시 음성번역(AST) 학습 데이터를 만든다. 원문 어순에 맞춘(단조) 번역으로 지연을 줄이는 계획이다([[output-monotonic-ast-translation-methods-20261008]]).
- 사용자가 계획 보고서를 보고 열린 질문 다섯 개에 답했다. 그리고 stable prefix 논문(Hoang·Axelrod·Post, arXiv 2609.05799)을 참고해 적용하라고 했다.

## 결정 (사용자, 2026-10-08)
1. **생성·판정 LLM**: Qwen3.8-27B 하나만 쓴다(속도 우선). 교차 판정은 하지 않는다.
2. **파일럿**: 방향별 50 건.
3. **규모**: 방향별로 **전체 ASR 학습 전사**를 번역한다(영→독·영→한·독→영·한→영).
4. **학습에는 단조 번역만** 쓴다. 일반 번역은 섞지 않는다(모델이 헷갈리지 않게). 일반 번역은 파일럿 비교와 내용 관문의 참고로만 만든다.
5. **출력 형식(prefix)**:
   - 시퀀스 앞 prefix 에 화자별 원문 언어를, 이어서 화자별 목표 언어를 둔다.
   - 형식은 Qwen3-ASR 입력에 최대한 자연스럽게 맞춘다.
   - 제안형(구현 때 확정): assistant 머리를 `language <SPK_A>English<SPK_B>German` 다음 줄 `translate <SPK_A>German<SPK_B>English<asr_text><DELAY_d>` 로 한다.
     - Qwen3-ASR 원 형식 `language English<asr_text>` 의 `language` 줄을 그대로 쓴다.
     - 화자는 이미 있는 `<SPK_A>`/`<SPK_B>` 로, 언어는 Qwen 이 아는 이름 그대로 쓴다. 그래서 `<EN>` 같은 새 언어 토큰은 필요 없다.
     - 화자가 한 명이면 한 쌍만 쓰고, ASR 만 할 때는 `translate` 줄을 뺀다. A/B 는 항목마다 무작위로 정한다([[decision-stage1-lookahead-320-1040ms]]).

## stable prefix 적용 (arXiv 2609.05799)
- 논문: 문장 전체 번역 f 와 원문 prefix 마다의 번역 pᵢ 를 만든다. mᵢ = max_{j≤i} lcp(pⱼ, f) 까지를 "안정된 prefix" 로 확정한다(영→독은 단어 단위, 영→일·중은 글자 단위). 이 데이터로 Qwen3-8B 를 미세조정해 지연-품질 균형이 좋아졌다(영→독·일·중; FLEURS·WMT24++·CoVoST2).
- 우리 적용:
  - f = 단조 번역으로 둔다. 원문 단어 prefix 마다 같은 단조 방식으로 부분 번역을 만든다.
  - 목표 단위(영·독은 단어, 한국어는 글자)가 확정되는 첫 원문 단어의 끝 시각(강제 정렬 시각)을 그 번역 단어의 방출 시각으로 쓴다.
  - 그래서 별도 단어 정렬기가 필요 없다(mxc 에는 쓸 만한 정렬 모델이 granite 임베딩뿐이다). 방출 시각은 곧 스트리밍 학습 타깃 시각(⌊t/80 ms⌋ + δ)이 된다.
  - 후보 비교(파일럿)는 이 시각으로 잰 AL·첫 방출·평균 방출로 한다.
- 비용 주의:
  - prefix 마다 LLM 을 부르므로 문장당 호출이 단어 수만큼 늘어난다.
  - 전체 규모에서는 prefix 간격(단어마다 → 0.3–0.5 s 마다)과 vLLM 같은 빠른 추론 엔진이 필요하다. vLLM 은 현재 teacher env 에 없고, 설치 파일은 사용자가 받아 올려야 한다.

## 재검토
- 파일럿(방향별 50 건) 결과 뒤: 관문 통과율, AL 감소, 사용자 검토를 보고 프롬프트와 규모 계획을 확정한다.
