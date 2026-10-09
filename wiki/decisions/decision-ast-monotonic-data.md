---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-11-08
created: 2026-10-08
updated: 2026-10-09
summary: AST 학습 데이터 — 전체 ASR 전사를 Qwen3.8-27B 로 단조(원문 어순) 번역만 만들어 쓰고, 번역 단어 시각은 stable prefix 로 정함, prefix 는 화자별 언어 → 화자별 목표 언어
sources:
  - '[[output-monotonic-ast-translation-methods-20261008]]'
  - '[[output-ast-mono-generation-20261009]]'
  - '[[2026-10-09-ast-mono-speedup]]'
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

## 결정 (사용자, 2026-10-09)
6. **4 방향을 한 번에** 생성한다.
7. **stable prefix 는 원문 단어마다** 만든다(가장 정확). 0.3–0.5 s 간격으로 줄이지 않는다.
- 구현과 비용: [[output-ast-mono-generation-20261009]]
  - Stage 1 학습 행 568 만 개 → 번역 단위 733 만 개, prefix 호출 약 8,700 만 번
  - 약 675 GPU 시간
  - 일반 번역은 만들지 않는다.
- 2026-10-09 가속 검토(사용자: 노드 2 개, 메모리 효율 검토 요청):
  - 프롬프트를 **v3** 로 바꾼다. v2 지시를 방향별 고정 시스템 메시지로 옮기고 예시를 늘려 공통 앞부분을 784 토큰 블록 너머로 맞춘 것이다(prefix cache).
  - bf16 으로, 메모리 0.95·동시 1,024 로 돌린다. v2 대비 2.98 배 빠르고 관문 통과는 +2 %p 다. fp8 은 1.3 %p 낮아 쓰지 않는다.
  - 약 226 GPU 시간, 2 노드 약 14 시간

## 결정 (사용자, 2026-10-09, 다음 단계)
8. 다음 단계 학습은 **0.6B q06-s1m3-d10/final** 에서 시작한다([[decision-stage1-lookahead-320-1040ms]]).
9. 번역 모드에서 모델은 **원문 전사와 번역을 함께** 출력한다(선택지 (b)). 번역문만 내는 (a) 는 쓰지 않는다.
   - 출력 시퀀스 규칙(사용자 확정, 2026-10-09):
     1. **prefix:**
        - 두 줄이다. 첫 줄은 `language <SPK_A>English<SPK_B>German`, 둘째 줄은 `translate <SPK_A>German<SPK_B>English<asr_text><DELAY_d>` 다.
        - 화자가 한 명이면 한 쌍만 쓴다. ASR 만 할 때는 `translate` 줄을 뺀다.
        - A/B 는 항목마다 무작위로 정한다.
     2. **배치 시각:** 80 ms 청크 규칙(k = ⌊t/80 ms⌋ + δ)을 그대로 쓴다.
        - 전사 토큰은 원문 단어 끝 시각에 나온다.
        - 번역 토큰은 stable prefix 확정 시각(어떤 원문 단어의 끝 시각)에 나온다. 그래서 그 원문 단어 바로 뒤, 같은 청크에 붙는다.
        - δ 는 둘에 같다.
     3. **구분 표시:** 새 토큰 `<TRANS>`(151726) 하나를 더한다.
        - `<SPK_x>` 는 x 의 전사 구간을, `<TRANS>` 는 번역 구간을 연다. 구간이 바뀔 때만 넣는다.
        - 다른 화자의 번역이면 `<SPK_y><TRANS>` 다.
        - 예(δ4): 청크 77 `<SPK_A> it <TRANS> Es`, 78 `<SPK_A> is`, 81 `about <TRANS> geht um`, 86 `<SPK_A> shared`, 92 `values <TRANS> gemeinsame Werte`(각 청크 끝 `<NEXT_AUDIO>`)
        - 정정(2026-10-09): 확정 때 보인 예시에는 81·92 청크에도 `<SPK_A>` 가 있었다. 그러나 바로 앞 구간이 이미 전사라 규칙상 태그가 없다. 구현과 테스트(`tests/test_ast_dataset.py`)는 규칙을 따른다.
     4. **텍스트:** 전사는 정규형(소문자·무구두점)이다. 번역은 생성된 표기형(구두점·대소문자)이다.

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
  - 2026-10-09:
    - vLLM 0.30.0 을 설치했다(`scripts/setup-vllm-env.sh`, 별도 env, 노드에서는 tar 를 풀어 쓴다).
    - HF 보다 약 7.5 배 빨랐고 품질은 같았다.
    - 단조 번역만 만들면 전체 약 500 GPU 시간으로 추정된다([[output-monotonic-ast-translation-methods-20261008]] 속도 항목).
    - flash-linear-attention 은 HF 경로를 빠르게 하지 못해 쓰지 않는다(`teacher-fla` env 는 남겨 둠).

## 재검토
- 파일럿(방향별 50 건) 결과 뒤: 관문 통과율, AL 감소, 사용자 검토를 보고 프롬프트와 규모 계획을 확정한다.
