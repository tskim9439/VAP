---
type: output
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: 절단 실험으로 측정한 encoder별 실효 lookahead — Nemotron 80ms chunk는 ≤80ms, Qwen AuT는 기본 경로 비인과·블록 모드 평균 420ms
sources:
  - [[source-nemotron-3-5-asr-streaming]]
  - [[source-qwen3-asr]]
  - [[question-encoder-lookahead-and-causality]]
---

# Encoder causality·lookahead 감사 결과

원본: `raw/sources/experiments/2026-09-03-causality-audit.json`, 코드: `experiments/causality_audit.py`

## 질문

각 encoder 가 `h_t` 를 계산할 때 실제로 미래를 몇 ms 보는가? 이 값이 latency 회계
([[streaming-causality-and-latency-budget]])의 항이고, [[decision-asr-backbone]] 의 조건이다.

## 방법

프론트엔드 특징(fbank)을 전체 오디오(14.84 s, LibriSpeech 샘플)에서 한 번 계산한 뒤
**특징 단위로 절단**해 encoder 에 넣고, 전체 입력의 `h_full` 과 절단 입력의 `h_cut` 을
프레임별로 비교했다. 절단점 이전 프레임 중 값이 바뀐 프레임 수 × frame_ms = 실효 lookahead.
fp32, TF32 off, 상대 허용오차 1e-3. 절단점 3/6/9/12 s (Qwen 블록 모드는 경계 비정렬 3.5/6.25/9.75/12 s).

## 결과

| encoder | frame | lookahead ms (min / mean / max) | 비고 |
|---|---:|---:|---|
| CPC encoder (원 VAP) | 20 | 0 / 0 / 0 | 완전 causal — **대조군, 방법 검증** |
| VAP full model (probs) | 20 | 0 / 0 / 0 | transformer 포함 causal |
| Nemotron `[56,0]` 80 ms chunk | 80 | 80 / 80 / 80 | 마지막 1 프레임만 변화 (subsampling conv 경계) |
| Nemotron `[56,1]` 160 ms | 80 | 80 / 120 / 160 | |
| Nemotron `[56,3]` 320 ms | 80 | 160 / 240 / 320 | |
| Nemotron `[56,6]` 560 ms | 80 | 160 / 320 / 480 | |
| Nemotron `[56,13]` 1120 ms | 80 | 160 / 600 / 880 | |
| **Qwen AuT — sdpa 기본 경로** | 80 | **3120 / 7800 / 12480** | **절단 시 모든 프레임이 바뀜 = 발화 전체 양방향** |
| Qwen AuT — 1 s 블록 독립 forward | 80 | 0 / **420** / 800 | FA2 varlen 의미 재현. lookahead = 블록 끝까지 거리 |
| Qwen AuT — 2 s 블록 | 80 | 0 / 940 / 1840 | |

## 해석

### Nemotron — 문서와 일치, 80/160 ms chunk 만 후보

- 프론트엔드 `normalize=NA`(utterance 정규화 없음), `conv_context_size=[8,0]`(causal conv) 확인.
- chunk 내 위치에 따라 lookahead 가 달라지며 **최대치 ≈ chunk 크기** 다 (cache-aware chunked attention 의 예상 동작).
- `[56,0]` 에서 남는 80 ms 는 8× subsampling conv 의 오른쪽 경계 효과로, "한 chunk 가 완성돼야 프레임이 나온다" 는 의미와 같다.
- **결론**: turn-taking 용도로는 `[56,0]`(≤80 ms) 또는 `[56,1]`(≤160 ms) 만 쓴다.
  `[56,3]` 이상은 320 ms 관문에 걸린다. 이 chunk 에서의 ko-KR CER 이 실제 관심 수치다.

### Qwen AuT — 두 가지 문제

1. **transformers/sdpa 경로는 비인과다 (구현 버그).** `Qwen3ASRAudioEncoder.forward` 가
   `_prepare_attention_mask` 를 호출하지 않고 레이어에 `cu_seqlens` 만 넘긴다. sdpa 는
   `attention_mask=None` 으로 전체 발화 양방향 attention 을 수행하며 `n_window_infer` 는
   무효다 (800 vs 100 출력 비트 동일). flash-attn 설치 시에만 varlen 으로 블록 분할이 살아난다.
   → **이 경로에서 뽑은 특징을 VAP 학습에 쓰면 미래가 통째로 샌다.**
2. **의도된 streaming 동작도 turn-taking 에 부적합하다.** 블록(≥ conv chunk 1 s)은 서로
   attention 하지 않으므로 (a) 블록 끝까지 기다려야 하고 (b) **블록 간 좌측 context 도 없다**.
   1 s 블록에서 lookahead 평균 420 ms(0–800). 이는 [[source-turnbench]] 의 VAP p50 368 ms 보다
   구조적으로 느리다. 더 작은 블록은 `n_window=50`(1 s conv chunk)에 막힌다.

→ **[[decision-asr-backbone]] 의 재검토 조건이 Qwen 쪽에서 발동했다.**

## 추가 실험 — attention 마스크 복원/변형 (같은 날)

`experiments/qwen_aut_mask.py` 로 18개 레이어 forward 를 감싸 `cu_seqlens` 기반 마스크를 주입했다
(인코더 forward 는 그대로). 원본: `raw/sources/experiments/2026-09-03-qwen-aut-mask-eval.json`.
WER 은 **단일 14.8 s 발화**에서 패치 전(sdpa, 마스크 없음) 전사 대비 — 정성적 신호로만 본다.

| 마스크 | lookahead mean / max (ms) | WER vs 패치 전 |
|---|---:|---:|
| none (현 sdpa 동작) | 8220 / 12480 | 0 |
| **block 1 s** (FA2 varlen 의도 복원) | 420 / 800 | 8.8 % |
| block 8 s (학습·추론 기본 의도) | 4060 / 6560 | 11.8 % |
| **chunked-causal 1 s** (chunk 내 양방향 + 이전 chunk 좌측 context) | 420 / 800 | **5.9 %** |
| chunked-causal 1 s, 좌측 7 블록 한정 | 420 / 800 | 5.9 % |
| chunked-causal 2 s | 940 / 1840 | 5.9 % |
| **causal (프레임 단위, 학습에 없던 마스크)** | **60 / 80** | 23.5 % |

해석:

1. **패치 검증.** `block 1 s` 의 절단점별 lookahead(560/320/800/0)가 per-block 독립 forward 와 정확히 일치.
2. **`chunked-causal` 이 `block` 보다 낫다.** 좌측 context 를 주면 lookahead 는 같고 WER 은 낮다.
   AuT 를 그대로 쓸 때의 최선은 chunked-causal 1 s 이지만, 여전히 평균 420 ms 로 관문 초과.
3. **프레임 causal 마스크에서도 단어 대부분이 살아남는다.** lookahead 가 conv 경계 80 ms 로 떨어지면서
   WER 23.5 % (고유명사·기능어 오류). 학습 때 본 적 없는 마스크임을 감안하면 **소규모 causal fine-tune 으로
   Nemotron 급 lookahead 를 얻을 가능성**이 있다. [[task-qwen-aut-causal-adaptation]] 의 핵심 실험이 된다.
4. `block 8 s` 가 패치 전과 11.8 % 다르다 — 즉 **현재 sdpa 경로의 출력은 FA2 로 배포된 모델의 출력과도 다르다.**
   재현성 관점에서 transformers 백엔드 결과를 인용할 때 주의.

## 후속

- Nemotron: Stage 1 probing 에 `[56,0]`/`[56,1]` 로 진행. → [[task-stage1-encoder-probing]]
- Qwen: 마스크 주입은 완료(`experiments/qwen_aut_mask.py`). 다음은 (a) 제대로 된 평가셋(LibriSpeech test-clean
  일부 + 한국어)에서 마스크별 WER/CER, (b) **causal 마스크 소규모 fine-tune** 으로 23.5 % 를 얼마나 회복하는지,
  (c) `n_window` 25/12 로 conv chunk 축소 — 적응 연구. Stage 1 에 AuT 를 넣는다면
  **1 s 블록 모드 + lookahead 0–800 ms 를 표에 명시**해야 한다. → [[task-qwen-aut-causal-adaptation]]
- 논문의 latency 표에는 이 측정값을 그대로 싣는다: `latency = frame/2 + lookahead + compute`.
