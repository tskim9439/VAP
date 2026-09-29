---
type: output
status: active
created: 2026-09-29
updated: 2026-09-29
summary: <SEM_END> 트리거 2-pass 재디코드 실험 계획(리뷰 반영, commit 지연 지표 5103556 사용). Qwen3-ASR 2차, VAD·고정창·오라클 대조, 예산 약 30 GPU-h
sources:
  - '[[output-semcommit-recipe-v0.3]]'
  - '[[output-semcommit-gold-v1]]'
  - '[[output-speechlm-semcommit-approval-audit-20260925]]'
  - '[[output-stage2-d2-final-eval]]'
  - '[[source-qwen3-asr]]'
  - '[[source-asr-output-style-probe]]'
  - '[[source-nemotron-3-5-asr-streaming]]'
  - https://arxiv.org/abs/2609.07549
related:
  - '[[decision-semcommit-turn-eot-scope]]'
  - '[[task-semcommit-main-labeling]]'
  - '[[streaming-causality-and-latency-budget]]'
  - '[[source-streaming-speech-llm-related-work]]'
raw_authors:
  - tskim
---

# T1 실험 계획: `<SEM_END>` 트리거 2-pass 재디코드

스트리밍 텍스트를 full-context 재디코드 결과로 바꾸는 실험의 계획이다.

- 작성일: 2026-09-29. 같은 날 리뷰 2 건(37 항목)을 반영해 고쳤다. 무엇을 왜 고쳤는지는 부록 B 에 있다.
- 상태: **계획 단계.** 코드와 학습은 아직 바꾸지 않았다. §10 의 파일은 모두 만들 예정인 것이다.
- 범위 정정(2026-09-29): 사용자 지시 "3번은 일단 제외하고 더 고민해보자"의 3번은 논문 인사이트 3번(합성 말더듬·자기 수정으로 영어 N 보강 = 영어 레이블 불균형)이다. 초판은 이를 T3(commit latency 작업)로 잘못 읽어 T3 를 뺐다.
  - T3 는 범위 안이다. 창 없는 지연 지표(commit·display·after_text, 가설 단어별 청크 `hyp.word_k`)는 커밋 `5103556` 으로 들어갔고, 이 계획의 1차 디코드·지연 지표는 그 판을 쓴다(§4.1, §6.2, §6.6, §10).
  - 단어 정렬이 없는 세트(AIHub 3 세트, ESB)의 단어 끝 기준 지연은 T3 와 무관하게 정렬이 없어서 근사로 남는다(§6.6).
- 코드 인용 규칙:
  - 추적 중인 파일의 `파일:줄` 은 커밋 `88220f7` 기준이다. `commit_metrics.py`·`semcommit_eval.py` 는 `5103556` 에서 줄이 밀렸으니 함께 적은 함수 이름으로 찾는다.
  - 저장소에 커밋되지 않은 파일은 **(작업 트리, 미커밋)** 으로 표시하고, 줄 번호는 2026-09-29 작업 트리 기준이다.
  - mxc 의 사실(경로, 규모, 배포 상태)은 2026-09-29 읽기 전용 조회 결과다. 조회 스크립트는 부록 A 에 있다.

---

## 0. 한 줄 요약

1차 스트리밍 모델(VapAsr semcommit, 80 ms 청크)이 `<SEM_END>` 를 내면, 직전 확정 이후의 오디오 구간을 full-context 디코더(주로 Qwen3-ASR-1.7B/0.6B)로 다시 디코드한다. 그리고 그 구간의 스트리밍 텍스트를 재디코드 결과로 바꾼다.

이 실험이 답하려는 질문은 세 가지다.

1. 이렇게 확정한 최종 전사는 스트리밍 단독보다 얼마나 정확한가? 오프라인 결과와의 격차를 얼마나 줄이는가?
2. 무음(VAD) endpoint 에 `<SEM_END>` 트리거를 더하면 확정(finalisation) 지연은 구성상 줄어든다. 그 대가로 정확도를 얼마나 잃는가?
3. 이득이 "우측 문맥을 더 본 효과"만으로 설명되지는 않는가?

---

## 1. 배경과 근거

### 1.1 외부 근거: Qwen-Audio-3.0-ASR

출처는 arXiv 2609.07549 이다(2026-09-29 arxiv.org/html 에서 확인).

- 방식은 "hybrid streaming and full-context recognition" 이다.
  - 스트리밍 모델이 잠정 전사를 계속 갱신한다.
  - **utterance-end 를 검출하면** full-context 대형 모델이 그 발화 또는 음성 구간 전체를 다시 평가한다.
  - 재평가 결과가 잠정 출력을 대체해 확정 전사가 된다.
  - endpoint 를 어떻게 검출하는지는 공개하지 않았다.
- Figure 11(사내 zh/en 산업 테스트셋, 400 ms algorithmic latency)의 수치:
  - zh CER 9.24, 비스트리밍 8.03(상대 +15.1 %)
  - en WER 13.38, 비스트리밍 11.78(상대 +13.6 %)
  - first-token < 200 ms, text-display < 300 ms
  - 청크 크기와 right-context 는 추론 때 따로 설정한다.
- 우리 설계와 다른 점:
  - 그들은 **endpoint(발화 끝)** 를 트리거로 쓰고 발화 전체를 재평가한다.
  - 우리는 발화 안에서도 나오는 **의미 단위 commit(`<SEM_END>`)** 을 트리거로 쓴다.
  - 그래서 우리는 발화 중간에서도 확정할 수 있다. 대신 재디코드 구간이 짧아져 문맥을 잃는다.
  - 관련 연구 비교는 [[source-streaming-speech-llm-related-work]] 에 있다.

### 1.2 내부 근거: 현재 수치와 재현 경로

**스트리밍 단독 정확도와 δ(우측 문맥)**

출처는 mxc `results/single-turn-35000-d2-d4-v1/summary.json` 과 `results/single-turn-35000-d6-v1/summary.json` 이다. 모델은 semcommit 이전의 hf-approved-qwen-v1-e10-n2/checkpoint-35000 이다.

| 세트 | 지표 | δ2 | δ4 | δ6 |
|---|---|---:|---:|---:|
| LibriSpeech test-clean | WER | 6.63 % | 4.74 % | 4.51 % |
| LibriSpeech test-other | WER | 14.19 % | 11.34 % | 10.88 % |
| Kspon eval_clean | CER | 13.18 % | 12.11 % | 12.16 % |
| Kspon eval_other | CER | 14.21 % | 12.71 % | 12.79 % |

- δ2 → δ4 에서는 이득이 크다. **δ4 → δ6 에서는 작다**(EN −0.2 / −0.5 %p, KO 는 +0.05 / +0.08 %p 로 사실상 0).
- 같은 모델의 δ8 single-turn run(`single-turn-35000-d8-v1`)은 완료되지 않았다. `cancelled.json` 에 사용자 요청으로 δ8 을 멈추고 δ6 을 대신 평가했다고 적혀 있다. 그래서 이 모델의 δ8 수치는 없다.
- 따라서 "우측 문맥만 늘린 효과"는 δ4 이후 작을 것으로 보인다. 그래도 평가 대상 모델에서 S0-δ8 을 직접 재서 대조한다(§12 S-3). 위 수치는 이전 모델의 것이라 그대로 옮기지 않는다.

**SEM commit 품질**

- 조건: rack4 소형 r5, δ4, 골드 v1 대조. 근거는 [[output-semcommit-recipe-v0.3]], `raw/sources/experiments/2026-09-24-semcommit-recipe-v0.3/scores/models-r5-*.json`.
- bias 0 에서 세트별 정밀도(P)·재현율(R):

  | 세트 | P | R | 비고 |
  |---|---:|---:|---|
  | ks-eval | 0.76 | 0.89 | |
  | ks-long | 0.61 | 0.91 | PCR_gold 0.35 |
  | ls-test | 0.84 | 0.71 | |
  | gs-test | 0.87 | 0.46 | |

- 맞힌 commit 이 단어 끝보다 늦는 정도는 p50 0.32–0.36 s, p90 약 0.40 s 이다.
- 이 수치는 골드 전체(dev·test 절반 합)에서 낸 것이다. 레시피 v0.3 의 교사 임계값은 골드 dev 절반에서 맞췄다([[output-semcommit-recipe-v0.3]]).

**골드 commit 밀도**

- 로컬 gold v1 기준([[output-semcommit-gold-v1]]) commit 당 단어 수: ls-test 14.0, gs-test 13.9, ks-eval 8.7.
- **ks-eval 은 300 발화에 COMMIT 이 247 개뿐이다.** 짧은 한국어 발화 상당수는 발화 안에 commit 자리가 없다(미완결 조각).
- 그래서 SEM 만으로는 모든 단어를 확정할 수 없다. **발화 끝 endpoint 와 결합해야 하며**, 실사용 후보는 S1+ 이다.
- 같은 이유로 짧은 발화 세트(R1) 전체에서는 SEM 트리거의 효과가 잘 드러나지 않는다. 주 판정은 장문(R2)과 "발화 중간 SEM" 부분집합에서 한다(§12).

**평가 대상 모델**

- 경로: `/soundai/Model/VAPASR/semcommit-v035-snap0929-d8`
- 설정:
  - init 은 hf-E2/final 이다. 구조는 Nemotron [56,0] 동결 → adapter → thinker 이고, thinker 구조는 Qwen3-ASR-0.6B 다.
  - thinker 가중치는 원본 Qwen3-ASR-0.6B 가 아니다. C2 → D2 → E2 학습에서 thinker 를 전체 미세조정(full FT)한 가중치를 이어받았다([[output-stage2-d2-final-eval]]).
  - delays [2,3,4,6,8], `<SEM_END>` 만 학습, turn_end 끔. 학습 범위 결정은 [[decision-semcommit-turn-eot-scope]] 참고.
- 5 epoch = 5,845 step 이다. 2026-09-29 13:30 무렵 checkpoint-4400 까지 저장됐다.
- **본 평가는 final 체크포인트로 한다.** 중간 체크포인트는 파이프라인 스모크에만 쓴다.

---

## 2. 가설

- **H1 (정확도 회수)**: S1+(δ4, Q17, prefix)의 최종 WER/CER 이 두 조건을 만족한다. 판정 세트는 R2 test 와 R1 "발화 중간 SEM" 부분집합이다(§12 S-1).
  - 같은 δ 의 스트리밍 단독(S0)보다 상대 10 % 이상 낮다.
  - 오프라인(S5)이 S0 보다 유의하게 나은 세트에서는, 그 격차의 50 % 이상을 회수한다. S5 가 S0 보다 낫다는 보장은 없다. 그래서 S5 ≥ S0 인 세트에서는 회수율을 정의하지 않는다(§12).
- **H2 (SEM 트리거의 정확도 비용)**: S1+ = SEM ∪ VAD 이므로, 확정 지연은 구성상 VAD 단독(S2)보다 길 수 없다(§6.2, 예외는 fallback 세그먼트의 늦은 단어뿐). 그래서 지연은 검정하지 않고 효과 크기로만 보고한다.
  - 검정하는 것은 정확도 비용이다. 추가 SEM 트리거가 구간을 더 잘게 나눠 생기는 최종 WER 상승이 S2 대비 상대 2 % 이내다.
  - 지연을 맞춘 비교는 S2 의 H 를 스윕해 보간한 점과 한다(§12 S-2).
- **H3 (문맥)**: 짧은 commit 구간을 고립시켜 재디코드하면 오히려 나빠질 수 있다. 정확도는 iso < text-context < prefix re-feed 순일 것이다.
- **H4 (commit 오류 비용)**: 조기 commit(bias 와 PCR 이 높은 쪽)이 많을수록 경계 근처 단어의 오류가 늘어 2-pass 이득이 준다. 골드 경계 오라클(S4b)과 S1 의 차가 commit 오류의 비용이다.
- **H5 (δ 대체)**: 2-pass 를 쓰면 1차 δ 를 δ2 로 줄여도 최종 정확도가 δ4/δ8 2-pass 와 비슷하다. 그러면 표시 지연은 줄고 확정 지연은 비슷하다.
- **H6 (번복 품질)**: 2차가 바꾼 단어 가운데 개선(오→정)이 악화(정→오)보다 3 배 이상 많다.
- **H0 (우측 문맥 대조)**: 이득이 우측 문맥이 늘어난 효과뿐이라면, S1+ Q17 의 WER 이 S0-δ8 과 유의하게 다르지 않다.
  - V8-dual 의 최종 텍스트는 δ8 가설 전체와 같다. 그래서 V8-dual 의 WER 은 **정의상 S0-δ8 과 같다**(§5.2).
  - 같은 트리거에서 V8-dual 의 algorithmic 확정 시각은 S1+ 이상이다. 즉 S0-δ8(= V8-dual)은 "확정 지연이 같거나 더 긴, δ8 우측 문맥 대조군"이다.
  - δ6 수치(§1.2)를 보면 δ4 이후 우측 문맥 이득은 작을 것으로 예상한다.

---

## 3. 비교 시스템

표기:

- δ: 1차 방출 지연(청크 수)
- b: SEM bias(`logits[<SEM_END>] += b`)
- θ: threshold 모드

1차 디코드는 모두 `experiments/semcommit_eval.py`(5103556 판; 디코드 규약은 88220f7 과 같다)의 규약을 따른다(greedy, sem-guard 켬, fp32, TF32 끔).

| 이름 | 트리거 | 2차 디코더 | 목적 |
|---|---|---|---|
| **S0(δ)** | 없음 | 없음(1차 텍스트가 곧 최종) | 기준선. δ ∈ {2, 4, 8}. 번복 0 |
| **S1(δ, dec, ctx)** | `<SEM_END>` 방출(+ 스트림 끝 EOS) | Q06 / Q17 | 본 제안(SEM 단독) |
| **S1+(δ, dec, ctx, H)** | `<SEM_END>` 와 무음 endpoint 중 먼저 오는 것 | 〃 | 실사용 후보. SEM 이 안 나오는 미완결 발화는 endpoint 로 확정 |
| **S2(H, dec, ctx)** | 인과 에너지 VAD endpoint(무음 ≥ H) | 〃 | Qwen-Audio-3 방식 재현. H ∈ {0.3, 0.5, 0.8} s + dev 로 고른 임계값 |
| **S2m(H, dec, ctx)** | 1차 방출 공백 endpoint(텍스트 없는 방출이 H 만큼 이어짐) | 〃 | 모델 기반 endpoint. 에너지 VAD 가 약한 저 SNR 세트의 대조군(§4.6) |
| **S3(W, dec, ctx)** | 고정 주기 W s(에너지 최소점에서 절단) | 〃 | 의미와 무관한 주기 확정. W ∈ {2, 4, 8} s |
| **S4a** | 참조 발화 경계(오라클 endpoint) | Q06 / Q17 | 발화 단위 정확도 기준, 지연 하한 |
| **S4b(δ)** | 골드 COMMIT 단어 끝(골드 4 세트). 트리거 = emit_time(ref_chunk(end_time, δ)) | Q17 | "완벽한 SEM 검출기" 기준. S1 과의 차 = commit 오류 비용 |
| **S5** | 스트림 끝에서 1 회(오프라인 전체) | Q06 / Q17 | 오프라인 기준, 지연 최대. **상한이라는 보장은 없다**(§12) |
| **V8-dual(δ)** | S1+ 와 같은 트리거 | 같은 가중치의 δ=8 VapAsr 스트림(병렬) | "δ4 잠정 표시 + δ8 확정" 조합의 표시 지연·번복 대조. WER 은 정의상 S0-δ8 과 같다(§5.2) |

2차 디코더 약칭:

- **Q06** = Qwen3-ASR-0.6B
- **Q17** = Qwen3-ASR-1.7B(주 디코더)
- **V8** = 우리 모델의 δ=8 스트림(§5.2)

문맥 모드 ctx ∈ {iso, text, prefix} 는 §4.3 에서 정의한다.

**1차와 2차는 완전히 분리한다.** 2차 결과를 1차 디코더의 KV 나 문맥으로 되먹이지 않는다. 1차 모델은 계속 자기 가설을 조건으로 디코드한다.

- 교정 텍스트로 1차 KV 를 다시 prefill 하는 되먹임 방식은 설계가 달라 범위 밖이다. 후속 후보로 §13 에 남긴다.
- 이렇게 분리했기 때문에 2차는 **1차 방출 기록을 입력으로 하는 오프라인 시뮬레이션**으로 정확히 재현할 수 있다.

---

## 4. 세그먼트 정의

### 4.1 시각 규약(commit_metrics 와 동일)

- 청크는 80 ms 이고 1차 방출은 `[(k, tid)]` 로 기록한다.
- 방출 시각은 `emit_time(k) = (k+1)·0.08` 이다. flush 라운드(k ≥ K)는 K·0.08 로 둔다.
  - 코드: `vapasr/hf/commit_metrics.py` 의 `emit_time`(`:37-39`)
- 학습 직렬화 규칙: 단어 끝이 e 이면 그 단어의 마지막 토큰과 `<SEM_END>` 는 청크 `int(e/0.08)+δ` 에 놓인다.
  - 코드: `ref_chunk`(`:32-35`), `oracle_hyp`(`:305-318`)
  - 그래서 이상적인 모델에서 끝이 e 인 단어의 마지막 토큰은 emit_time(ref_chunk(e, δ)) = (int(e/0.08)+δ+1)·0.08 에 보인다. 이 값은 (e + δ·0.08, e + (δ+1)·0.08] 구간에 있다.
- 따라서 청크 k_c 에서 SEM 이 나왔다면, 직전 단어 끝의 추정값 ê 는 [(k_c−δ)·0.08, (k_c−δ+1)·0.08) 구간에 있다.
  - 다만 채점은 1 청크 이른 commit 도 정답으로 본다(`EARLY = 1`). 실제 단어 끝은 이 구간보다 한 청크 뒤일 수도 있다.
- 가설 단어 h 의 기호(모든 트리거 공통):
  - k_last(h): h 의 마지막 토큰을 방출한 청크
  - t_emit(h) = emit_time(k_last(h))
  - 추정 끝 ê(h) = (k_last(h) − δ + 0.5)·0.08
  - 가설 단어별 청크는 스트림 jsonl 의 `hyp.word_k`(`5103556` 의 `split_hyp` 가 기록)를 그대로 쓴다. `hyp.word_k` 가 없는 옛 스트림은 `semcommit_eval --score-only` 의 backfill(같은 tokenizer 로 `emitted` 를 다시 나눔)로 채운다.

### 4.2 SEM 세그먼트 j (연속한 두 확정 사이)

**텍스트 쪽(교체 대상)**

- 직전 확정 뒤부터 SEM_j 앞까지의 1차 가설 단어다. `split_hyp` 가 내는 이벤트의 `after`(가설 단어 번호)로 나눈다.
- SEM 세그먼트의 텍스트는 정의상 모두 t_trig 이전에 방출된 것이다.
- 1차 토큰은 정확히 한 세그먼트에 속한다.
  - 예외: S1+ 에서 앞선 hard 세그먼트의 "늦은 단어"(§4.6)는 그 hard 세그먼트에 속한다. 그래서 뒤 SEM 세그먼트에서는 뺀다.
- 단어 중간에서 commit 이 나오면(`mid_word`) 그 단어의 앞 조각은 j, 뒤 조각은 j+1 에 속한다.
  - 교체는 토큰 단위로 그대로 한다.
  - 번복 지표에서는 그 단어 전체를 뒤 세그먼트로 센다.

**트리거와 발송 시각**

- t_trig(j) = emit_time(k_c) 이다. 이 SEM 을 방출한 청크의 끝, 즉 그때까지 들어온 오디오의 끝이다. R_extra 와 무관하게 늘 이 값이다.
- t_dispatch(j) = t_trig(j) + R_extra 이다. 2차 디코드를 시작하는 시각이다. 기본 R_extra = 0 이다.

**오디오 절단점 b_j (다음 세그먼트와의 경계)**

- 기본값 **snap**:
  - 탐색 구간은 [nominal, t_dispatch] 이다. 하한을 추정 끝 구간의 하한 (k_c−δ)·0.08 이 아니라 nominal 로 둔다. 그래야 에너지 최소점이 확정 단어의 마지막 0–80 ms 안에 떨어져 단어를 자르는 일을 막는다(`EARLY = 1` 때문에 실제 끝은 더 뒤일 수 있다).
  - 구간 안 20 ms 프레임 에너지의 최소점이 인과 노이즈 바닥 + 8 dB 이하일 때만 그 점에서 자른다. 아니면(쉼 없이 이어지는 발화) nominal 에서 자른다. 그래야 다음 단어 onset 에서 자르는 일을 줄인다.
  - 이 구간은 이미 받은 오디오만 포함한다.
- **nominal**: (k_c−δ+1)·0.08 + 0.04
- **avail**(어블레이션): t_dispatch 그대로. 다음 단어 onset 이 섞일 위험이 있다. 확정 뒤 중복·절단 오류를 재기 위한 대조군이다.
- 절단 종류별로 "snap 이 무음을 찾은 비율"을 보고한다.

**재디코드 입력 오디오**

- 구간은 [b_{j−1} − pad_l, b_j + pad_r] 이다. prefix 모드는 §4.3 의 창을 쓴다.
  - **pad_l 기본값은 0 이다.** pad_l > 0 이면 앞 세그먼트 끝 오디오와 겹쳐 경계 단어가 두 번 전사될 수 있다. 그래서 pad_l = 0.16 s 는 어블레이션으로만 두고, 그때는 §4.3 의 경계 중복 제거를 적용한다.
  - pad_r = min(0.04, t_dispatch − b_j)
- **기본적으로 우측 문맥을 더 기다리지 않는다.** commit 시점까지 들어온 δ·80 ms 가 전부다.
- 어블레이션 R_extra ∈ {0.16, 0.32} s: 발송을 R_extra 만큼 늦추고 snap 구간도 그만큼 넓힌다. 이 지연은 t_dispatch 를 통해 L_final 에 한 번만 더해진다(§6.2).

**교체 규칙**

- 최종 전사는 세그먼트 순서대로 2차 텍스트를 공백으로 이은 것이다.
- 2차 출력이 비면 1차 텍스트를 유지한다(fallback).
- **hallucination guard**: 2차 단어 수 n2 가 max(3·n1, n1 + 3, ⌈r_max·seg_s⌉) 를 넘거나 n-gram 반복 루프가 보이면 1차를 유지한다.
  - n1 = 트리거 시점에 알던 1차 단어 수, seg_s = 세그먼트 오디오 길이(s)다.
  - r_max 는 EN 5 단어/s, KO 4 어절/s 로 둔다.
  - 오디오 길이 항을 넣은 이유: n1 이 0–1 이면 "n1 의 3 배" 기준만으로는 1차가 빠뜨린 음성을 2차가 맞게 복원해도 되돌린다. VAD·fixed·oracle 세그먼트는 1차 단어가 0 개일 수 있어서, 이 편향이 트리거 비교(H2)를 한쪽으로 기울인다.
  - n-gram 반복 판정: `detect_and_fix_repetitions` 가 출력을 바꿨거나, 같은 3-gram 이 4 번 이상 나오면 반복으로 본다.
- fallback 과 guard 발동 수를 **트리거 종류별로** 보고한다. guard 를 껐을 때의 오류율도 함께 보고해 guard 의 영향을 따로 보인다.

### 4.3 좌측 문맥 모드(ctx)

**iso**

- 위에서 정한 오디오만 준다. 텍스트 문맥은 없다.
- pad_l > 0 어블레이션에서는 아래 경계 중복 제거를 적용한다.

**text**

- iso 에 더해 직전 확정 텍스트(최대 3 세그먼트)를 Qwen3-ASR 의 `context`(system 프롬프트)로 준다.
- 근거: qwen_asr 0.0.6 의 `Qwen3ASRModel.transcribe(audio, context, language, …)` 와 `_build_messages`(system = context).
  - 위치: mxc env 의 `site-packages/qwen_asr/inference/qwen3_asr.py:300-345, 448-465`
- 위험: 모델이 context 를 되풀이(복사)해 삽입할 수 있다.

**경계 중복 제거(text 모드, 그리고 pad_l > 0 인 iso)**

- 2차 출력 앞머리의 최대 3 단어를 직전 확정 텍스트 꼬리와 정규화한 뒤 비교한다. 겹치는 가장 긴 접두를 잘라낸다.
- 잘라낸 횟수와 단어 수, 그리고 참조 대비 "경계 중복 삽입률"을 ctx 별로 보고한다.

**prefix (주 설정)**

- 창의 시작 세그먼트 q 는 다음 조건을 모두 만족하는 가장 작은 번호다(q ≤ j).
  1. b_{j−1} − b_{q−1} ≤ W_left(기본 15 s). 즉 **현재 세그먼트 [b_{j−1}, b_j] 는 늘 들어가고**, W_left 는 그 앞의 좌측 문맥 길이만 제한한다. max_seg(20 s)가 W_left 보다 길어도 창이 정의된다.
  2. b_{q−1} 은 마지막 endpoint 경계보다 앞서지 않는다. endpoint 경계는 VAD·S2m endpoint, oracle-utt 발화 끝, 스트림 시작이다. SEM·fixed·max_seg 경계는 endpoint 가 아니다.
- 오디오 창은 [b_{q−1}, b_j + pad_r] 이다.
- assistant 응답을 **세그먼트 q..j−1 의 확정 텍스트로 prefill** 한다. 이 텍스트는 정확히 오디오 창 [b_{q−1}, b_{j−1}] 에 든 세그먼트의 것이다. 이어서 생성되는 부분만 세그먼트 j 의 텍스트로 쓴다.
  - q = j 이면 prefill 이 없고 창은 현재 세그먼트뿐이다. 그래서 iso 와 같다.
  - 결과: S2 처럼 세그먼트가 endpoint 로만 나뉘면 prefix 가 iso 와 같아진다. 이것은 설계에 따른 것이며 보고서에 적는다.
- 근거: qwen_asr 의 스트리밍 모드가 정확히 `prompt = state.prompt_raw + prefix` 형태로 누적 오디오를 다시 먹인다(`qwen3_asr.py:657-748`, rollback 규칙).
  - 다만 그 함수는 vLLM 전용이고 단일 스트림만 받는다. vLLM 이 아니면 `:626, :698` 에서 ValueError 가 난다. mxc env 에는 vLLM 이 없다.
  - 그래서 같은 프롬프트 구성을 transformers `generate` 배치로 직접 구현한다(§10.1 래퍼).
- 장점: 음향 좌문맥과 텍스트 좌문맥을 모두 주면서도 이미 확정한 텍스트는 바뀌지 않는다(중복 없음).
- 비용: 창 전체를 매번 다시 인코드한다(§6.3 재처리 배율).

**직렬 의존**

- text·prefix 모드에서 세그먼트 j 의 입력은 j−1 의 확정 결과에 의존한다. 그래서 한 스트림 안에서는 순서대로만 디코드할 수 있다.
- 이 의존은 compute-inclusive 확정 지연에 반영한다(§6.2 재귀식).

**공통 규칙**

- 언어는 세트별로 강제한다(`language Korean|English` 를 주고 `<asr_text>` 뒤의 텍스트만 쓴다; `_build_text_prompt`).
- 모든 Qwen 조건(S5 포함)은 같은 래퍼로 돌린다(§10.1 `QwenAsrDecoder`). 래퍼는 `transcribe()` 가 하는 처리를 똑같이 재현한다.
  - 0.5 s 미만 입력은 뒤를 0 으로 채운다. 이 패딩은 qwen_asr 에서 `split_audio_into_chunks` 안에만 있다(`utils.py:246, 322-330`). processor + generate 를 직접 부르면 적용되지 않으므로 래퍼가 직접 한다.
  - 출력은 `parse_asr_output`(`utils.py:403`)으로 파싱하고, 그 안의 `detect_and_fix_repetitions`(`:432`)도 적용한다. prefix 모드에서는 생성된 부분에만 적용한다.
  - max_new_tokens = max(512, ⌈12·창 길이(s)⌉ + 32). qwen_asr 기본값이 512 다(`qwen3_asr.py:153, 182`). EOS 없이 상한에 닿은 출력(n_gen == max_new_tokens)은 "잘림"으로 세어 조건별로 보고한다.
- V8 과 V8-dual 은 자기 스트림 상태를 쓰므로 ctx 가 해당되지 않는다.

### 4.4 조기 commit, 단어 중간 commit

- 확정은 되돌리지 않는다. 기본 rollback 은 r = 0 이고, 한 번 교체한 텍스트는 final 이다.
- 어블레이션 r = 1:
  - 다음 prefix 재디코드 때 직전 세그먼트의 마지막 1 단어를 prefill 에서 빼고 다시 생성하게 한다. qwen_asr 의 unfixed_token 과 같은 발상이다.
  - 이때 바뀐 단어 수는 **확정 후 수정(post-final revision)** 으로 따로 센다. final 계약을 어긴 비용이다.
- 세그먼트마다 그 SEM 의 범주를 붙인다.
  - `text_position_metrics`(`:238-260`)의 correct / ambiguous / premature(mid_word, no_word, N 범주, other_inside)
  - 골드 세트는 `gold_commit_counts`(`:425-441`)의 hit / at_ambig / at_no
- 범주별로 경계 ±1 단어의 최종 오류율과 내부 단어의 오류율을 나눠 본다(H4).
- sem-guard(텍스트 없는 SEM 금지)는 켠 채로 둔다(`semcommit_eval.py:218-221`, `decode_rows`). 그래서 **SEM 세그먼트는** 비지 않는다. 다른 트리거의 세그먼트는 1차 단어가 0 개일 수 있다.

### 4.5 트리거 종류, 최소·최대 길이, 스트림 끝

**트리거 종류**

- **soft 트리거**: SEM 뿐이다.
- **hard 트리거**: 그 밖의 모든 트리거다. VAD·S2m endpoint, EOS, max_seg 강제 확정, fixed, oracle-utt, oracle-gold, offline 이 해당한다.
- 합치기(S1+, `union_triggers`): 트리거를 발생 시각 순서로 처리한다. 뒤에 온 트리거의 절단점이 이미 확정한 경계 이하이면 그 트리거는 무시한다. 예를 들어 SEM 절단점이 VAD 절단점 바로 앞이어도, VAD 트리거가 먼저 발동해 그 구간을 확정했다면 늦게 나온 SEM 은 아무 일도 하지 않는다.

**min_seg(기본 0.8 s): soft 트리거에만 적용**

- SEM 세그먼트의 오디오가 min_seg 보다 짧거나 1차 단어가 2 개 미만이면 트리거를 **보류**한다. 보류한 세그먼트는 다음 트리거(종류 무관)와 합친다.
- **hard 트리거는 길이와 관계없이 항상 확정한다.**
  - 예: VAD endpoint 뒤 0.4 s 짜리 한국어 대답어 "네"는 다음 발화를 기다리지 않고 즉시 확정한다.
  - 스트림 끝 꼬리가 min_seg 보다 짧아도 EOS 에서 확정한다.
  - 그래서 S1+ 가 보류 때문에 S2 보다 늦게 확정하는 역전은 생기지 않는다.
- 짧은 hard 세그먼트의 디코드:
  - prefix 모드는 §4.3 창(앞 확정 텍스트 prefill)으로 디코드한다.
  - iso·text 모드는 래퍼의 0.5 s 0 패딩 규칙을 따른다.
- 보류된 단어는 확정 지연이 늘고, 그 증가가 지표에 그대로 반영된다.
- 어블레이션: {0, 0.8, 1.5} s

**max_seg(기본 20 s)**

- 마지막 확정 뒤 20 s 동안 트리거가 없으면 S3 방식(§4.6 fixed 절단 구간)으로 강제 확정한다. 연산·메모리 상한이다.
- 트리거별로 max_seg 강제 확정 비율을 보고한다.

**스트림 끝(EOS)**

- 마지막 확정 뒤의 꼬리(flush 라운드 방출 포함)는 EOS 에서 확정한다.
- EOS 트리거 시각 = min(endpoint, 오디오 끝) 이다.
  - endpoint 는 인과 VAD 가 마지막 음성 끝 뒤에 H_eos(0.8 s) 무음을 확인한 시각이다.
  - endpoint 가 오디오 끝까지 발동하지 않으면 오디오 끝(K·0.08)을 쓰고, 그 단어들도 **지연 계산에 포함한다**. 가장 나쁜 경우를 통계에서 빼지 않기 위해서다.
  - 세트·트리거별로 "endpoint 미발동(오디오 끝 대체) 비율"을 보고한다.
- R1 오디오 규약(§7.1)은 꼬리를 H_eos + 0.5 s 보다 길게 둔다. 그래서 깨끗한 꼬리라면 endpoint 가 오디오 끝 전에 발동한다.

### 4.6 다른 트리거의 세그먼트

**hard 트리거 공통: 텍스트 쪽 배정과 인과성**

- 세그먼트의 1차 텍스트는 **t_trig 까지 방출된 가설 단어** 가운데 추정 끝 ê(h) ≤ b 이고, 아직 앞 세그먼트에 배정되지 않은 것이다.
- **늦은 단어**: t_trig 뒤에 방출됐지만 ê(h) ≤ b 인 단어다.
  - 예: δ8(0.72 s)에서 H 0.3·0.5 인 VAD, 또는 δ4(0.40 s)에서 H 0.3 인 VAD 에서는 b 직전 단어의 마지막 토큰이 트리거 시점에 아직 방출되지 않는다(§4.1: 방출은 e + δ·0.08 뒤).
  - 2차가 성공한 세그먼트에서는 그 오디오가 이미 확정됐다. 그래서 늦은 단어는 잠정 표시하지 않고 버린다. 이런 단어는 **direct-final**(표시 전 확정)로 센다.
  - fallback·guard 로 1차 텍스트를 유지한 세그먼트에서는, 늦은 단어가 방출될 때 그 세그먼트의 최종 텍스트 뒤에 붙인다. 이 단어의 확정 시각은 자기 방출 시각이다.
- fallback·guard·min_seg 판단은 **t_trig 시점에 알던 텍스트로만** 한다. 미래 정보를 쓰지 않기 위해서다.
- t_trig 까지 방출됐지만 ê(h) > b 인 단어는 다음 세그먼트의 잠정 텍스트로 남는다.

**VAD(S2)**

- 기존 `vapasr/data/vad.py:10-23` 의 `energy_vad` 는 인과적이지 않다. 노이즈 바닥을 **신호 전체의 10 백분위**로 잡기 때문이다(`:15`). 또 "분리 stereo(누설 −64 dB)"를 전제한다(`:1`).
- 그래서 인과 버전을 새로 만든다.
  - 20 ms 프레임
  - 노이즈 바닥은 **고정 절대값(−60 dBFS)에서 시작**한다. 이후 지수 이동 최소값으로 갱신한다(내려갈 때는 빠르게, 올라갈 때는 느리게).
  - 처음 0.5 s 로 초기화하지 않는다. mxc 표본(세트당 150 발화)에서 상담 115 개, 강의 70 개는 처음 0.5 s 의 절반 이상이 음성이었기 때문이다(10 백분위 + 12 dB 기준).
  - 히스테리시스는 +12 / +8 dB(dev 에서 조정)
  - 무음이 H 만큼 이어지면 endpoint
- 절단 b = 무음 시작, 트리거 t_trig = 무음 시작 + H.
- **검증(dev)**: 정렬이 있는 dev(71631 tune 절반, LibriSpeech dev streams)에서 참조 발화 간격을 정답으로 삼는다.
  - 간격 ≥ H 인 자리의 endpoint 재현율, endpoint 정밀도(발화 안에서 발동한 비율), endpoint 지연(참조 발화 끝 → 트리거)을 잰다.
  - 이 결과로 임계값을 고르고, 고른 값의 test 결과를 보고한다.
- 보고 항목: 세트별 S2 세그먼트 길이 분포, max_seg 강제 확정 비율, endpoint 미발동 비율.

**1차 방출 공백 endpoint(S2m)**

- S2 의 정식 변형으로 격자에 넣는다. 모델 기반이고 인과적이며 추가 부품이 필요 없다.
- 마지막 텍스트 토큰 방출 뒤 텍스트 없는 방출이 H 동안 이어지면 endpoint 로 본다.
  - t_trig = t_emit(마지막 단어) + H
  - 절단 b = 그 단어의 nominal 끝 (k_last − δ + 1)·0.08 + 0.04
- 에너지 VAD 가 약한 저 SNR 세트(§7.2 의 71631)와 누설이 있는 세트에서 S2 가 허수아비 대조군이 되지 않게 하려는 것이다.

**fixed(S3)**

- T_j = j·W 에서 트리거한다.
- 절단 b_j 는 [T_j − (δ+1)·0.08 − 0.48, T_j − (δ+1)·0.08] 안의 에너지 최소점이다.
  - 끝이 e ≤ T_j − (δ+1)·0.08 인 단어의 이상적 마지막 토큰 방출은 (int(e/0.08)+δ+1)·0.08 ≤ T_j 이다. 그래서 이 구간의 단어는 T_j 까지 이미 방출돼 있다.
  - 이전 정의 [T_j − 0.48, T_j − δ·0.08] 은 δ6 에서 폭이 0, δ8 에서 빈 구간이었고, 끝이 T_j − δ·0.08 인 단어는 T_j + 0.08 에야 방출돼 한 청크 어긋났다.
- 실제 모델의 방출은 이상적 방출보다 늦을 수 있다. 그 경우는 위의 hard 트리거 공통 규칙(늦은 단어)을 따른다.

**oracle-utt(S4a)**

- 구간은 참조 발화 경계다(± 0.08 s).
  - manifest·stub 세트: words 행의 `segments[].offset_s .. offset_s + dur_s`
  - 71631 장문 창: words 행의 `utt_bounds`(§7.2)
- 트리거는 참조 발화 끝이다. 오라클이라 검출 지연은 0 이다. 텍스트 쪽은 hard 트리거 공통 규칙을 따르므로, 발화 끝 시점에 아직 방출되지 않은 마지막 단어는 늦은 단어가 된다.

**oracle-gold(S4b)**

- 절단은 골드 COMMIT 단어의 `end_time` + 0.04 다.
- 트리거는 **emit_time(ref_chunk(end_time, δ)) = (int(end_time/0.08)+δ+1)·0.08** 이다. 이상적 SEM 검출기가 방출하는 시각이다.
  - 이전 정의(end_time + δ·0.08)는 실제 방출 시각보다 0–0.08 s 일러서 S1 대 S4b 지연 비교가 S4b 쪽으로 기울었다.

**offline(S5)**

- 스트림 전체를 한 세그먼트로 두고, 트리거는 스트림 끝이다.

---

## 5. 2차(full-context) 디코더 선택지: mxc 에서 확인한 것

### 5.1 (a) 원본 Qwen3-ASR (비스트리밍, written-form 출력)

모델 설명은 [[source-qwen3-asr]] 에 있다.

**가중치**

| 모델 | 경로 | 크기 | text | audio |
|---|---|---|---|---|
| Qwen3-ASR-0.6B | `/soundai/Model/Qwen3-ASR-0.6B` | 1.8 GB | 28 층, hidden 1024 | 18 층 |
| Qwen3-ASR-1.7B | `/soundai/Model/Qwen3-ASR-1.7B` | 4.4 GB, 2 shard | 28 층, hidden 2048 | 24 층 |

**패키지(mxc env `/soundai/users/tskim/VAPKT-data/conda/envs/vapasr`)**

- 설치돼 있음: `qwen_asr 0.0.6`(transformers==4.57.6 고정), `whisper_normalizer 0.1.15`, `jiwer 4.0.0`, `rapidfuzz 3.14.6`
- **vLLM 은 없다.** 서버에 설치할 수 없으므로 transformers 백엔드만 쓴다.
- 입력 상한은 1200 s(`qwen_asr/inference/utils.py:34`)다. 타임스탬프는 Qwen3-ForcedAligner 로 얻으며 상한은 180 s 다.
- API: `transcribe(audio, context, language, return_time_stamps)`(`qwen3_asr.py:300`). transformers 경로의 생성 상한 기본값은 max_new_tokens = 512 다(`:153, 182`).

**출력 표기**

- 구두점과 대소문자를 포함한 written form 이다.
- 한국어 숫자는 두 근거가 엇갈린다.
  - 소표본 조사에서는 한글 읽기였다([[source-asr-output-style-probe]]).
  - speechlm Qwen 전사에는 아라비아 숫자 단어가 흔했다. 학습 run 에서 KO 숫자 단어 39,209 개를 사람 전사로 복원했다(run json `sets.ko.ko_digit_spoken`).
- 채점 정규화 `score_ko` 는 숫자·Latin 을 한글로 바꾸지 않는다(`vapasr/data/textnorm.py:97-100`). 그래서 2차 출력의 아라비아 숫자와 로마자(예: 'OK' 대 참조 '오케이')는 그대로 오류로 잡힌다.
  - AIHub 참조는 한글 읽기다(예: 전화망 "예 금 저 삼만 팔천 원 입금했어요"). 금액·번호가 흔한 상담·전화망에서 2-pass 쪽에 체계적으로 불리하다.
  - 대응은 §6.1 의 보조 지표와 §12 S-1 의 이중 조건이다.

**오류 상관**

- 우리 thinker 는 Qwen3-ASR-0.6B 에서 시작해 전체 미세조정됐다(`config.thinker_name_or_path=/soundai/Model/Qwen3-ASR-0.6B`, §1.2).
- 그래서 Q06 의 오류는 1차 모델의 오류와 상관이 있을 것이다. Q17 을 주 디코더로 둔다.

**학습 데이터와 성능**

- Qwen3-ASR 의 학습 데이터는 공개되지 않았다. LibriSpeech·Kspon 같은 공개 벤치마크가 학습에 들어갔을 가능성을 배제할 수 없다(§13).
- 위키에는 Qwen3-ASR 의 Kspon·AIHub 성능 실측이 없다([[source-qwen3-asr]]). 그래서 S5 가 우리 1차 모델보다 낫다는 것은 전제하지 않고, P0-5 에서 먼저 잰다(§10.2).

### 5.2 (b) 우리 VapAsr 를 2차로 쓰는 방법

**b-literal: 학습된 적 없는 방식이다**

- 학습 시퀀스는 청크마다 오디오 임베딩 1 개, (δ 지연된) 텍스트, `<NEXT_AUDIO>` 가 교차하는 배열이다(`modeling_vapasr.py:182-210`, `batch_decode.py`).
- 오디오 끝 뒤의 `<EMPTY_AUDIO>` flush 는 tail_margin(2 청크) 정도의 꼬리만 학습했다.
- "세그먼트 오디오를 전부 먼저 넣고 디코드"하는 배열은 한 번도 본 적이 없다.
- 그래서 **b-literal 은 50 스트림 스모크로 실패 여부만 확인**하고 본 비교에서는 뺀다.

**V8-dual: 학습 분포 안의 대조군**

- 같은 가중치를 학습된 최대 δ 인 δ=8(config delays [2,3,4,6,8])로 병렬 디코드해 둔다.
- 세그먼트 j 의 최종 텍스트는 δ8 스트림에서 추정 끝 (k − 8 + 0.5)·0.08 이 (b_{j−1}, b_j] 에 드는 단어들이다.
- 확정 시각은 max(t_trig, δ8 스트림이 그 세그먼트의 마지막 단어를 방출한 시각)이다.
- **구성상 따라 나오는 성질**
  1. 모든 δ8 단어는 정확히 한 세그먼트에 들어간다. 그래서 세그먼트 텍스트를 이은 최종 전사는 δ8 가설 전체와 같다. **V8-dual 의 WER 은 S0-δ8 과 정의상 같다.**
  2. 확정 시각은 S0-δ8 의 방출 시각 이상이다. 그래서 (WER, L_final) 양쪽에서 S0-δ8 보다 나을 수 없다.
  3. 같은 트리거에서 V8-dual 의 확정 시각은 S1+ 의 algorithmic 확정 시각(= t_trig) 이상이다.
- 쓰임새:
  - "δ4 잠정 표시 + δ8 확정" 조합의 표시 지연·번복 대조
  - H0/S-3 에서 "확정 지연이 같거나 더 긴 우측 문맥 대조군". WER 대조군으로는 S0-δ8 하나만 센다(성질 1).
- 추가 GPU 비용은 없다. δ8 1차 run 은 어차피 S0-δ8 로 돌린다.
- Nemotron 인코더([56,0])는 인과적이다. 그래서 스트림 전체 특징을 잘라 쓴 결과가 세그먼트만 인코드한 결과와 같다(좌문맥 56 프레임 한도 안). V8 계열에서는 특징을 다시 계산할 필요가 없다.

### 5.3 (c) `/soundai/Model` 의 기타 후보

- `nemotron-3.5-asr-streaming-0.6b`([[source-nemotron-3-5-asr-streaming]]):
  - RNN-T 이고, att_context [56,13] = 1.12 s 청크 또는 오프라인으로 돌 수 있으며, 구두점·대소문자를 출력한다.
  - 과거 기록에서 kspon-dev CER 이 0.202 로 E2(0.167)보다 나빴다. EN 보조 후보로만 둔다(선택).
- Whisper large-v3:
  - 가중치는 `/soundai/users/tskim/VAPKT-data/baselines/whisper-large-v3/model.safetensors` 에 있다(+ fp32 shard, v2 폴더도 있음).
  - EN 에서 Qwen 이 아닌 2차 디코더 대조로 쓸 수 있다(선택). 한국어는 과거 NIKL 평가 기록만 있다.
- 대형 오디오 LLM(Qwen3-Omni-30B-A3B, A.X-K2-Raon-Speech-21B-A3B, Raon-SpeechChat-9B, MiniCPM-o-4_5, Nemotron-Labs-Audex):
  - 비용이 크고 프롬프트 의존도 커서 1 차 계획에서는 뺀다.
- **권고**: 주 디코더는 Q17, 보조는 Q06, 추가 모델 없는 대조는 V8-dual(= S0-δ8 WER).

---

## 6. 지표

### 6.1 정확도와 정규화

**지표**

- KO: `score_ko` 로 계산한 CER_nospace(주), CER_space, WER
- EN(LibriSpeech, 골드): `score_en` WER(주). `score_en` 은 지원하는 숫자만 canonical form 으로 바꾼다.
  - 보조로 Whisper `EnglishTextNormalizer`(`whisper_normalizer.english`)를 참조와 가설 양쪽에 적용한 WER 도 보고한다. Q17 의 written form(숫자, 약어 표기) 차이가 남기 때문이다.
- 오류 카운트는 `commit_metrics.asr_counts`(`:159-166`) 및 `single_turn_eval.score_pair` 와 같다.
- Open-ASR 계열(ESB): Whisper 정규화 WER 을 주 지표로 하고, `score_en` WER 을 함께 적는다.

**집계**

- micro 평균(Σ 오류 / Σ 참조 단위)으로 세트별, 그리고 언어별 풀로 낸다.
- 2차 원출력과 1차 원출력은 저장해 두고, 정규화한 문자열로 덮어쓰지 않는다.

**KO 숫자·로마자 보조 지표**

- 참조에 숫자가 든 발화는 AIHub 3 세트에 0 건, kspon_clean 에 1 건이다(mxc 측정).
- P0-5 에서 세트·시스템별로 "아라비아 숫자 또는 로마자가 든 가설"의 비율을 먼저 잰다.
- 보조 지표: 비교하는 두 시스템 가운데 어느 쪽이든 가설에 아라비아 숫자나 로마자가 있는 발화를 뺀 **paired CER**. 뺀 발화 수를 세트별로 표에 적는다.
- 한 세트에서 제외 비율이 10 % 를 넘으면, 발화를 통째로 빼는 대신 **숫자·로마자 구간 마스킹 CER** 을 함께 낸다.
  - 정의: 참조–가설 문자 정렬에서 가설의 숫자·로마자 연속 구간과, 그 구간에 정렬된 참조 문자(바로 이웃한 삭제 포함)를 양쪽에서 지우고 CER 을 다시 센다.
- §12 S-1 의 KO 판정은 주 CER 과 이 보조 CER 을 **모두** 만족해야 한다.

**참조 규약(1차 전사 기준, 기존 그대로)**

- Kspon: `target_ko(raw,'kspon')`(프로젝트 manifest)
- LibriSpeech: lexical
- AIHub: 벤치마크 Transcript(`target_ko(raw,'aihub')` 로 구두점만 정리)

**상담 세트**

- text-seen 34 발화(§7.4)를 포함한 CER 과 제외한 CER 을 함께 보고한다.

### 6.2 지연(5 종)의 정의

기호(§4.1 에 더해):

- e_i: 참조 단어 i 의 끝(words.jsonl `end_time`)
- 세그먼트 j: 절단 구간 (b_{j−1}, b_j], 트리거 t_trig(j), 발송 t_dispatch(j) = t_trig(j) + R_extra
- **seg(i)** (참조 단어 i 의 세그먼트): 시각 기준이 주 정의다. e_i ∈ (b_{j−1}, b_j] 인 j 다.
  - 최종 전사와 참조를 정렬해 정하는 방식은 보조로만 보고한다. written form 과 한국어 띄어쓰기·숫자 차이 때문에 정렬이 흔들리고, 삭제된 단어를 빼면 삭제가 많은 시스템이 유리해지기 때문이다.
  - 시각 기준에서는 최종 전사에서 삭제된 참조 단어도 지연이 정의된다. 삭제율은 따로 보고한다.
- **seg(h)** (1차 가설 단어 h 의 세그먼트): 텍스트 쪽 배정(§4.2, §4.6, `assign_first_pass`)으로 정한다. 참조 정렬이 필요 없다.

**1. 표시 지연** L_disp(i) = t_emit(h(i)) − e_i

- h(i) 는 참조 단어 i 에 `align_pairs`(`commit_metrics.py:136-139`, `norm_word` 정규화)로 짝지은 1차 가설 단어다.
- 1차 잠정 텍스트가 보이기까지 걸린 시간이다. 삭제된 참조 단어는 빼고, 삭제율을 따로 보고한다.
- 이 정의는 `5103556` 의 `display_metrics`(참조 단어별 표시 지연)와 같다. `twopass.py` 는 그 함수를 재사용한다.

**2. 트리거 지연** L_trig(j) = t_trig(j) − e_last(j)

- e_last(j) 는 seg(i) = j 인 참조 단어 가운데 마지막 단어의 끝이다(시각 기준).
- 주의: SEM 트리거의 이 값은 기존 `timing_metrics` 의 latency_s 와 일반적으로 같지 않다. `timing_metrics`(`:262-275`)는 A 등급 경계와 청크 창으로 매칭된 쌍만 센다. 두 값이 같은 것은 매칭된 A 쌍에 한해서다.

**3. 확정 지연(finalisation latency)** L_final(i) = t_final(seg(i)) − e_i

- t_final(j) 를 두 가지로 계산한다.
  - **algorithmic**(C = 0): t_final(j) = t_dispatch(j). 트리거 시각이 단조 증가하므로 큐 대기는 없다.
  - **compute-inclusive**: t_final(j) = max(t_dispatch(j), t_final(j−1)) + C(j)
    - 스트림 하나를 GPU 워커 하나가 순서대로 처리한다고 본다. text·prefix 는 j 의 입력이 j−1 의 결과에 의존하므로 이 순서가 필수다. iso 도 같은 단일 워커 큐 식을 쓴다.
    - C(j) 는 §6.3 의 batch-1 실측 지연 모델 값이다(입력 오디오 길이와 생성 토큰 수의 함수).
    - SEM 트리거가 몰릴 때(bias +2)나 C 가 큰 Q17 에서, 세그먼트를 독립으로 보는 식은 지연을 과소평가한다.
- R_extra 는 t_dispatch 에서 한 번만 더해진다. t_trig 자체는 늦추지 않는다.
- 특수한 경우:
  - EOS 는 §4.5 규약(min(endpoint, 오디오 끝))을 따르며, 오디오 끝 대체도 지연에 포함한다.
  - S0 는 번복이 없으므로 t_final = t_emit, L_final = L_disp 다(삭제는 빼고 삭제율로).
  - V8-dual 은 t_final(j) = max(t_trig(j), δ8 스트림이 세그먼트 j 의 마지막 단어를 방출한 시각)이다.
  - fallback 세그먼트의 늦은 단어는 자기 방출 시각에 확정된다(§4.6).
- 구성상 성질: min_seg 보류는 soft 트리거에만 걸리고 트리거는 시각 순서로 합친다(§4.5). 그래서 같은 H 의 S1+ 는 algorithmic 기준으로 모든 참조 단어를 S2 보다 늦지 않게 확정한다. 예외는 fallback 된 세그먼트의 늦은 단어(자기 방출 시각에 확정, §4.6)뿐이며, 그 수를 따로 보고한다.

**4. 잠정 표시 시간(provisional duration)** P(h) = max(0, t_final(seg(h)) − t_emit(h))

- 1차 가설 단어 h 기준이다. 참조 정렬이 필요 없으므로 모든 세트에서 계산한다. 참조 단어 기준 지표(1–3, 5)와는 따로 보고한다.
- 1차가 삭제했고 2차가 복원한 단어는 가설 단어가 없으므로 P 에 들어가지 않는다. 이런 단어 수는 번복 지표(§6.4)에서 센다.
- t_final(seg(h)) < t_emit(h) 인 단어(늦은 단어 포함)는 **direct-final**(표시 전 확정)이다. P 에는 0 으로 넣고, 그 비율을 트리거·δ 별로 따로 보고한다.

**5. 발화 끝 확정 지연** EOU(u) = t_final(seg(u 의 마지막 참조 단어)) − e_last(u)

- LLM 에 넘기는 시점과 턴테이킹에 바로 연결되는 지표다.
- 정렬이 없는 세트는 e_last(u) 를 원 클립의 오프라인 에너지 VAD 음성 끝으로 근사한다(±40 ms 로 표시).

**보고 형식**

- 단어 가중 p50 / p90 / p99
- 0.5 s 이내, 1.0 s 이내에 확정된 비율
- 세그먼트 길이 분포(초 단위, 단어 수 단위)
- direct-final 비율, endpoint 미발동 비율, max_seg 강제 확정 비율

**손 계산 예(테스트에 그대로 쓴다)**

1. 단일 세그먼트(δ4, 단어 끝 1.00 s)
   - SEM 청크 = int(1.00/0.08)+4 = 16
   - t_trig = 17·0.08 = 1.36 s
   - C = 0.2 s 이면 t_final = 1.56 s, L_final = 0.56 s(algorithmic 이면 0.36 s)
2. 연속 트리거 큐(compute-inclusive)
   - 세그먼트 j−1: t_dispatch 1.36 s, C = 0.40 s → t_final = 1.76 s
   - 세그먼트 j: t_dispatch 1.60 s, C = 0.20 s → t_final = max(1.60, 1.76) + 0.20 = **1.96 s**
   - 세그먼트를 독립으로 보는 식이면 1.80 s 가 되어 0.16 s 를 과소평가한다.
3. VAD 인과성(δ8, H 0.3)
   - 단어 끝 e = 2.00 s, 무음 시작 b = 2.00 s → t_trig = 2.30 s
   - 이상적 마지막 토큰 방출 = (int(2.00/0.08)+8+1)·0.08 = 34·0.08 = 2.72 s > 2.30 s
   - 그래서 이 단어는 늦은 단어다. 2차가 성공하면 direct-final 이고 P = 0 이다.
4. fixed 절단 구간(δ8, T = 4.00 s)
   - 구간 = [4.00 − 0.72 − 0.48, 4.00 − 0.72] = [2.80, 3.28] s
   - 끝 3.20 s 인 단어의 이상적 방출 = (40+8+1)·0.08 = 3.92 s ≤ 4.00 s

### 6.3 2차 연산 비용

- 세그먼트별로 입력 오디오 길이(s), prefill 텍스트 토큰 수, 생성 토큰 수, 잘림 여부, wall time 을 기록한다. 배치 실행의 wall time 은 참고치다.
- **지연 모델**:
  - `bench-latency` 로 H200 1 장, batch 1, bf16 조건에서 잰다.
  - 세그먼트 길이 bucket(0.5/1/2/4/8/15/30 s)과 생성 길이 조합마다 500 개를 잰다.
  - 이 측정으로 C(len, n_tok) 를 적합하고, compute-inclusive L_final 에 쓴다.
- 처리량: batch 32 에서의 GPU-s / audio-h 와 peak GPU memory.
- **재처리 배율** = Σ_j (2차 입력 창 길이) / 스트림 오디오 길이.
  - prefix 는 §4.3 정의를 따라 Σ_j (b_j + pad_r − b_{q(j)−1}) / 스트림 길이다. 근사하면 1 + E[좌측 문맥 길이] / E[세그먼트 길이] 이고, 좌측 문맥은 W_left 와 마지막 endpoint 경계로 잘린다.
  - 예상값:
    - iso 는 약 1.0(pad_l 0)이다.
    - prefix 는 endpoint 가 자주 오는 S1+ 에서 약 1.3–2 다.
    - endpoint 가 없는 S1(SEM + EOS)은 골드 기준 세그먼트 약 5 s(EN)에서 약 1 + 15/5 = 4 다.
    - S2 는 세그먼트가 endpoint 로만 나뉘어 약 1.0 이다(prefix = iso).
    - fixed W2 는 약 8.5 다.
  - 그래서 prefix 의 배율은 구성상 3 을 넘는 설정이 있다. 배율 상한은 성공 기준에서 빼고, W_left 격자 결과와 함께 보고한다(§12 S-5).
- RTF_2nd = Σ C / Σ 스트림 오디오. 단일 스트림 기준의 GPU 점유율이다.

### 6.4 번복(flicker)

- 세그먼트별 **어휘 번복률** = edit(정규화한 1차 단어열, 정규화한 2차 단어열) / 1차 단어 수.
  - 1차 단어열은 t_trig 시점에 알던 텍스트(늦은 단어 제외)다.
  - 구두점이나 대소문자 차이는 번복으로 세지 않는다.
- 화면 표시용 **표면 번복률**(구두점·대소문자 포함)은 따로 센다.
- 번복된 세그먼트의 비율, 그리고 번복된 단어가 가장 최근에 표시된 단어에서 몇 단어 떨어져 있는지 본다.
- **참조 대비 3-way 분류**(참조 단어 기준으로 두 번 정렬):
  - fixed: 1차 오류 → 최종 정답
  - broken: 1차 정답 → 최종 오류
  - changed-still-wrong: 바뀌었지만 여전히 오류
  - unchanged: 바뀌지 않음
  - H6 는 fixed / broken 비로 판정한다.
- 경계 중복 삽입률(§4.3)을 ctx 별로 보고한다.
- r = 1 어블레이션에서는 확정 후 수정 수를 센다.

### 6.5 commit 품질(해석용 맥락)

같은 1차 run 에서 아래 값을 그대로 얻는다.

- `semcommit_eval` 보고서의 text P/R, PCR, pcr_raw, sem_per_min, late2 P/R, 지연
- 골드 세트는 `semcommit_gold.py score-eval` 의 P/R/F1 과 PCR_gold
- **stub 세트(참조 단어 없음)는 commit 지표에서 뺀다.** words = [] 인 행에서는 모든 SEM 이 `map_events` 에서 −1 로 사상돼 premature(no_word)로 잡힌다(`commit_metrics.py:192-209, 250-251`). `semcommit_eval` 보고서는 언어별로 합산하므로, stub 세트를 정렬 세트와 한 번에 돌리면 KO PCR 이 오염된다. 그래서 stub 세트는 별도 `semcommit_eval` 실행(별도 `--out`)으로 돌린다(§10.1).

### 6.6 T3(commit latency 지표)와의 관계

T3 는 범위 안이다(머리말의 범위 정정). 1차 디코드와 commit·display·after_text 지연은 커밋 `5103556` 판 `semcommit_eval.py`·`commit_metrics.py` 로 계산하고, 2-pass 고유 지표(L_trig, L_final, P(h), 번복)는 새 파일(`twopass.py`)이 계산한다.

| 지표 | 지금 계산할 수 있는 범위 | 단어 정렬이 없어 근사로 남는 부분 |
|---|---|---|
| WER/CER, 번복률, 3-way, 2차 비용, RTF | 전 세트 | 없음 |
| 잠정 표시 시간 P(h), direct-final 비율 | 전 세트(가설 시각만 사용) | 없음 |
| L_disp, L_trig, L_final(algorithmic·compute-inclusive) | 단어 정렬이 있는 세트: LibriSpeech test(streams/utt), Kspon eval, 골드 4 세트, 71631 장문 창. dev 세트(kspon-dev, librispeech-dev, 71631 tune 절반, 골드 dev 절반)에서도 계산하지만 **튜닝에만 쓰고 판정에는 쓰지 않는다** | **정렬이 없는 AIHub 3 세트와 ESB 에서 단어 끝 기준 지연.** 참조 단어 끝 시각을 만들고 검증하는 정렬 작업이 먼저 필요하다 |
| EOU | 정렬 세트, 그리고 비정렬 세트는 VAD 음성 끝 근사값(근사임을 표시) | 근사 대신 정밀한 단어 끝 기준 EOU |
| compute-inclusive L_final | 2차 C 실측(§6.3)과 재귀 큐 식(§6.2) | 1차 스트리밍 tick(실시간 wall-clock)까지 포함한 종단 commit 지연 |
| δ·bias 가 commit 지연에 주는 영향 | 골드·정렬 세트에서 `5103556` 의 창 없는 commit 지연(commit_latency_metrics)과 기존 timing_metrics 로 | 비정렬 세트로 확장하는 일 |

결론:

- H1·H3·H5·H6 은 전 세트에서 판정할 수 있다.
- H2(지연 효과 크기, 정확도 비용)와 H4 는 정렬·골드 **test** 세트에서 판정한다.
  - 한국어: Kspon eval, 골드 ks-eval·ks-long 의 test 절반, 71631 창 test 절반
  - 영어: LibriSpeech test(utt, streams), 골드 ls-test·gs-test 의 test 절반
  - dev 세트는 튜닝에 쓰므로 판정 세트에서 뺀다. 튜닝에 쓴 데이터로 판정하면 결과가 낙관 쪽으로 치우치기 때문이다.
- AIHub 3 세트의 지연 결론은 P(h) 와 근사 EOU 로만 말하고, "단어 끝 기준 지연"이라고 부르지 않는다.

---

## 7. 평가 데이터 (mxc 위치, 규모, 학습 데이터와의 중복)

### 7.1 R1: 발화 단위 벤치마크

리더보드와 비교할 수 있는 세트다.

**오디오 규약(R1 전 세트 공통)**

- 앞 무음 0 s, 원 발화, **꼬리 1.5 s digital zero** 로 만든다. 1.5 s 는 H_eos(0.8 s) + 0.5 s 보다 길다(§4.5).
- 구현: `prepare-bench` 가 16 kHz wav(발화 + 0 꼬리)를 결과 루트 아래에 미리 렌더링한다. words 행은 그 파일 하나를 segments=[{offset_s: 0, silence_before_s: 0, dur_s: 렌더 길이}] 로 가리킨다.
  - `semcommit_eval` 은 5103556 판을 그대로 쓴다. 오디오는 `streams.assemble_stream` 이 조립한다.
  - 파일 끝 20 ms 가 0 이므로, K′ 패딩으로 늘어나는 꼬리도 0 이다. `silence_like` 는 edge RMS 가 1e-5 미만이면 0 을 돌려준다(`vapasr/data/streams.py:67-73`).
  - 정렬이 있는 세트(Kspon eval, LibriSpeech test, dev)는 원 manifest 의 앞 무음(offset_s)만큼 단어 시각을 당긴다(P0-0 어댑터).
- 이 규약을 새로 정한 이유(2026-09-29 mxc 확인):
  - single-turn v1 은 "앞 무음 없음, 꼬리 1 s digital zero"다(`experiments/eval_single_turn_asr.py:5`, `:108` 의 `np.pad`).
  - 프로젝트 manifest 의 utt 행에는 `silence_like` 앞 무음(중앙값 0.64 s)과 뒤 무음(0.65 s)이 붙어 있다. kspon-eval manifest 는 8.11 h 인데 음성은 6.07 h 다. librispeech-test utt 는 12.76 h 에 음성 10.75 h 다.
    - 여기에 K′ 패딩(`train_pad_K`: max(K0, int(t_last/0.08)+8+2))을 더해도 마지막 단어 끝 뒤 여유는 약 0.72–0.80 s 뿐이다. 그래서 H_eos 0.8 s endpoint 가 오디오 끝 전에 발동할 수 없다.
  - `assemble_stream` 으로 만든 stub 은 꼬리를 마지막 20 ms 를 타일링한 `silence_like` 로 채운다(`streams.py:84-86`).
    - 상담 세트 150 발화 표본 가운데 15 개는 마지막 20 ms 가 노이즈 바닥 + 12 dB 를 넘었다. 이런 발화는 꼬리 1 s 가 말소리 조각의 반복이 되어 인과 VAD 가 무음을 찾지 못한다.
  - 즉 이전 계획의 "원 발화 + 1 s 무음(single-turn v1 규약)" 문구는 세트마다 달랐고 사실과도 맞지 않았다.
- single-turn v1 과의 차이는 꼬리 길이(1.0 → 1.5 s)뿐이다. S0 수치를 v1 과 비교할 때 이 차이를 적는다.

| 세트 | mxc 위치 | 발화 | 음성(h) | SR | 단어 정렬 |
|---|---|---:|---:|---|---|
| Kspon eval_clean / eval_other | 프로젝트 manifest `/soundai/users/tskim/VAPKT-data/data/manifests/kspon-eval/streams.jsonl`(5,687 = 3,000 + 2,687; databricks 사본에서 eval_other 313 누락; manifest 전체 8.11 h 는 무음 포함). 벤치마크 사본 `/soundai/DB/Benchmark/tasks/ko_asr/data/kspon_speech_eval_16k/`(3,000 + 3,000) | 5,687 | 6.07 | 16 k | 있음(`align-asr-tn-v1/kspon-eval`, 스트림별 파일) |
| AIHub 상담 음성 | `/soundai/DB/Benchmark/slm_eval_kit/datalists/asr/ko/counsel_speech_8k.jsonl` → `tasks/ko_asr/data/consulting_speech_8k/상담_음성/Validation/...` | 3,000 | 3.93 | 8 k | 없음 |
| AIHub 한국어 강의 | `.../ko/korean_lecture_speech_16k.jsonl` → `korean_lecture_16k/korea_lecture/Validation/...` | 3,000 | 3.91 | 16 k | 없음 |
| AIHub 저음질 전화망 | `.../ko/lowquality_telephony_asr_8k.jsonl` → `low_quality_telephone_8k/01.데이터/2.Validation/...` | 3,000 | 3.82 | 8 k | 없음 |
| LibriSpeech test-clean / other | manifest `manifests/librispeech-test`(utt 5,559 = 2,620 + 2,939). 벤치마크 사본 `tasks/en_asr/data/librispeech/` | 5,559 | 10.75 | 16 k | 있음 |
| (선택) ESB = Open-ASR 계열 | `tasks/en_asr/data/esb_datasets_test_only_sorted/{ami,common_voice,earnings22,gigaspeech,spgispeech,tedlium,voxpopuli}` + datalists `asr_en_*.jsonl` | 아래 참고 | 약 183 | 혼합 | 없음 |
| (선택) AIHub 회의, CV15-ko | `ko/meeting_speech_16k.jsonl`(3,000), `ko/cv-15-ko.jsonl`(228) | | | | |

ESB 세트별 발화 수와 추정 시간(200 개 표본의 평균 길이 × 고유 발화 수):

| 세트 | 발화 | 추정 시간 | 비고 |
|---|---:|---:|---|
| AMI | 12,643 | 9.0 h | |
| CV | 16,334 | 25.7 h | 32–48 kHz |
| Earnings22 | 2,741 | 5.5 h | 16–24 kHz |
| GigaSpeech | 19,931 | 35.5 h | podcast·YouTube 원천. 학습의 yodas-en129(YouTube)와 겹칠 수 있음(미점검) |
| SPGISpeech | 39,341 | 100.9 h | |
| TED-LIUM | 1,155 | 약 2.6 h | datalist 는 2,310 행이지만 WavPath 고유값은 1,155 개다(모든 행이 두 번씩). YouTube 계열 원천과 겹칠 수 있음(미점검) |
| VoxPopuli | 1,842 | 4.6 h | **in-domain**: 학습에 voxpopuli-train 이 있다. 세션 ID 로 split 중복을 점검한 뒤 쓴다(P4) |
| 합계 | | 약 183 h | |

- 다른 ESB datalist 6 개는 WavPath 중복이 없었다(2026-09-29 확인).
- `prepare-bench` 는 WavPath 로 중복을 없앤 뒤 표본을 뽑는다(테스트 포함, §10.3).

참고:

- 8 kHz 세트는 `streams.load_utt_audio` 가 soxr 로 16 kHz 로 올린다(`vapasr/data/streams.py:38`). Qwen3-ASR 에도 같은 16 kHz 파형을 준다.
- AIHub 발화 길이는 p50 3.8–3.9 s, p95 9.6–11.8 s, 최대 18–21.6 s 다(mxc 측정).
  - 그래서 R1 에서는 발화 중간의 SEM 이 드물다. 많은 발화가 EOS 에서 세그먼트 1 개로 확정되며, 그런 발화에서는 S1+ ≈ S4a ≈ S5 다.
  - 그래서 R1 전체 수치는 주로 "디코더 효과"로 해석한다. SEM 트리거의 효과는 R1 의 "발화 중간 SEM" 부분집합과 R2 에서 본다(§12 S-1).

### 7.2 R2: 장문·연속 스트림 (트리거 선택이 실제로 결과를 가르는 곳)

**LibriSpeech test streams**

- `manifests/librispeech-test` 의 mode=stream 1,483 개다(test-clean 742 / other 741, 길이 p50 약 30 s, 합계 12.15 h).
- 발화 사이에 무음을 삽입해 만든 스트림이다. **삽입 무음이라 VAD 에 유리하다**는 점을 명시한다.

**골드 v1**

- 경로: `/soundai/users/tskim/VAPKT-data/data/semcommit/gold/v1/words-{ls-test,gs-test,ks-eval,ks-long}.jsonl` + `gold-*.jsonl`
- 규모: 599 스트림(ls-test 100, gs-test 79, ks-eval 300, ks-long 120), 경계 14,545, COMMIT 1,099([[output-semcommit-gold-v1]])
- **mxc 에서 그대로는 못 쓴다.** words 행이 rack4 경로(`/data4/…`, `/data5/LibriSpeech/…`)를 가리켜, 599 행 모두 첫 오디오 경로가 mxc 에 없다(2026-09-29 확인). P0-0 에서 경로를 바꾸고 PCM 해시로 rack4 원본과 대조한다. gs-test 는 `gold/v1/wav` 를 쓴다.
- **dev/test 분리**: `semcommit_gold.split_half`(sha1(id) 짝수 = dev, 홀수 = test, `experiments/semcommit_gold.py:139-141`)를 그대로 쓴다.
  - dev 절반은 레시피 v0.3 교사 임계값 튜닝에 이미 쓰였고, 이번 2-pass 튜닝에도 쓴다.
  - **판정·보고는 test 절반으로만 한다.**
- 골드 ls-test 는 LibriSpeech test-clean, ks-eval·ks-long 은 Kspon eval 발화에서 만들었다. 그래서 골드 dev 절반은 R1 Kspon eval·LibriSpeech test 와 R2 LS streams 에 같은 오디오로 들어 있다. 이 세트의 보고 수치는 **골드 dev 절반과 겹치는 발화·스트림을 포함한 값과 뺀 값을 함께** 낸다.

**한국어 자연 장문(aihub71631-dev 세션 창)**

2026-09-29 mxc 확인(`manifests/aihub71631-dev/streams.jsonl`, 부록 A `probe2.py`):

- utt 모드 66,365 발화, 68.31 h(음성 44.36 h)다.
- 각 발화는 세션 wav 채널(`…/Validation/01.원천데이터/VS_02.실외/<세션>.wav#ch0|ch1`)에서 `src_offset_s` 로 잘라낸 조각이다(정렬 `align-asr-tn-v1/aihub71631-dev`).
- 세션 185 개, 세션·채널 369 개다. 184 세션은 두 채널 모두 발화가 있다(2 채널 대화).
- 같은 채널 연속 발화 간격은 p50 1.04 s 이고, 72.4 % 가 3 s 미만이다. 같은 채널 안에서 겹치는(음수 간격) 쌍이 3,267 개다.
- ch0 의 양수 간격 33,232 개 가운데 16,482 개에 ch1 발화가 겹친다. 즉 간격의 절반에서 상대 화자가 말하고, 그 소리가 ch0 에 누설로 남는다.
  - 리뷰 표본(6 세션)에서 상대만 말하는 구간의 누설 레벨은 자기 음성보다 10–13 dB 낮았다. 두 채널 모두 발화가 없는 구간이 −18.5 dB, 자기 음성이 −10.1 dB 인 저 SNR 세션도 있었다(실외 녹음).

만드는 방법:

- **tune/test 분리**: 세션(파일 이름) 단위로 sha1(세션) 짝수는 tune, 홀수는 test 로 나눈다. 같은 세션의 두 채널은 같은 쪽에 둔다. 튜닝은 tune 창으로만, 판정·보고는 test 창으로만 한다.
- **창 자격**(세션·채널별 연속 발화 run):
  1. 같은 채널 연속 발화 간격이 0 이상 3 s 미만이다. 음수 간격(겹침)이 나오면 run 을 끊는다. 겹친 구간의 참조를 이어 붙이면 단어가 중복되거나 순서가 틀어지기 때문이다.
  2. 다른 채널 발화와 겹치는 시간이 창 길이의 5 % 미만이다(누설 제한).
  3. 창 길이는 30–60 s 이고, 창 경계는 발화 경계다. 긴 run 은 여러 창으로 나눈다.
- 2026-09-29 조회로는 조건 1 을 만족하는 30 s 이상 run 이 1,391 개, 조건 2 까지 만족하는 run 이 374 개였다(리뷰 측정은 정의가 조금 달라 약 430). 긴 run 에서 창이 여러 개 나올 수 있지만, **500 창에는 못 미칠 수 있다.** P0-2 에서 자격을 만족하는 창 수를 먼저 세고, 500 을 넘으면 sha1(seed:창 id)로 500 을 뽑는다. 모자라면 전부 쓰고 수를 보고한다. 제외한 run 수도 사유별로 보고한다.
- **오디오와 발화 경계**:
  - 창 오디오는 원 채널 파일의 한 구간으로 읽는다. words 행의 segments 는 창 전체를 가리키는 항목 1 개다(`src_offset_s` = 창 시작, `dur_s` = 창 길이). 그래야 발화 사이 간격이 원 녹음 그대로 남는다.
    - segments 를 발화별로 넣으면 안 된다. `assemble_stream` 이 발화 사이 간격을 `silence_like` 로 채워(`streams.py:79-83`) 자연 간격이 사라지기 때문이다.
  - 발화별 경계는 새 필드 `utt_bounds`(발화마다 `[시작, 끝]` 한 쌍, 창 기준 초)에 넣는다. S4a(oracle-utt), EOU, VAD 검증이 이 필드를 쓴다.
  - 단어 시각은 창 오프셋만큼 옮긴다.
- **창별 기록**: 누설 비율(다른 채널 발화가 겹친 시간 비율), SNR(자기 발화 구간 프레임 dB 중앙값 − 두 채널 모두 발화가 없는 구간 프레임 dB 중앙값). 결과는 SNR 3 분위로 층화해 보고한다.
- 대안(P4): 두 채널을 합성하고 두 채널 전사를 시간순으로 합친 참조를 쓰는 창. 화자 겹침 처리 규칙이 더 필요해서 이번에는 뒤로 미룬다.
- 위험: 자격 규칙으로 줄여도 창 안에 전사되지 않은 누설 음성이 남을 수 있다. 강한 full-context 디코더가 이를 전사하면 삽입 오류처럼 보인다. 누설 비율 층화 결과를 함께 보고한다.
- 과거 D2 에서 71631 오디오 조립 버그(`src_offset_s` 누락)가 있었다([[output-stage2-d2-final-eval]]). 창 빌더는 `src_offset_s` 를 반드시 반영하고 테스트로 확인한다.

**한국어 긴 발화**

- ks-long(골드 120, test 절반만 판정)

### 7.3 튜닝(dev)과 판정(test)의 분리

- 하이퍼파라미터(ctx, min_seg, snap/nominal, H, VAD 임계값, W_left, bias)는 다음 세트에서만 고른다.
  - kspon-dev(2,545 발화, 음성 3.94 h)
  - librispeech-dev(utt 5,567 발화, 음성 10.51 h; streams 1,470 개는 VAD 검증용)
  - aihub71631-dev 창의 tune 절반(세션 sha1 짝수)
  - 골드의 dev 절반(sha1(id) 짝수)
- 판정·보고 세트: R1 test 세트, LS test streams, 71631 창 test 절반, 골드 test 절반.
- dev 세트의 수치는 참고로만 보고하고, 어떤 판정에도 쓰지 않는다.
- AIHub 3 세트에는 dev 가 없으므로 튜닝에 쓰지 않고 전량 test 로 쓴다.
- 격자 결과는 전부 보고하고, "권장 운영점"만 dev 에서 고른다.

### 7.4 학습 스냅샷(`speechlm-all19-v035/snapshots/20260929-0928`)과의 중복 점검

**스냅샷 구성**

- main 2,380 + short 951 words 파일(958 MB)이다.
- 원천:
  - LibriSpeech-960(train), swbd-train, voxpopuli-train, yodas-en129
  - speechlm 19 원천. 실제 경로 기준으로 000004·05·08·09·11·12·13·14·15·16·17·21·22·23·24·25·26·27·29 다.
  - summary 의 "?" 1,274 파트는 원천 표기만 빠진 것이다.

**학습 init 모델(hf-E2)의 데이터**

- C2 에서 시작해 8 코퍼스로 학습했다: librispeech-960, swbd-train, voxpopuli-train, yodas-en129, kspon-full, nikl-1000, aihub71631-train, aihub-bc-train([[output-stage2-d2-final-eval]]).
- 모두 train 계열 manifest 이고, 평가용 dev/test/eval manifest(kspon-eval, librispeech-test, aihub71631-dev)는 별도 디렉터리다.
- kspon-full dataset.json 은 splits {train: 619,932}, subsets train-01…05 로 eval 이 없다(mxc 확인).

**Kspon eval, LibriSpeech test, gs-test**

- 보호 인덱스 `semcommit-work/heldout-index-20260925-v1`(PCM 해시 19,437 구간, 오류 0)과 speechlm 전량 후보를 대조한 결과 **파형·경로 충돌이 0 건**이다.
  - 근거: `approval-audit-20260925-v1/summary.json`, [[output-speechlm-semcommit-approval-audit-20260925]]
  - 보호 범위: `pcm_scope` = librispeech-dev, librispeech-test, kspon-dev, kspon-eval, gold-gs-test 다(PCM 해시). aihub71631-dev 는 `path_only_scope`, 즉 **경로로만** 보호된다(`summary.json`).
- LibriSpeech-960, swbd, voxpopuli, yodas 파트는 train split 에서만 왔다.

**aihub71631-dev**

- 2026-09-29 mxc 조회(부록 A `probe4.py`): dev manifest 세션 185 개와 train manifest(aihub71631-train) 세션 757 개 사이에 **공유 세션 0**, 파일 이름의 화자 코드(dev 198 명, train 376 명) 사이에 **공유 화자 0** 이다.
- 보호 인덱스는 경로만 본다. 그래서 파형 수준 중복은 확인하지 않았다. 원천이 같은 데이터셋의 Validation 이라 파형 중복 가능성은 낮지만, 필요하면 P0-1 에서 PCM 해시 범위에 넣는다.

**AIHub 3 세트**

벤치마크는 각 데이터셋의 **Validation** 경로를 쓴다. 학습 쪽은 다음과 같다.

1. speechlm Arrow 의 split 필드:
   - 000029(상담): 1,583,840 행 전부 train
   - 000033(한국어 강의, 벤치마크 korea_lecture 와 같은 NIA_3 lecture 계열): 2,353,407 행 전부 train
   - 000023(대학 강의, NIA24/137, 벤치마크 강의와 다른 데이터셋): 전부 train
   - 000011(저음질 전화망): train 4,397,622 + **validation 39,729**
2. QC 와 후보 빌더는 `split == "train"` 만 통과시킨다.
   - `experiments/speechlm_semcommit_qc.py:119-120, 208-209` (작업 트리, 미커밋)
   - `experiments/semcommit_build_speechlm_candidates.py:87-90`
   - 그래서 000011 의 validation 행은 학습 후보에서 빠진다.
3. 주의할 점: split 판정 함수 `split_from` 은 경로에 `/validation/`·`/valid/`·`/dev/` 가 있어야 validation 으로 본다(`build_speechlm_single_speaker_arrow.py:115-121`).
   - 벤치마크 전화망처럼 `2.Validation/` 형식이면 train 으로 잘못 분류될 수 있다.
   - Arrow 의 000011 경로는 `NIA23/007_LowQualityTelephone/{Training,Validation}/Wavs/…` 형식이라 해당되지 않아 보인다.
   - 하지만 규칙만 믿지 않고, 스냅샷 words 의 실제 경로·세션·전사를 직접 대조했다(아래, 2026-09-29 mxc 읽기 전용 실행).

**스냅샷 대조 결과**

대상은 words 3,331 파일, 1,599,656 행이다(부록 A 의 `overlap_probe.py`, `counsel_probe.py`).

- 학습 segments 경로에 Validation/valid/dev/test/eval 문자열이 **0 건**이다. KsponSpeech eval, LibriSpeech test/dev 경로도 **0 건**이다.
- 원천별 구간 수: 000011 80,814, 000023 35,630, 000029 55,726, 000016 25,268 등.
  - **000033(벤치마크 한국어 강의와 같은 NIA_3 lecture)은 스냅샷에 없다.**
  - 000023(NIA24 대학 강의)은 다른 데이터셋이다.
- 상담:
  - 벤치마크 3,000 발화 = 260 세션(D,J,S), 학습 000029 = 29,638 세션이다.
  - **공유 세션 0, 같은 (D,J,S,번호) 키 0** 이다. S 번호만 보면 181/256 이 겹치지만, D·J 가 달라 ID 공간이 우연히 일치한 것이다.
  - 공백·구두점을 뺀 전사가 정확히 같은 경우: 8 자 이상 132 건(000029 80, 000027 26, 000011 12 등), **20 자 이상 34 건**.
  - 시나리오형 상담 문장(예: "약관 동의 여부 체크를 먼저 확인하겠습니다")이 다른 화자의 녹음으로 학습에 있다.
  - 이 34 발화(1.1 %)를 "text-seen" 부분집합으로 표시하고, 포함·제외 CER 을 함께 보고한다.
- 전화망:
  - 학습 쪽 파일명(`NIA23/007_LowQualityTelephone/Training/Wavs/S000506_d10_000111.wav`)과 벤치마크 원 배치(`2.Validation/D04/J18/S023423/0005.wav`)는 번호 체계가 달라 화자를 대조할 수 없다.
  - 전사 일치는 8 자 이상 186 건, 12 자 이상 80 건, 20 자 이상 1 건이다. 대부분 "학습지원센터입니다 무엇을 도와드릴까요", "네 감사합니다 좋은 하루 되세요" 같은 콜센터 정형 문구다.
  - 데이터셋 수준의 split(Validation 제외)은 지켜졌다.
- 한국어 강의: 전사 일치 1 건("말씀드리겠습니다")
- Kspon: clean 1 건, other 0 건

**ESB(선택 세트)**

- VoxPopuli 는 학습에 voxpopuli-train 이 있으므로 **in-domain** 으로 표시한다. ESB test 파일 이름의 세션 ID(예: `20180613-0900-PLENARY-24`)를 voxpopuli-train manifest 와 대조한 뒤 쓴다(P4, CPU).
- GigaSpeech·TED-LIUM 은 podcast·YouTube 계열 원천이라 학습의 yodas-en129(YouTube)와 겹칠 수 있다. 이번에는 점검하지 않았으므로 결과에 그 가능성을 표시한다.

**판단**

- 학습 스냅샷에 평가 발화 자체가 들어갔다는 증거는 없다.
- 남은 위험은 세 가지다.
  1. 상담·전화망은 **같은 데이터셋의 다른 split** 이다. 화자·도메인·정형 문구를 공유할 수 있어 1차 모델이 도메인 이점을 가진다. Kspon(kspon-full), 71631(aihub71631-train)도 같은 코퍼스의 train 으로 학습했다. 도메인 밖 대조는 한국어 강의뿐이다. 그래서 KO 결과는 도메인 안/밖으로 층화해 보고한다(§12).
  2. 전화망은 번호 체계가 달라 파형 수준의 중복이 아직 확정되지 않았다.
     - P0-1 에서 AIHub 3 세트를 보호 인덱스에 추가한다. 도구는 `experiments/speechlm_semcommit_heldout_index.py`(작업 트리, 미커밋)이고, 같은 16 kHz 디코더로 계산한 PCM 해시를 쓴다.
     - 그다음 `experiments/speechlm_semcommit_approval_audit.py`(작업 트리, 미커밋) 대조를 다시 돌려 확정한다(CPU 작업).
     - 두 도구는 mxc 에 배포돼 있다(2026-09-29 `ls` 확인).
  3. ESB 의 VoxPopuli(in-domain)와 GigaSpeech·TED-LIUM(yodas 중복 가능성).

**참고**

- 벤치마크 Validation 폴더에는 세션의 발화가 전부 있다(상담 S00012652 약 40 발화, 강의 S001029 666 조각, 전화망 S023423 약 36 발화).
- 그래서 세션을 이어 붙인 장문 스트림을 만들 수 있다. 다만 발화 사이 간격은 삽입 무음이 된다(P4).

---

## 8. 어블레이션 격자

### 8.1 주 격자

| 축 | 값 | 비고 |
|---|---|---|
| 1차 δ | 2, 4, 8 | 학습한 δ 범위 안(config delays [2,3,4,6,8]). δ4 가 중심 |
| 트리거 | SEM, SEM∪VAD(H 0.5), VAD(H 0.3 / 0.5 / 0.8), S2m(H 0.5), fixed(W 2 / 4 / 8), oracle-utt, oracle-gold(골드만), offline | |
| 2차 디코더 | Q17(주), Q06, V8-dual | V8-dual 은 GPU 비용 없음 |
| ctx | iso, text, prefix(W_left 15 s) | δ4·SEM 에서는 셋 다. 나머지 트리거는 SEM dev 에서 고른 ctx 하나를 똑같이 쓴다 |

실행 범위:

- **δ4**: Q17 과 Q06 모두 위 트리거 전부. 조합 수는 SEM × 3 ctx + 나머지 트리거 10 개 = 13 이다(골드 세트는 oracle-gold 를 더해 14).
- **δ2·δ8**: Q17, dev 최선 ctx 로 δ 에 따라 결과가 달라지는 트리거(SEM, SEM∪VAD)만 돌린다. fixed·S2m 은 δ4 에서만 돌린다.
- **δ 와 무관한 2차 결과는 공유한다.**
  - VAD, oracle-utt, oracle-gold, offline 은 절단과 2차 입력이 오디오와 앞 세그먼트의 2차 결과로만 정해진다. 그래서 δ 가 달라도 2차 입력이 같다.
  - 2차 결과는 입력 지문(오디오 구간, prefill, context, 디코더, 생성 설정)으로 캐시한다. 같은 입력이면 δ 사이에서 자동으로 재사용된다.
  - 앞 세그먼트가 fallback 된 스트림은 prefill 이 1차 텍스트(δ 의존)가 되어 입력이 달라진다. 이런 경우는 캐시가 맞지 않아 다시 디코드한다.
  - fallback·guard·늦은 단어 처리는 δ 별 1차 텍스트로 따로 한다.

### 8.2 SEM 발화 민감도(δ4, R2 + 골드 + dev)

- bias b ∈ {−2, −1, 0, +1, +2} 와 θ ∈ {0.2, 0.35, 0.5} 를 스윕한다.
- 한 번 인코드해서 여러 설정 행을 함께 디코드한다(`semcommit_eval.py:165-166, 194-235`).
- b 가 커지면 세그먼트가 짧아지고, 조기 commit 이 늘고, 확정 지연이 준다.
- 그릴 곡선: 확정 지연 p50/p90 대 최종 WER, 세그먼트 길이, PCR.
- 운영점 선택은 dev(골드 dev 절반, 71631 tune 절반)에서 하고, 곡선은 test 에서도 그린다.

### 8.3 세그먼트 규칙(δ4, SEM, Q17 에서 dev 로 고르고 test 로 보고)

기본값에서 한 축씩 바꾸는 방식으로 돌린다(기본값 포함 13 조합).

| 축 | 값(굵게 = 기본) |
|---|---|
| ctx | iso / text / **prefix** |
| 절단 | **snap** / nominal / avail |
| R_extra | **0** / 0.16 / 0.32 s |
| min_seg | 0 / **0.8** / 1.5 s |
| W_left | 5 / **15** / 30 s |
| rollback r | **0** / 1 |
| pad_l | **0** / 0.16 s(경계 중복 제거 켬) |

### 8.4 규모 제한

- ESB 는 δ4 에서 {S0, S1+ Q17 prefix, S2 H0.5 Q17, S5 Q17} 만 돌린다.

---

## 9. 표본 수와 통계

**표본**

- R1: Kspon 5,687, AIHub 3 × 3,000, LibriSpeech 5,559 전량(test)
  - 발화 독립을 가정하면 3,000 발화·CER 약 12 % 에서 paired bootstrap 95 % CI 반폭은 0.3–0.5 %p(절대)로 예상된다. 아래 군집 bootstrap 을 쓰면 더 넓어진다.
- R2(test): LibriSpeech streams 1,483 전량, 골드 test 절반(약 300 스트림), aihub71631-dev 창 test 절반(§7.2, 목표 250 창)
- dev: kspon-dev, librispeech-dev, 골드 dev 절반, 71631 창 tune 절반
- ESB(선택): 세트당 최대 1,000 고유 발화(sha1 결정적 표본, 7 세트 약 13 h)
  - AMI, SPGI, GigaSpeech, CV 는 표본만 쓴다. 전량 수치와 섞지 않는다.

**유의성**

- **군집 단위 paired bootstrap** 을 2,000 회 돌린다(seed 고정). 발화(스트림)를 독립으로 보면 같은 세션·화자의 상관을 무시해 CI 가 좁아지기 때문이다.
  - 군집: AIHub 상담·전화망·강의는 경로의 세션, 71631 창은 세션, LibriSpeech(utt·streams·골드 ls-test)는 화자-chapter, gs-test 는 원천 녹음 id 다.
  - Kspon eval 은 화자 정보가 없어 발화 단위로 둔다. 골드 ks-eval·ks-long 도 발화 단위다.
- ΔWER 의 CI 가 0 을 포함하지 않으면 유의하다고 본다. p 값은 bootstrap 분포에서 0 을 넘는 비율로 낸다.
- 지연은 단어 가중 분위수에 같은 군집 bootstrap CI 를 붙인다.

**다중 비교**

- 성공 주장 하나하나(기준 × 언어 풀)를 한 family 로 본다. 이 family 에 Holm–Bonferroni(FWER 0.05)를 적용한다.
- 여러 조건을 **모두** 만족해야 하는 기준(예: KO 의 주 CER 과 보조 CER, S-1 의 조건 (a)와 (b))은 각 조건을 보정 없이 α 로 검정한다. 모든 조건을 요구하는 intersection–union 검정이라 보정이 필요 없다.
- **계층적 검정 순서**:
  1. 게이트 1 = S-1(정확도). S-1 을 통과한 언어 풀만 다음 게이트로 간다.
  2. 게이트 2 = S-2(SEM 트리거 비용)와 S-3(우측 문맥 대조). 둘은 서로 독립인 질문이므로 같은 게이트에 두고 Holm 으로 보정한다.
- 나머지 비교(H3–H6, 어블레이션)는 기술 통계로만 보고한다.

**주 판정 쌍(미리 정함)**

1. S1+ Q17 prefix δ4 대 S0 δ4(S-1 a)
2. 같은 S1+ 의 회수율 D = (S0 − S1+) − 0.5·(S0 − S5 Q17)(S-1 b, 회수율이 정의되는 세트만)
3. 같은 S1+ 대 S2 H0.5 Q17 prefix, 그리고 H 스윕 보간점(S-2)
4. 같은 S1+ 대 S0 δ8(= V8-dual WER, S-3)

---

## 10. 구현 단계(파일, 함수, 데이터 흐름)

학습 코드와 기존 평가 코드는 바꾸지 않는다. 새 파일만 추가하고 `semcommit_eval.py` 의 함수는 import 해서 재사용한다.

- experiments 는 패키지가 아니다. 그래서 `tests/test_commit_metrics.py:19` 처럼 importlib spec 으로 로드하거나, 공용 함수만 `vapasr/hf/twopass.py` 로 모은다.
- **버전 고정**: 1차 디코드와 commit 지표는 커밋 `5103556` 판 `experiments/semcommit_eval.py` 와 `vapasr/hf/commit_metrics.py` 로 한다(창 없는 지연 지표·`hyp.word_k` 포함, T3). 디코드 규약은 88220f7 과 같고 채점만 늘었다.
  - 가설 단어별 청크(word_k)는 스트림 jsonl 의 `hyp.word_k` 를 쓴다(§4.1). 옛 스트림은 `--score-only` backfill 로 채운다.
  - 이어하기 지문(CODE_FILES sha256)은 `5103556` 판의 값이다. 88220f7 판으로 시작한 디코드 출력에 이어 붙이지 않는다(새 `--out`).
- **mxc 배포 상태**:
  - 2026-09-29 오전 조회 때는 `semcommit_eval.py`·`semcommit_build_words.py` 가 mxc 에 없었다. 같은 날 오후 골드 평가를 위해 88220f7 판 `semcommit_eval.py` 를 배포했다.
  - `5103556` 판(`semcommit_eval.py`·`commit_metrics.py`)은 진행 중인 골드 평가가 끝난 뒤 배포한다. `semcommit_build_words.py` 는 아직 없다(P0-0).

### 10.1 새 파일(예정)

#### `vapasr/hf/twopass.py`

모델·GPU 와 무관한 순수 함수 모음이며, 단위 테스트 대상이다.

- 가설 단어:
  - 가설 단어별 청크는 새로 만들지 않고 `5103556` 의 `split_hyp`(→ `hyp.word_k`)를 import 해서 쓴다.
- 세그먼트 생성:
  - `sem_segments(emitted, K, delta, sem_id, is_text, word_start, decode, wav=None, cut='snap', min_seg=0.8, max_seg=20.0, r_extra=0.0) -> list[Segment]`
  - `causal_endpoints(wav, sr, H, hop_ms=20, on_rel=12, off_rel=8, floor_init_dbfs=-60.0) -> list[(t_sil_start, t_trigger)]`
  - `emission_gap_endpoints(word_chunks, K, delta, H) -> list[(cut, t_trigger)]`(S2m)
  - `vad_segments(...)`, `fixed_segments(...)`, `oracle_utt_segments(wrow)`(`utt_bounds` 가 있으면 그것), `oracle_gold_segments(wrow, gold, delta)`, `offline_segment(dur)`, `union_triggers(soft, hard)`
- 절단과 배정:
  - `snap_cut(wav, lo, hi, floor_db, margin_db=8, hop_ms=20)`: 최소점이 바닥 + margin 이하일 때만 그 점, 아니면 None(nominal 사용)
  - `assign_first_pass(word_chunks, segments, delta)`: 트리거 시점 텍스트, 늦은 단어, direct-final 표시(§4.6)
- 문맥과 조립:
  - `context_window(segments, j, W_left, finals)`: (시작 세그먼트 q, 오디오 창 시작 b_{q−1}, prefill = 세그먼트 q..j−1 의 확정 텍스트)를 돌려준다(§4.3).
  - `dedupe_boundary(prev_final_tail, second_words, max_words=3)`
  - `assemble_final(segments, texts, fallback, late_words)`
  - `guard_hallucination(n1, second_text, seg_s, lang)`: §4.2 의 길이 기준과 반복 판정
- 지표:
  - `latency_rows(wrow, segments, word_chunks, delta, cost_model=None)`: L_disp/L_trig/L_final(algorithmic·재귀 큐)/P/EOU, direct-final, EOS 대체 여부를 계산한다. 정렬이 없으면 P 와 근사 EOU 만 계산한다.
  - `revision_counts(first_norm, second_norm, ref_norm)`: 어휘·표면 번복, fixed/broken 등을 센다.
  - `digit_latin_mask(ref, hyp)`: KO 보조 지표(§6.1)
  - `cluster_bootstrap(rows_a, rows_b, key, cluster, n=2000, seed=0)`, `holm(pvalues)`
- `Segment` dataclass 필드: stream_id, j, trigger(sem|vad|gap|fixed|eos|forced|oracle_utt|oracle_gold|offline|merged), hard(bool), t_trig, t_dispatch, cut0, cut1, win0, win1, first_word_idx, late_word_idx, sem_category, eos_fallback(bool)

#### `vapasr/hf/second_pass.py`

GPU 를 쓰는 디코더 어댑터다.

- `QwenAsrDecoder(model_dir, dtype=bf16, batch=32)`: **S5 를 포함한 모든 Qwen 조건이 이 래퍼 하나를 쓴다.**
  - `qwen_asr.Qwen3ASRModel.from_pretrained(..., backend transformers)` 의 processor 와 model 을 쓴다.
  - `decode(batch=[(wav, lang, context, prefill)]) -> [(text, n_gen, n_prefill_tok, truncated, rep_fixed)]`
  - 프롬프트는 `_build_text_prompt(context, lang)` 뒤에 prefill 을 붙인 것이고, greedy `generate` 로 생성한다.
  - `transcribe()` 와 같은 처리를 직접 재현한다: 0.5 s 미만 입력의 뒤 0 패딩, `parse_asr_output`(+ `detect_and_fix_repetitions`).
  - max_new_tokens = max(512, ⌈12·창 길이(s)⌉ + 32). EOS 없이 상한에 닿으면 truncated = True.
- `DualDelayDecoder(first_pass_streams_d8)`: GPU 를 쓰지 않는다. δ8 1차 기록에서 구간 단어를 뽑는다.
- `LiteralVapAsrDecoder`: 스모크 전용(b-literal).

#### `experiments/semcommit_twopass_eval.py` (CLI)

**`adapt-align`**(P0-0)

```
adapt-align --manifest manifests/<set> --align align-asr-tn-v1/<set> --tokenizer <dir> --render-root <results>/r1-audio/<set> --tail-silence 1.5 --out words.jsonl labels.jsonl
```

- mxc 의 `align-asr-tn-v1/<set>` 는 스트림별 `.jsonl`(kspon-eval 은 `ks-eval_clean-utt-*.jsonl` 등 5,692 개 파일)과 `align.json` 구조다.
- 기존 `experiments/semcommit_build_words.py` 는 rack4 배치(`<manifests>/<set>/aligned-manifest.jsonl.gz` + `original/<set>/streams.jsonl`)만 읽는다(`:106-108`). 그래서 mxc 에서는 그대로 쓸 수 없다.
- 이 명령은 스트림별 정렬 파일을 words.jsonl 형식으로 바꾼다. R1 utt 세트는 §7.1 규약대로 오디오를 렌더링하고 단어 시각을 앞 무음만큼 당긴다. streams 모드(LibriSpeech test/dev streams)는 manifest 의 오디오 조립을 그대로 쓴다.
- 결과를 `semcommit_build_words.py` 로 rack4 에서 만든 words 표본과 비교하는 테스트를 둔다(같은 id 의 단어·시각·토큰 일치).

**`prepare-longform-71631`**

```
prepare-longform-71631 --manifest manifests/aihub71631-dev --align align-asr-tn-v1/aihub71631-dev --win 30:60 --max-gap 3 --max-xtalk 0.05 --n 500 --out words.jsonl labels.jsonl windows.json
```

- §7.2 의 장문 창을 만든다: 자격 판정, 세션 sha1 tune/test 분리, `utt_bounds`, 누설 비율·SNR 기록, 자격별 제외 수 집계.

**`prepare-bench`**

```
prepare-bench --datalist <slm_eval_kit jsonl> --set <name> --lang <Korean|English> --tail-silence 1.5 --render-root <results>/r1-audio/<set> --out words.jsonl labels.jsonl
```

- WavPath 로 중복을 없앤 뒤(TED-LIUM 2,310 → 1,155) 처리한다. 제거 수를 기록한다.
- §7.1 규약으로 오디오를 렌더링하고, semcommit_eval 이 읽는 words/labels 의 **stub** 을 만든다.
  - words: id, set, lang, K, duration_s(렌더 길이), text, segments=[{path: 렌더 wav, offset_s: 0, silence_before_s: 0, dur_s}], tokens=[], words=[]
  - labels: {id, candidates: []}
- words 가 비면 `eval_audio_len` 은 패딩 없이 K0 를 쓰고(`commit_metrics.py:57-63`), ASR 은 `wrow.text` 로 채점된다.
- 참조 단어가 없어서 commit 텍스트 지표(P/R, PCR)는 의미가 없다. **stub 세트는 정렬 세트와 다른 `semcommit_eval` 실행(별도 `--out`)으로 돌리고**, commit 지표는 정렬 세트 실행에서만 가져온다(§6.5).
- 같은 명령이 보호 인덱스 입력용 streams.jsonl 도 쓴다(P0-1).

**1차 디코드**

- 5103556 판 `experiments/semcommit_eval.py` 를 수정 없이 쓴다.
- 세트 × δ × bias/θ 조합마다 `<out>.streams.jsonl`(emitted, K, hyp, trace, metrics)과 report.json 을 낸다.

**`segment`**

```
segment --words --streams <1차 jsonl> --config bias=0 --trigger sem|vad|gap|fixed|oracle-utt|oracle-gold|offline|sem+vad [--H --W --cut --min-seg --max-seg --r-extra] --out segments.jsonl
```

- 오디오가 필요한 트리거는 `semcommit_eval.load_stream_audio`(`:148`)로 1차와 같은 파형을 다시 조립한다(같은 seed, 같은 K′).
- 1차 jsonl 의 words_digest 로 같은 words 행인지 확인한다.

**`second-pass`**

```
second-pass --segments --decoder q17|q06|v8dual|literal --ctx iso|text|prefix [--W-left] --cache <dir> --gpu --batch --out hyp2.jsonl
```

- text·prefix 모드는 스트림 안에서 순서에 의존한다. 그래서 세그먼트 번호 j 별로 wave 를 나눠 여러 스트림을 한 배치로 묶는다.
- `--cache`: 입력 지문(오디오 구간 sha, prefill, context, 디코더, 생성 설정) → 출력. δ 사이 공유(§8.1)와 이어하기에 쓴다.
- 이어하기 지문: 디코더 가중치의 크기·mtime, config sha256, segments 파일 sha256, 코드 sha256, 디코드 설정(semcommit_eval 의 decode_config 와 같은 방식).

**`bench-latency`**

```
bench-latency --decoder --buckets --n 500 --gpu
```

- cost_model.json 을 낸다. C(len, n_tok) 의 선형 적합과 잔차 분위수를 담는다.

**`score`**

```
score --words --labels --streams --segments --hyp2 [--gold] [--split dev|test] [--normalizer asr-tn|whisper] [--cost-model] --out report.json
```

- 세트·언어·설정별로 계산한다.
  - WER/CER(`commit_metrics.asr_counts`, whisper 정규화), KO 숫자·로마자 보조 지표
  - 지연 5 종(재귀 큐 포함), direct-final·EOS 대체·max_seg 비율, 비용, 번복, 3-way, 군집 bootstrap CI, Holm 보정
  - 골드 dev 절반 겹침 포함·제외 두 벌
  - 권장 운영점 표(dev 에서만 선택)
- markdown 보고서도 함께 낸다.

#### 그 밖의 파일

- `tests/test_semcommit_twopass.py`: §10.3 의 테스트.
- `slurm/semcommit-twopass-eval.sbatch`, `slurm/submit-semcommit-twopass-eval.sh`
  - **제출은 사용자가 한다.**
  - apex 권장(비선점), `--exclude` 로 hpc-52·hpc-4 제외, 컨테이너 기본 python 사용 금지, env python 절대경로 사용.
- (평가 후) `wiki/outputs/output-semcommit-twopass-eval.md` 와 로그 샤드.

### 10.2 단계

- **P0-0 (CPU) mxc 실행 전제 만들기**
  - (a) 5103556 판 `semcommit_eval.py`·`commit_metrics.py` 와 새 파일을 mxc 에 배포한다(azcopy 또는 mx.sh). 오디오 경로가 다르면 `--path-remap` 을 쓴다.
  - (b) `adapt-align` 으로 `align-asr-tn-v1` 스트림별 파일을 words.jsonl 로 바꾼다.
  - (c) 골드 words 의 rack4 경로를 mxc 경로로 바꾸고, PCM 해시로 rack4 원본과 대조한다. gs-test 는 `gold/v1/wav` 를 쓴다.
  - (d) 1차 처리량 45 audio-h/GPU-h(§11)는 다른 디코더(`eval_single_turn_asr`)의 값이다. `semcommit_eval` 의 처리량과 메모리는 P0-3 에서 실측해 §11 을 다시 추정한다.
- **P0-1 (CPU)**
  - 경로·세션·전사 대조는 끝났다(§7.4).
  - 남은 일: AIHub 3 세트 9,000 발화를 보호 인덱스에 PCM 해시로 추가한다. `speechlm_semcommit_heldout_index.py` 입력용 streams.jsonl 은 `prepare-bench` 가 함께 쓴다.
  - 그다음 `speechlm_semcommit_approval_audit.py` 로 speechlm QC 통과 후보와 파형 충돌을 다시 확인한다.
  - 충돌한 발화는 평가에서 빼고 그 목록을 보존한다.
- **P0-2 (CPU)**
  - ESB 시간은 표본으로 추정을 마쳤다(§7.1, TED-LIUM 중복 반영).
  - `prepare-longform-71631` 로 장문 창을 만들고 **자격을 만족하는 창 수를 먼저 센다**(§7.2).
  - ESB-lite 표본 목록(WavPath 중복 제거 뒤 sha1 결정적)을 만든다.
- **P0-3 (GPU 1 장, 1 시간 이하) 스모크**
  - 세트당 20 발화를 δ4 로 1차 디코드 → SEM/VAD 세그먼트 → Q06/Q17 iso/prefix 2차 → score 까지 돌린다.
  - `--verify`(`semcommit_eval.py` 의 기능)로 1차 decode_rows 와 stream_decode 의 토큰을 대조한다.
  - `semcommit_eval` 과 qwen_asr(transformers 백엔드, 배치 32, bf16)의 처리량과 메모리를 재서 예산을 보정한다.
  - b-literal 50 스트림 스모크도 여기서 한다.
- **P0-4 (GPU 1 장, 약 1 시간)**: Q06·Q17 을 batch 1 로 `bench-latency`.
- **P0-5 (GPU 1 장, 1 시간 이하) S0 대 S5 사전 점검**
  - 판정 세트마다 sha1 결정적 300 발화(R2 는 50 스트림·창)에서 S0 δ4 와 S5 Q06/Q17 을 비교한다.
  - 세트·시스템별 숫자·로마자 포함 가설 비율도 잰다.
  - S5 ≥ S0 인 세트를 미리 알리는 용도다. 회수율 정의 여부의 최종 판단은 본 run 의 test 결과로 한다(§12). 이 점검은 하이퍼파라미터를 바꾸는 데 쓰지 않는다.
- **P1 (GPU)**: 1차 run. 전 세트 × δ {2,4,8} × bias 0. δ4 의 bias/θ 스윕은 dev·골드·R2 에서만.
- **P2 (GPU)**: 2차 run. 먼저 dev 에서 VAD 임계값(CPU)과 §8.3 격자를 돌려 운영점을 고른다. 그다음 §8.1 격자를 test 에 돌린다.
- **P3 (CPU)**: score, 군집 bootstrap, Holm, Pareto 그림, 보고서.
- **P4 (선택)**: ESB-lite(VoxPopuli 세션 대조 포함), Nemotron/Whisper 2차, AIHub 세션 연속 장문 스트림, 71631 두 채널 합성 창.

### 10.3 테스트 계획(로컬 `vapasr-local` 환경, GPU 없이)

- `sem_segments`: 합성 방출열(텍스트, SEM, NEXT, flush 토큰 포함)에서 다음을 확인한다.
  - 세그먼트 경계와 텍스트 배정
  - EOS 꼬리 처리, 그리고 endpoint 가 없을 때 오디오 끝 대체와 지연 포함
  - min_seg 병합(SEM 만)과 max_seg 강제 확정
  - **hard 트리거는 min_seg 와 무관하게 즉시 확정**: VAD 뒤 0.4 s 꼬리("네")는 다음 트리거를 기다리지 않는다. 스트림 끝의 짧은 꼬리도 EOS 에서 확정된다.
  - guard 가 켜져 있으면 빈 SEM 세그먼트가 없음
  - mid_word 배정
- `snap_cut`: 합성 파형(톤 + 무음 구간)에서 다음을 확인한다.
  - 무음 안에서 절단함
  - 탐색 하한이 nominal 이라 확정 단어의 마지막 80 ms 안에서 자르지 않음
  - 최소점이 바닥 + 8 dB 보다 높으면(무음 없음) None → nominal
  - 받은 오디오 밖으로 나가지 않음(인과성)
- `causal_endpoints`:
  - 앞부분만 준 파형과 전체 파형에서, 앞부분 구간의 endpoint 가 같다(인과성 성질 테스트).
  - 첫 0.5 s 가 음성이어도 바닥이 −60 dBFS 에서 시작해 이후 무음에서 endpoint 가 발동함.
  - 꼬리가 말소리 조각 반복(`silence_like` 타일링)이면 endpoint 가 발동하지 않고, EOS 가 오디오 끝으로 대체됨.
  - H 경계값에서의 동작.
- 인과성과 늦은 단어(§4.6, §6.2 예 3): δ8, H 0.3 에서 b 직전 단어가 늦은 단어로 분류되고, 2차 성공 시 direct-final, fallback 시 방출 시각에 붙는다. fallback·guard 판단이 t_trig 뒤 방출을 쓰지 않는다.
- `fixed_segments`: δ2·δ4·δ6·δ8 모두에서 절단 구간이 비지 않고 폭이 0.48 s 이며, 구간 안 단어의 이상적 방출이 T_j 이하다(§6.2 예 4).
- `oracle_*`, `union_triggers`: 손으로 계산한 시각과 대조한다. S4b 트리거 = emit_time(ref_chunk(end_time, δ)).
- `context_window`:
  - prefill 텍스트가 정확히 오디오 창 [b_{q−1}, b_{j−1}] 의 세그먼트 텍스트다(한 세그먼트 어긋남 없음).
  - 현재 세그먼트가 20 s(max_seg) 여도 창이 정의된다(W_left 15 s 는 좌측 문맥만 제한).
  - endpoint 경계를 넘지 않고, q = j 이면 prefill 이 비어 iso 와 같다.
- `dedupe_boundary`: pad_l 0.16 s 의 iso 와 text 모드에서 앞 세그먼트 마지막 단어가 반복되면 잘라내고 센다.
- `guard_hallucination`: n1 = 0 인 2 s VAD 세그먼트에서 2차 5 단어는 통과하고, 반복 루프는 막는다.
- 지연:
  - §6.2 손 계산 예 1–4 와 일치한다(예 2 의 재귀 큐 1.96 s 포함).
  - R_extra 가 한 번만 더해진다.
  - flush 방출은 K·0.08 로 계산된다.
  - seg(i) 는 시각 기준이며, 최종 전사에서 삭제된 참조 단어도 지연이 있다.
  - 정렬 없는 행은 P 만 계산되고, P 는 0 미만이 없으며 direct-final 이 따로 세진다.
- `QwenAsrDecoder` 의 순수 부분(가짜 processor·model): 0.4 s 입력이 0.5 s 로 0 패딩됨, max_new_tokens 규칙, n_gen == 상한이면 truncated, `parse_asr_output` 경로 적용.
- `revision_counts`: fixed/broken 등의 손 계산과 일치하고, 정규화 뒤 같으면 번복 0 이다.
- `digit_latin_mask`, 빈 출력 fallback, text 문맥의 앞머리 중복 제거.
- `prepare-bench`: WavPath 중복 제거(2 행 → 1 행), 렌더 꼬리가 0 이고 `assemble_stream` 으로 조립해도 꼬리가 0 으로 남음.
- `prepare-longform-71631`: 합성 manifest 에서 `src_offset_s` 반영, 음수 간격에서 run 이 끊김, 다른 채널 겹침 5 % 이상 제외, 세션 sha1 분리가 결정적이고 두 채널이 같은 쪽, `utt_bounds` 가 창 기준 시각.
- stub 분리: stub 행을 정렬 행과 한 보고서에 합치면 KO PCR 이 오염되는 것을 재현하고, 별도 실행에서는 정렬 세트 PCR 이 그대로인지 확인한다.
- `cluster_bootstrap`, `holm`: 고정 seed 에서 손 계산 예와 일치한다.
- CLI 종단 테스트:
  - 가짜 1차 streams 와 가짜 디코더(`tests/test_commit_metrics.py` 의 `test_run_loop_with_fake_model` 방식)로 segment → second-pass → score 를 끝까지 돌린다.
  - 이어하기, 지문이 다를 때 거부하는 동작, 캐시 재사용을 확인한다.
- `prepare-bench` stub 을 5103556 판 `semcommit_eval.run` 이 받아들이는지 가짜 모델로 확인한다.
- `whisper_normalizer` 정규화 결과를 고정 예로 확인한다.
- 실행 명령:

  ```
  /Users/taesookim/anaconda3/envs/vapasr-local/bin/python -m pytest -q tests/test_semcommit_twopass.py tests/test_commit_metrics.py -p no:cacheprovider --import-mode=importlib --rootdir=.
  ```

---

## 11. 계산 예산(mxc H200)

**처리량 가정**

- 1차: single-turn v1 전량 평가에서 rank0 가 16.8 audio-h(LibriSpeech test + Kspon eval, δ2·δ4 절반)를 1,356 s 에 처리했다. 그래서 약 45 audio-h / GPU-h 로 둔다(fp32, batch 128; `results/single-turn-35000-d2-d4-v1/rank0.log`).
  - **이 값은 `eval_single_turn_asr`(다른 디코더)의 것이다.** `semcommit_eval` 은 mxc 에서 잰 적이 없다. P0-3 에서 실측해 바꾼다.
- 2차 Qwen3-ASR 의 처리량도 아직 재지 않았다. Q17 150× 실시간, Q06 300× 실시간(transformers, bf16, batch 32)을 가정한다.

**오디오 양(§7.1 규약: 앞 무음 0, 꼬리 1.5 s)**

| 묶음 | 구성 | 시간 |
|---|---|---:|
| R1 test | Kspon 6.07 + 2.37(꼬리), AIHub 11.66 + 3.75, LibriSpeech 10.75 + 2.32 | 약 36.9 h |
| R2 | LS streams 12.15, 골드 1.7, 71631 창 약 6(tune + test) | 약 20 h |
| dev(utt) | kspon-dev 3.94 + 1.06, librispeech-dev 10.51 + 2.32 | 약 17.8 h |

- 이전 판의 "dev 약 7 h"는 틀렸다. mxc manifest 기준 kspon-dev 4.86 h, librispeech-dev utt 12.52 h 로 약 17.4 h(앞뒤 무음 포함)다.

**2차 조합과 재처리 배율(ρ) 가정**

- R1(짧은 발화, 스트림 = 발화 1 개):
  - δ4 Q17 조합: SEM iso 1.1, text 1.1, prefix 1.3, S1+ prefix 1.3, VAD 3 개 × 1.2, S2m 1.3, fixed W2/W4/W8 2.1/1.7/1.0, oracle-utt 1.0, offline 1.0. ρ 합계 16.5.
  - δ4 Q06 도 같은 조합이다.
  - δ2·δ8 은 SEM·S1+(prefix) 2 조합씩이다. VAD·oracle·offline 은 δ4 결과를 공유한다.
- R2(장문):
  - δ4 Q17 조합: SEM iso 1.1, text 1.1, prefix 4, S1+ prefix 2, VAD 3 개 × 1.1, S2m 1.1, fixed W2/W4/W8 8.5/4.75/2.9, oracle-utt 1.0, offline 1.0. ρ 합계 약 30.8. 골드에는 oracle-gold(ρ 4)를 더한다.
  - δ2·δ8 은 SEM prefix(4)·S1+ prefix(2)다.
- dev 튜닝: §8.3 의 13 조합, 평균 ρ 약 1.5. dev utt 17.8 h + 71631 tune 약 3 h + 골드 dev 약 0.85 h = 약 21.7 h.

| 단계 | 계산 | GPU-h(추정) |
|---|---|---:|
| P0-3 스모크, 처리량 보정, b-literal 스모크 | | 2.0 |
| P0-4 지연 벤치(Q06·Q17, batch 1, bucket 7 × 500) | | 1.0 |
| P0-5 S0 대 S5 사전 점검(세트당 300 발화) | 약 3 h × (1차 + Q06 + Q17) | 0.5 |
| P1 1차: (R1 36.9 + R2 20 + dev 17.8) h × δ 3 개 | 약 224 audio-h ÷ 45 | 5.0 |
| P1 bias/θ 스윕(δ4, 8 행, 골드 + 71631 200 창 + LS streams 200 = 약 6 h; batch 16 이라 처리량 절반 가정) | 약 48 row-h ÷ 22.5 | 2.2 |
| P2 2차 R1: δ4 Q17 16.5 × 36.9 = 609, δ4 Q06 609(비용 ½), δ2·δ8 2 × 2.6 × 36.9 = 192 | 609 ÷ 150 + 609 ÷ 300 + 192 ÷ 150 | 7.4 |
| P2 2차 R2: δ4 Q17 30.8 × 20 + 골드 oracle-gold 7 = 623, δ4 Q06 623, δ2·δ8 2 × 6 × 20 = 240 | 623 ÷ 150 + 623 ÷ 300 + 240 ÷ 150 | 7.9 |
| P2 dev 튜닝(§8.3, 13 조합) | 13 × 21.7 × 1.5 ≈ 423 ÷ 150 | 2.8 |
| P2 스윕 행의 2차(Q17 prefix, 8 행 × 6 h × ρ 3) | 144 ÷ 150 | 1.0 |
| 소계(필수) | | **약 30 GPU-h** |
| P4 ESB-lite(선택, 7 세트 × 최대 1,000 발화 = 약 13 h: 1차 δ4 + 2차 3 조합) | | 0.7 |

- δ 와 무관한 트리거(VAD 3 개, oracle-utt, oracle-gold, offline)의 2차 결과를 δ 사이에서 공유해 절감한 양이 위 표에 들어 있다. 공유하지 않으면 δ2·δ8 에서 Q17 약 640 audio-h(약 +4.3 GPU-h)가 더 든다.
- 처리량 가정이 틀릴 여유를 ×1.5 로 두면 **상한은 45 GPU-h** 다. 이전 판의 22 GPU-h(상한 35)는 dev 시간, 조합 수, dev 튜닝 격자의 2차 비용을 빠뜨린 값이었다.
- **보정 뒤 재판정 규칙**: P0-3·P0-4 실측으로 표를 다시 계산한다. 필수 소계가 45 GPU-h 를 넘으면 아래 순서로 줄이고, 줄인 내용을 보고서에 적는다.
  1. R2 의 fixed W2(prefix ρ 8.5)를 뺀다.
  2. Q06 을 SEM·S1+·S5 조합으로만 줄인다.
  3. librispeech-dev 를 sha1 결정적 2,000 발화 표본으로 줄인다.

**운영**

- apex 1 노드 8 GPU 로 약 5–6 시간 걸린다. P0 는 GPU 1 장이면 된다.
- 학습 run(semcommit-v035-snap0929-d8)이 끝난 뒤 시작한다.
- 제출은 사용자가 한다.
- 산출물 루트: `/soundai/users/tskim/VAPKT-data/results/semcommit-twopass-v1/<set>/<run>`(R1 렌더 오디오는 `…/r1-audio/<set>/`, 약 4 GB). 기존 run 디렉터리와 /lustre 에는 쓰지 않는다.

---

## 12. 성공 기준과 판정 규칙

모든 판정은 test 세트로만 한다(§7.3). 검정 방법은 §9(군집 bootstrap, Holm, 계층 순서)를 따른다.

**S-1 정확도(게이트 1)**

- 판정 세트:
  - R2 test: LS streams, 골드 test 절반(ls-test, gs-test / ks-eval, ks-long), 71631 창 test 절반
  - R1 test 세트의 **"발화 중간 SEM" 부분집합**: S1+ 가 EOS 세그먼트 전에 SEM 트리거로 확정한 세그먼트가 1 개 이상인 발화. 같은 1차 run 에서 정하므로 S0 도 같은 부분집합에서 비교한다.
- 조건 (a): S1+(δ4, Q17, prefix) 대 S0(δ4)의 상대 WER/CER 점추정이 −10 % 이하이고, 차가 유의하다.
- 조건 (b): 회수율이 정의되는 세트, 즉 S0 − S5 Q17 이 유의하게 양수인 세트에서 D = (S0 − S1+) − 0.5·(S0 − S5 Q17) 의 CI 하한이 0 보다 크다.
  - S5 가 S0 보다 유의하게 낫지 않은 세트는 회수율을 "정의 안 됨"으로 보고하고, 그 세트에는 (a) 만 적용한다.
  - 이유: 1차 모델은 상담(000029)·전화망(000011)·Kspon·71631 의 같은 데이터셋 train 으로 학습해 도메인 이점이 있고, KO 숫자 표기 차이는 Q17 에만 벌점으로 작용한다. 그래서 S5 ≥ S0 인 세트가 있을 수 있고, 그러면 우변이 음수가 되어 (b) 가 자동으로 충족되거나 의미를 잃는다.
- 풀과 세트: KO 풀·EN 풀 각각에서 (a)(b)를 판정한다. 그리고 풀 안 **모든 세트에서 방향(S1+ < S0)이 같아야** 한다(세트별 유의는 요구하지 않음). 세트별 수치를 모두 보고한다.
- KO: 주 CER 과 숫자·로마자 보조 CER(§6.1) **둘 다에서** (a)(b)를 만족해야 한다. 제외·마스킹한 발화 수를 표에 적는다.
- KO 결과는 도메인 안(Kspon, 상담, 전화망, 71631)과 도메인 밖(강의)으로 층화해 함께 보고한다.
- 함께 보고(판정에 쓰지 않음):
  - R1 전체 수치: "디코더 효과"로 해석한다. S1+ 대 S4a·S5(같은 Q17)의 차를 "SEM 분할 비용"으로 따로 제시한다.
  - 발화 중간 SEM 으로 확정된 단어의 비율(R1·R2 세트별)

**S-2 SEM 트리거의 정확도 비용(게이트 2)**

- 대상 쌍: S1+(δ4, Q17, prefix, H 0.5) 대 S2(H 0.5, 같은 디코더·ctx)
- 확정 지연은 구성상 S1+ ≤ S2 다(algorithmic, §6.2). 그래서 검정하지 않고 L_final p50·p90 차와 군집 bootstrap CI 를 효과 크기로만 보고한다.
- 검정(비열등성): 상대 WER 증가 (S1+ − S2) / S2 의 단측 CI 상한이 +2 % 보다 작다.
- 지연 맞춤 비교(보조): S2 의 H ∈ {0.3, 0.5, 0.8}(+ dev 로 고른 임계값) 결과에서 L_final p50 을 H 에 대해 선형 보간한다. S1+ 의 p50 과 같은 점의 WER 을 구해 S1+ 와 비교한다. S1+ 의 p50 이 S2 범위 밖이면 "맞춤 불가"로 보고한다.
- 판정 세트: 정렬·골드 test 세트(LibriSpeech test utt·streams, Kspon eval, 골드 test 절반, 71631 창 test 절반). 결론은 자연 장문(71631 test, ks-long)을 우선한다.
- S2 가 허수아비가 아닌지 함께 확인한다: 세트별 S2 의 max_seg 강제 확정 비율과 endpoint 미발동 비율을 보고한다. 강제 확정 비율이 10 % 를 넘는 세트는 "VAD 대조군 부적합"으로 표시하고, 그 세트에서는 S2m 을 대조군으로 같은 판정을 한다.

**S-3 우측 문맥 대조(게이트 2)**

- S1+ Q17 의 WER 이 S0-δ8 보다 유의하게 낮아야 한다.
  - S0-δ8 의 WER 은 같은 트리거의 V8-dual 과 정의상 같고, V8-dual 의 algorithmic 확정 시각은 S1+ 이상이다(§5.2). 그래서 이 비교는 "확정 지연이 같거나 더 긴, δ8 우측 문맥 대조군"을 이기는지 보는 것이다.
  - 그렇지 않으면 "이득은 우측 문맥 효과로 설명된다"고 결론짓는다.
- 함께 보고:
  - L_final–WER Pareto 그림: S0 δ2/4/8 점, S1·S1+·S2·S3 의 2-pass 점, V8-dual 점
  - Q06 대 Q17 2-pass(디코더 크기 효과, 기술 통계)
- 이전 판의 "같은 L_final p50 근방의 S0-δ8" 비교는 성립하지 않았다. S0-δ8 의 L_final p50 은 약 0.7 s 인데, 2-pass 의 L_final 은 세그먼트 대기 때문에 초 단위이고, 학습된 최대 δ 가 8 이라 지연을 맞출 S0 가 없기 때문이다.

**S-4 번복 품질(기술 통계 기준)**

- fixed / broken ≥ 3
- 어휘 번복률 ≤ 1차 WER 의 1.5 배
- r = 0 에서 확정 후 수정 0

**S-5 비용**

- Q17 2차의 batch-1 p90 C ≤ 300 ms
- 단일 스트림 RTF_2nd ≤ 0.1
- 재처리 배율은 합격 기준에서 뺀다. prefix 는 설정에 따라 구성상 3 을 넘는다(§6.3). 대신 W_left 격자(5/15/30 s)의 배율과 정확도를 함께 보고하고, 권장 운영점의 배율을 적는다.
- 이 기준을 넘으면 Q06 으로 같은 판정을 다시 한다.

**최종 판정**

| 결과 | 결론 |
|---|---|
| S-1, S-2, S-3 모두 만족 | "SEM 트리거 2-pass 채택 후보". 실시간 구현과 vLLM 류 가속 검토로 넘어간다 |
| S-1, S-3 만족, S-2 불만족 | "2-pass 는 유효하지만 SEM 트리거를 더하면 정확도를 잃는다". 확정 트리거는 VAD 로 두고, SEM 은 downstream LLM prefill 신호로만 쓴다 |
| S-1 만족, S-3 불만족 | "이득이 우측 문맥 효과로 설명된다". δ·지연 설계(δ8 확정 등)를 먼저 검토한다 |
| S-1 불만족 | 폐기하고 δ·데이터 쪽 개선에 집중한다 |

- AIHub 3 세트의 지연 결론은 P(h) 와 근사 EOU 로만 말한다(단어 정렬 없음).

---

## 13. 위험과 대응

1. **표기 불일치**
   - 문제: Qwen3-ASR 은 written form(구두점, 대소문자, KO 아라비아 숫자·로마자, EN 숫자)을 내지만 참조는 lexical 이다. AIHub 참조는 숫자를 한글로 읽어 적는다.
   - 대응: 채점 정규화를 똑같이 적용한다. P0-5 에서 숫자·로마자 가설 비율을 먼저 잰다. KO 판정은 주 CER 과 보조 CER 을 모두 요구한다. EN 은 Whisper 정규화 WER 을 보조로 둔다. 표시 번복과 어휘 번복을 나눠 센다.
   - 교체 뒤 표시 스타일이 세그먼트마다 바뀌는 UX 문제는 이번 지표 밖이다(기록만 한다).
2. **짧은 세그먼트 품질 저하**
   - 문제: 한국어 대답어나 조각 commit 은 짧고, 0.5 s 미만은 0 으로 패딩된다(래퍼가 직접 재현).
   - 대응: SEM 에는 min_seg 를 쓰고, 짧은 hard 세그먼트는 prefix 문맥으로 디코드한다. 세그먼트 길이 bucket 별 WER 을 보고한다.
3. **조기·단어 중간 commit 의 절단 오류**
   - 대응: snap 절단(하한 nominal, 무음 확인), 경계 ±1 단어 층화, rollback r = 1 대조.
4. **text 문맥 복사·환각 삽입**
   - 대응: 앞머리 중복 제거, 오디오 길이 기반 guard·반복 가드, 삽입률 별도 보고. guard 를 껐을 때의 결과도 보고한다.
5. **Q06 오류 상관**
   - 문제: 우리 thinker 는 Q06 에서 시작해 미세조정됐다.
   - 대응: Q17 을 주 디코더로 쓴다.
6. **오염**
   - 경로·세션 수준의 중복은 없음을 확인했다(§7.4). 71631-dev 도 train 과 세션·화자가 겹치지 않는다.
   - 남은 것:
     - 전화망의 파형 수준 확인(P0-1 PCM 해시)
     - 71631-dev 는 보호 인덱스가 경로만 본다
     - 상담 text-seen 34 발화(별도 보고)
     - split 판정 규칙이 `2.Validation` 형식을 놓칠 가능성(이번 스냅샷에는 해당 없음)
     - ESB VoxPopuli(in-domain), GigaSpeech·TED-LIUM(yodas 중복 가능성)
   - Qwen3-ASR 학습 데이터는 공개되지 않아 공개 벤치마크 오염을 배제할 수 없다. 그래서 2-pass 이득을 "S5 대비 회수율"로도 표현해 절대치에 덜 의존하게 한다. 단 S5 가 S0 보다 낫지 않은 세트에서는 회수율을 정의하지 않는다.
   - 우리 1차 모델은 000011(전화망), 000029(상담), kspon-full, aihub71631-train 으로 학습했다. 그래서 해당 벤치마크에서 도메인 이점이 있다.
   - 한국어 강의(000033)는 학습에 없으므로 강의 세트는 도메인 밖 대조가 된다.
7. **R2 삽입 무음이 VAD 에 유리함(LibriSpeech streams)**
   - 대응: 자연 장문(aihub71631-dev 창, ks-long)을 함께 보고하고, 결론은 자연 장문을 우선한다.
8. **71631 장문 창의 교란**
   - 문제: 2 채널 대화라 상대 화자 누설이 남고, 실외 녹음이라 SNR 이 낮은 세션이 있다. 누설 음성은 강한 디코더에게 삽입 오류로 잡히고, 에너지 VAD 의 endpoint 를 깨뜨린다.
   - 대응: 다른 채널 겹침 5 % 미만 창만 쓴다. 창별 누설·SNR 을 기록해 층화 보고한다. 모델 기반 endpoint(S2m)를 대조군에 넣는다.
9. **VAD 대조군의 품질**
   - 문제: 새로 만드는 인과 에너지 VAD 가 endpoint 를 놓치면 S2 가 max_seg 강제 확정으로 퇴화하고, S-2 가 저절로 유리해진다.
   - 대응: dev 에서 endpoint 재현율·정밀도·지연을 재서 임계값을 정한다. S2m 을 정식 변형으로 둔다. 강제 확정 비율이 높은 세트는 판정에서 표시한다(§12 S-2).
10. **평가 오디오 규약**
    - R1 전 세트를 한 규약(앞 0 s, 꼬리 1.5 s digital zero)으로 렌더링한다(§7.1). single-turn v1 과는 꼬리 길이만 다르다.
    - R2 는 원 manifest 조립(LS streams 는 `silence_like` 삽입 무음, 71631 창은 원 녹음 구간)을 따른다. 세트 사이 규약 차이는 결과표에 적는다.
11. **2차 연산 지연**
    - 문제: transformers generate 는 토큰당 지연이 커서 compute-inclusive L_final 을 키운다. 세그먼트 사이 직렬 의존도 있다. vLLM 은 서버에 설치할 수 없다.
    - 대응: algorithmic 과 compute-inclusive(재귀 큐 식)를 나눠 보고하고, 가속은 후속 과제로 둔다.
12. **2차 출력 잘림**
    - 대응: max_new_tokens 를 창 길이에 비례시키고(최소 512), 잘림 수를 보고한다. 잘림이 S5 에서 많으면 오프라인 기준이 삭제 오류로 오염된 것이므로 회수율 해석에서 표시한다.
13. **학습 중인 체크포인트**
    - final 이전에는 스모크만 한다.
    - final 이 나오면 checkpoint 지문(semcommit_eval decode_config)을 모든 결과에 남긴다.
14. **mxc 운영**
    - 로그인 노드가 불안정하다(2026-09-29 12:00–13:08 접속 거부 관측).
    - 모든 단계를 이어하기 가능하게 만든다. P0-1 같은 CPU 작업도 체크포인트 파일을 남긴다.
    - 평가 코드가 아직 배포되지 않았다(P0-0).
15. **되먹임이 없어서 생기는 한계**
    - 1차 모델은 교정 전 가설을 조건으로 계속 디코드한다.
    - 교정 텍스트로 1차 KV 를 다시 채우는 변형은 후속 과제다.

---

## 부록 A. 이번 조사에 쓴 읽기 전용 스크립트

scratchpad 위치: `/private/tmp/claude-501/-Users-taesookim-Desktop-VAPKT/49169b81-fc5f-478d-acc2-17601e549321/scratchpad/`

| 스크립트 | 내용 |
|---|---|
| `overlap_probe.py` | 스냅샷 words 전체와 벤치마크 KO 세트 대조(경로 split 문자열, S 번호, 정규화 전사) |
| `counsel_probe.py` | 상담 (D,J,S,번호) 키·세션 대조, 전화망 12 자 이상 전사 대조 |
| `split_probe.py` | speechlm Arrow 000011/23/29/33 의 split 값과 시간 |
| `bench_probe.py`, `esb_probe.py` | 벤치마크 세트의 발화 수, SR, 시간(KO 는 전량, EN 은 200 개 표본) |
| `twopass_review/probe1.py` | 골드 words 경로 존재, TED-LIUM WavPath 중복, kspon/librispeech/71631 manifest 시간·앞뒤 무음 |
| `twopass_review/probe2.py` | aihub71631-dev 세션·채널 수, 같은 채널 간격·겹침, 다른 채널 겹침, 창 자격 run 수 |
| `twopass_review/probe3.py`, `probe3b.py` | ESB datalist 중복, AIHub 3 세트 표본의 처음 0.5 s·마지막 20 ms 에너지 |
| `twopass_review/probe4.py` | aihub71631 dev/train 세션·화자 코드 중복 |

실행 방법:

```
mx.sh '/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python -' < <script>
```

- 모두 읽기 전용이고 파일을 쓰지 않는다.
- 그 밖에 mxc 에서 `ls`·`md5sum`·`grep` 으로 배포 상태(§10), qwen_asr 설정(§4.3), single-turn δ6 결과와 δ8 취소 기록(§1.2), 보호 인덱스 범위(§7.4)를 확인했다.
- **출처 보존 주의**: 이 스크립트들은 세션 scratchpad(`/private/tmp`)에만 있고 저장소에는 없다.
  - §7 의 수치를 재현하려면 P0-1 전에 `raw/sources/experiments/` 에 옮겨 보존할지 정해야 한다.
  - 보존하기 전까지 이 수치는 "2026-09-29 mxc 읽기 전용 실행에서 관측" 수준의 근거다.
  - 리뷰가 보고한 수치 가운데 이번에 다시 재지 않은 것(71631 누설 레벨 −21.8 대 −34.5 dB 등, Kspon eval Qwen 토큰 속도 p50 4.8 tokens/s)은 "리뷰 측정"으로 표시했다.

---

## 부록 B. 리뷰 반영 기록(2026-09-29)

리뷰 2 건의 37 항목을 코드와 mxc 에서 직접 확인했다. 모두 실제 문제였고(일부는 두 리뷰가 같은 문제를 지적), 아래처럼 고쳤다.

| 영역 | 무엇이 틀렸나 | 고친 곳 |
|---|---|---|
| 사용자 지시 해석 | 리뷰 반영판은 "3번 제외"를 T3(commit latency) 제외로 읽어 T3 코드를 쓰지 않게 고쳤다. 3번은 논문 인사이트 3번(영어 레이블 불균형)이어서 다시 정정했다: T3 는 `5103556` 으로 커밋됐고 이 계획이 그 판을 쓴다 | 머리말, §4.1, §6.2, §6.6, §10 |
| mxc 실행 전제 | `semcommit_eval.py`·`semcommit_build_words.py` 가 mxc 에 없다. build_words 는 rack4 배치만 읽는다. 골드 words 599 행이 rack4 경로다 | §7.2, §10 P0-0, `adapt-align` |
| dev/test 분리 | 71631-dev·골드를 튜닝과 판정에 함께 썼다 | §6.6, §7.2, §7.3, §9, §12 |
| H0/S-3 | V8-dual WER ≡ S0-δ8 인데 따로 셌다. "같은 L_final 근방 S0-δ8" 은 존재하지 않는다. δ6 수치와 δ8 취소를 빠뜨렸다 | §1.2, §2, §3, §5.2, §12 |
| S5 와 회수율 | S5 가 S0 보다 낫다는 보장이 없는데 격차 회수율을 썼다 | §2, §5.1, §10 P0-5, §12 |
| KO 숫자·로마자 | 주 지표가 2-pass 에 체계적으로 불리하고, 보조 지표는 판정에 쓰이지 않았다 | §6.1, §12 |
| R1 에서의 S-1 | 짧은 발화에서는 SEM 효과를 거의 재지 못한다 | §1.2, §7.1, §12 |
| min_seg | hard endpoint 까지 보류했다 | §4.5 |
| 인과성 | VAD·오라클 트리거가 아직 방출되지 않은 단어를 썼다 | §4.6, §6.2 |
| 지연 정의 | 직렬 큐 없음, R_extra 이중 가산, P 정의 모순, seg(i) 정렬 의존, timing_metrics 동치 오류, S4b 트리거가 실제 방출보다 0–0.08 s 이름 | §4.2, §4.6, §6.2 |
| EOS | oracle EOS 를 지연에서 빼 생존 편향이 생겼다. endpoint 가 오디오 끝 전에 발동할 수 없는 세트가 있었다 | §4.5, §7.1 |
| R1 오디오 규약 | 문서 안 모순, 세트마다 다른 앞뒤 무음, `silence_like` 꼬리 | §7.1, §4.6 VAD |
| 71631 창 | 창 = 세그먼트 1 개라 S4a 가 퇴화했다. 겹침·누설·SNR 을 다루지 않았다. 500 창이 가능한지 확인하지 않았다 | §7.2 |
| VAD 대조군 | 검증 절차가 없었다 | §3 S2m, §4.6, §12 S-2 |
| guard | 1차 단어 0–1 개에서 트리거 비교가 편향됐다 | §4.2 |
| 절단·중복 | snap 창 하한이 단어 안이었다. pad_l 0.16 의 iso 중복, fixed 구간이 δ ≥ 6 에서 비었다 | §4.2, §4.3, §4.6 |
| prefix 창 | max_seg > W_left 에서 정의가 안 됐다. prefill 이 한 세그먼트 어긋났다. 배율 식이 정의와 달랐다 | §4.3, §6.3, §12 S-5 |
| 2차 호출 경로 | 0.5 s 패딩·반복 수정 누락, max_new_tokens 256 | §4.3, §10.1 |
| 통계 | 다중 비교 미보정, 발화 단위 bootstrap | §9 |
| H2/S-2 | 지연 부분은 구성상 반증할 수 없다 | §2, §12 |
| 예산 | dev 시간·조합 수·dev 격자 2차 비용 누락. δ 공유 절감도 빠졌다 | §8.1, §11 |
| 데이터 사실 | TED-LIUM 중복(2,310 → 1,155), Kspon 시간 표기 | §7.1 |
| 누수·인용 | VoxPopuli in-domain, yodas 중복 가능성, 71631 보호 범위, 미커밋 파일 인용, thinker 가중치 표현, H1 대상 | §1.2, §2, §7.4, 머리말 |
| stub 세트 | KO commit 지표를 오염시킨다 | §6.5, §10.1 |
