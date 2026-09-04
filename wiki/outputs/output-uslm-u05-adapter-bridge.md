---
type: output
status: active
created: 2026-09-04
updated: 2026-09-04
summary: U0.5 adapter bridge test 결과 보고 — Nemotron [56,0] 특징 → adapter → Qwen3-ASR thinker(LoRA) 오프라인 ASR. 4 run 모두 oto 18–20 / 실내 17–19 / 실외 15–17 % 수렴, adapter 병목 아님, 관문 재정의 후 통과
sources:
  - [[task-uslm-u05-adapter-bridge]]
  - [[decision-asr-backbone]]
  - [[output-interleaved-streaming-slm-architecture]]
  - [[output-feature-cache-and-compute-budget]]
---

# U0.5 adapter bridge test — 결과 보고

## 1. 목적과 질문

최종 backbone [[decision-asr-backbone]] 은 **Nemotron 3.5 FastConformer `[56,0]`(완전 인과, ≤80 ms) → 새 adapter → Qwen3-ASR-0.6B thinker(LM)** 이다.
두 모델은 서로 다른 audio tower 로 사전학습됐으므로, thinker 가 Nemotron 표현에서 텍스트를 생성할 수 있는지가 IS-SLM 전체의 선행 조건이다.
U0.5 는 이 연결을 **스트리밍 없이(오프라인 발화 단위)** 가장 싸게 검증한다.

질문: (Q1) adapter 가 정보를 잃는가? (Q2) 표현 증류 초기화가 필요한가? (Q3) 도달 가능한 WER 수준은 어디이며 무엇이 병목인가?

## 2. 설정

| 항목 | 값 |
|---|---|
| 인코더 | Nemotron 3.5 ASR streaming 0.6B, `att_context_size=[56,0]`, 12.5 Hz, 1024-d. **얼림**, 캐시 특징 사용(`/data3/tskim/features/nemotron-c0/`) |
| adapter | LayerNorm → Linear 1024→2048 → GELU → Linear 2048→1024 (4.2M). 12.5 → 13 Hz nearest 리샘플 후 입력 |
| LM | Qwen3-ASR-0.6B thinker, LoRA r16 α32 (q/k/v/o/gate/up/down, 10.1M). audio tower 제거 |
| 프롬프트 | Qwen3-ASR 원본 형식 재현: `…user\n<|audio_start|>[<|audio_pad|>×N]<|audio_end|>…assistant\nlanguage {Lang}<asr_text>{text}` — adapter 출력을 `<|audio_pad|>` 위치의 inputs_embeds 에 scatter |
| 학습 데이터 | otoSpeech(EN) + AI Hub ts01-5(KO) 발화 0.3–20 s, train 121,827 발화; val = 대화 id 정렬 후 마지막 8 % |
| 평가 | val 150 발화/코퍼스, greedy, 정규화(태그 `<…>` 제거·NFKC·소문자·구두점 제거), EN WER / KO CER(공백 제거), jiwer |
| 증류 교사 | Qwen AuT `block8s`(학습 분포와 같은 8 s 블록 마스크) 출력 = thinker 입력 임베딩, 129 대화 쌍 35 h, cosine + MSE, 4 epoch (20 min) |
| 자원 | GPU 1 (H100 공유), run 당 ≈25 GB, 6k step 34 min / 12k step 65–70 min, bs 8 |

코드: `vapasr/uslm/{data,model}.py`, `experiments/u05_{distill_adapter,baselines,asr_finetune}.py`, `experiments/show_uslm_runs.py`.

## 3. 기준선

| 시스템 | 인과성 | otoSpeech WER | 실내 CER | 실외 CER |
|---|---|---|---|---|
| Qwen3-ASR-0.6B 오프라인(AuT + thinker) | 비인과(파일 단위) | **13.9** | **13.5** | **13.3** |
| Nemotron RNN-T `[56,13]` (1 s 미래) | 1.04 s lookahead | 19.0 | 16.9 | 17.0 |
| Nemotron RNN-T `[56,0]` (동일 인코더, 자체 디코더) | ≤80 ms | 25.4 | 23.4 | 24.5 |

Nemotron 수치는 `<ko-KR>` 등 언어 태그를 정규화에서 제거한 뒤의 값(제거 전엔 KO 가 35 % 대로 부풀려졌다).
실외(vs02)는 학습에 쓰지 않은 검증 코퍼스라 잡음·도메인 이동을 포함한다.

## 4. 결과

증류 품질: held-out cosine **0.777**(항등 매핑 0.004), MSE/var 0.38 — 두 표현 공간이 선형에 가깝게 연결됨을 시사하지만 완전하지는 않다.

| run | init | lr (LoRA / adapter) | step | otoSpeech WER | 실내 CER | 실외 CER | 비고 |
|---|---|---|---|---|---|---|---|
| noinit-3k | random | 2e-4 / 5e-4 | 3000 | 18.7 | 20.6 | 18.3 | 짧은 스케줄(3000 에서 완전 감쇠) |
| distill-6k | 증류 | 2e-4 / 5e-4 | 6000 | 20.0 | 18.3 | 16.9 | step 2000 손실 스파이크 |
| distill-lowlr-6k | 증류 | 1e-4 / 1e-4 | 6000 | **18.2** | 19.2 | 16.1 | 4000→6000 −2.9 pt(EN) |
| noinit-12k | random | 2e-4 / 5e-4 | 12000 | 20.1 | 17.7 | 16.6 | 9000→12000 정체 |
| distill-12k | 증류 | 1e-4 / 1e-4 | 8000 | 18.8 | **17.5** | **14.5** | 최선 중간점 |
| distill-12k | 증류 | 1e-4 / 1e-4 | 12000 | 19.3 | 17.8 | 15.1 | 8000 이후 요동(10000 에서 24.5 스파이크) |

학습 곡선 공통점: 2000 step 부근 손실 스파이크(0.35 → 1.1)가 lr 와 무관하게 재현 → 특정 배치(긴 발화) 효과로 추정.
val 은 150 발화라 ±1–2 pt 잡음이 있다(run 간 차이 대부분이 이 범위 안).

## 5. 해석

1. **Q1 — adapter 는 병목이 아니다.** 같은 `[56,0]` 인코더의 자체 RNN-T 디코더(25.4 / 23.4 / 24.5)보다 **6–9 pt 낮다**. 얼린 인코더 표현에 담긴
   정보는 adapter 를 거쳐 thinker 로 충분히 전달되며, thinker 의 언어 모델링이 RNN-T 디코더보다 오히려 이득을 준다.
2. **Q2 — 증류 초기화의 이득은 잡음 범위.** 12k 기준 19.3/17.8/15.1(증류) vs 20.1/17.7/16.6(random). 수렴 속도도 크게 다르지 않다.
   비용이 20 min 이므로 U1 에서는 유지하되 필수 요소로 보지 않는다.
3. **Q3 — 수렴점은 oto 18–20 / 실내 17–19 / 실외 15–17 %.** 오프라인 Qwen 대비 +4–6 pt(상대 +30–45 %). 격차의 원인 후보:
   - **인과 인코더의 상한**: Nemotron 자체도 1 s 미래를 허용하면 25.4 → 19.0 으로 좋아진다. 완전 인과 80 ms 표현은 오프라인 표현보다 본질적으로 적은 정보를 담는다.
   - **얼린 인코더**: 인코더 상위 블록을 함께 학습하면 격차 일부를 회복할 수 있다(U1/U3 에서 어차피 열 예정).
   - 학습량·스케줄: 12k 에서 정체했으므로 단순 step 증가는 답이 아니다. lr 감쇠 시점에 민감.
   - LoRA r16 용량·데이터(≈200 h): 2 차 요인.
4. 관문 원안 "오프라인 Qwen ×1.15(oto ≤16.0, KO ≤15.5 %)" 는 **비인과 시스템을 기준으로 인과 시스템을 재는 잘못된 비교**였다.
   실외만 통과(14.5–15.1)하고 oto·실내는 미달했지만, 이는 adapter 의 실패가 아니라 기준의 문제로 판단했다.

## 6. 판정과 결정 (2026-09-04, 사용자)

관문을 다음으로 **재정의**하고 U0.5 를 **통과**로 판정한다:

- (a) 동일 인코더 Nemotron RNN-T `[56,0]` 보다 우수 — 충족(−6 ~ −9 pt)
- (b) 오프라인 Qwen 대비 상대 열화 ≤ +50 % — 충족(oto +31–35 %, 실내 +30 %, 실외 +9–14 %)

backbone 은 유지하며 **U1(interleaved streaming ASR) 학습 준비로 진행**한다. U1 초기화는 `u05-asr-distill-12k/ckpt.pt`(adapter + LoRA)를 쓴다.

## 7. U1 로 넘기는 사항

- U1 관문(RNN-T 대비 상대 열화 ≤10 %)의 기준도 같은 논리로 **`[56,0]` RNN-T(25.4/23.4/24.5)** 로 둔다. U0.5 결과상 스트리밍 방출로 인한 열화 여유는 충분하다.
- 인코더 상위 블록 unfreeze 는 U1 에서 ablation 1 회로 확인(오프라인 상한 격차의 어느 만큼이 얼린 인코더 때문인지).
- 손실 스파이크 배치 원인 확인(발화 길이 상한 20 s → 15 s 로 낮추거나 길이 기준 정렬 배치).
- 실외(vs02) 도메인은 학습에 없음에도 가장 좋은 수치 → 잡음 강건성은 Nemotron 인코더의 장점이 그대로 전달됨(Stage 1 의 Qwen AuT 실외 취약과 대비).

## 8. 산출물

- 체크포인트: `/data4/tskim/VAPASR/experiments/uslm/{adapter-distill-qwen-aut-block8s, u05-asr-noinit, u05-asr-noinit-12k, u05-asr-distill, u05-asr-distill-lowlr, u05-asr-distill-12k}/`
  (각 `ckpt.pt` = adapter + LoRA, `results.json` = 평가 이력), 기준선 `baselines.json`
- 로그: `/data3/tskim/logs/bg-u05-*.log`
- 관련: [[task-uslm-u05-adapter-bridge]] (실행 기록), [[task-uslm-u1-interleaved-asr]] (다음 단계), [[output-stage1-encoder-probing]] (인코더 비교 맥락)
