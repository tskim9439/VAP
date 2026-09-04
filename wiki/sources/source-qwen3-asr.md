---
type: source
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: Qwen3-ASR-0.6B — AuT encoder 180M, 8x downsample 12.5Hz/80ms, Apache 2.0, 한국어 지원, main backbone 후보
url: https://arxiv.org/html/2601.21337v2
observed: 2026-09-03
---

# Qwen3-ASR

## 스펙 (확인됨)

| 항목 | 값 |
|------|-----|
| 공개 모델 | Qwen3-ASR-**0.6B**, Qwen3-ASR-1.7B |
| AuT encoder | 0.6B 판 **180M**, hidden **896** (1.7B 판은 300M / 1024) |
| 입력 | 128-dim FBank |
| downsampling | **8×** → 보고서상 12.5 Hz. **실측: 1 s conv chunk 당 13 프레임 = 13 Hz** (ceil(100/8), 2026-09-03) |
| attention | **dynamic flash attention window 1s ~ 8s** (streaming·offline 겸용) |
| 언어 | 30개 언어 + 22개 중국어 방언 (한국어·영어 포함) |
| 라이선스 | **Apache 2.0** |

**Qwen3-ForcedAligner-0.6B**: 11개 언어 (**한국어 포함 확인됨**), **80 ms 해상도**
(index × 80 ms 로 타임스탬프 복원).

## 로딩 방법과 제약 (2026-09-03 모델 카드 확인)

- 별도 패키지 **`qwen-asr`** 로 로드한다: `Qwen3ASRModel.from_pretrained(...)`,
  `model.transcribe(audio=...)`. transformers 백엔드와 vLLM 백엔드가 있다.
- **streaming 추론은 vLLM 백엔드에서만 지원** 되며 batch·timestamp 를 지원하지 않는다.
  → 연구용으로 encoder hidden state 를 streaming 으로 뽑으려면 **AuT 를 직접 chunk 단위로
  호출하는 코드를 써야 한다.** transformers 백엔드는 offline 이 기본이므로 이대로 특징을
  뽑으면 미래가 샌다 ([[streaming-causality-and-latency-budget]] 문제 1).
- **Qwen3-ForcedAligner-0.6B 는 한국어를 지원한다** (중·영·광동·불·독·이·일·**한**·포·러·서 11개).
  `Qwen3ForcedAligner.align(audio, text, language="Korean")`.

## 감사 결과 (2026-09-03) — 현 상태로는 turn-taking encoder 부적합

[[output-encoder-causality-audit]]. `Qwen3ASRAudioEncoder.forward` 가 `_prepare_attention_mask` 를
호출하지 않아 sdpa/eager 경로는 **전체 발화 양방향** 이고 `n_window_infer` 가 무효다 (flash-attn 전용).
의도된 블록 모드(1 s 독립 블록)도 lookahead 0–800 ms(평균 420) + 블록 간 좌측 context 없음.
→ [[decision-asr-backbone]] 수정, [[task-qwen-aut-causal-adaptation]].

**마스크 복원은 가능하다** (`experiments/qwen_aut_mask.py`, 레이어 forward 래핑). chunked-causal 1 s 가
as-is 최선(lookahead 420 ms, WER +5.9 %), 프레임 causal 은 lookahead 80 ms 에 WER 23.5 % — fine-tune 후보.

## 왜 main backbone 후보인가 (감사 전 판단)

- **80 ms / 12.5 Hz 가 [[source-muse-voice-transcribe]] 와 정확히 동일**하다.
  Muse-style audio/text interleaving 시퀀스를 그대로 구성할 수 있다.
- Apache 2.0 + 공식 fine-tuning recipe → 연구 자유도가 가장 높다.
- LLM decoder 의 incremental hidden state 를 그대로
  [[acoustic-linguistic-fusion]] 입력으로 쓸 수 있다.

## 미해결 위험 (반드시 먼저 확인)

기술 보고서는 **encoder 가 strictly causal 인지, lookahead 가 몇 ms 인지 명시하지
않는다.** "dynamic attention window 1–8s" 는 window 크기이지 right-context 가
0 이라는 뜻이 아니다.

- offline 모드로 특징을 뽑으면 미래 audio 가 `h_t` 에 새어 들어가 VAP 결과가 무의미해진다.
- streaming 모드라도 right-context 가 있으면 그만큼을 latency 에 **더해서** 보고해야
  VAP 와의 비교가 성립한다.

→ [[streaming-causality-and-latency-budget]], [[task-audit-encoder-causality-lookahead]]
