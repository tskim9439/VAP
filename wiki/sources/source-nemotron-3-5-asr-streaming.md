---
type: source
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: NVIDIA Nemotron 3.5 ASR Streaming 0.6B — cache-aware FastConformer 24층 + RNNT, ko-KR 7.12% WER, 첫 baseline backbone
url: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
observed: 2026-09-03
---

# Nemotron 3.5 ASR Streaming 0.6B

## 스펙 (확인됨)

| 항목 | 값 |
|------|-----|
| encoder | **Cache-Aware FastConformer, 24 layers** |
| decoder | **RNNT** |
| 파라미터 | 600M |
| chunk 옵션 | **80 / 160 / 320 / 560 / 1120 ms** (각각 1/2/4/7/14 frame) |
| frame rate | **80 ms (12.5 Hz)** |
| cache-aware | ✅ 겹치는 재계산 없음 |
| 한국어 | ko-KR **transcription-ready**, 1.12 s chunk + 언어 지정 시 **WER 7.12%** |
| 라이선스 | **OpenMDW-1.1** (Apache 아님, 상업적 사용 가능) |

## 왜 첫 baseline 인가

streaming engineering 이 이미 해결되어 있다. chunk 크기를 5단계로 바꿔가며
**latency-quality curve** 를 바로 뽑을 수 있는 것이 이 모델의 가장 큰 가치다.
RNNT predictor state 가 명시적으로 존재해 [[acoustic-linguistic-fusion]] 실험이
Qwen 보다 단순하다.

## chunk 는 곧 우측 attention context 다 (2026-09-03 모델 카드 확인)

chunk 설정은 encoder 의 `att_context_size = [좌측, 우측]` (단위: 80 ms frame) 이다.

| chunk | att_context_size | 우측 context = **명시적 lookahead** |
|------:|------------------|------------------------------------:|
| 80 ms | `[56, 0]` | **0 ms** |
| 160 ms | `[56, 1]` | 80 ms |
| 320 ms | `[56, 3]` | 240 ms |
| 560 ms | `[56, 6]` | 480 ms |
| 1120 ms | `[56, 13]` | 1040 ms |

즉 **80 ms chunk 는 attention 수준에서 strictly causal** 이고, 좌측 context 는
56 frame = 4.48 s 다. 이것으로 [[question-encoder-lookahead-and-causality]] 의
Nemotron 쪽은 문서상 답이 나왔다. 남은 것은 convolution subsampling 의 암묵적
lookahead 가 0 인지 절단 실험으로 확인하는 것뿐이다.

**함의**: 320 ms chunk 이상은 lookahead 만으로 VAP 의 368 ms 에 근접한다.
turn-taking 용도라면 **80 또는 160 ms chunk 만 후보** 이며, 그 chunk 에서의
ko-KR CER 이 실제 관심 수치다 (7.12% 는 1120 ms 기준).

## 실측 (2026-09-03, rack4 / vapasr env)

| 항목 | 값 |
|------|-----|
| 파라미터 | 638M |
| encoder | `ConformerEncoder`, d_model **1024**, subsampling 8, 기본 `att_context_size=[56,3]` |
| frame rate | 14.84 s 오디오 → 187 frame = **79.4 ms/frame** (12.5 Hz 확인) |
| 로드 | ~60 s (캐시 후) |
| streaming API | `encoder.cache_aware_stream_step`, `get_initial_cache_state`, `set_default_att_context_size` |
| NeMo | PyPI 3.0.0 과 git main 3.1.0 모두 로드·추론 OK |

### 언어 전달 — 함정

모델 카드의 `target_lang=` 은 **`transcribe()` 키워드로 넘겨도 dataset 까지 도달하지 않는다**
(3.0.0, 3.1.0 공통: `_setup_transcribe_dataloader` 가 `default_lang` 을 넘기지만
`audio_to_text_lhotse_prompt_index` 는 `cut.supervisions[0].language` 만 읽는다 → `None` 오류).
lhotse NeMo 어댑터가 manifest 의 **`"lang"`** 필드를 supervision.language 로 매핑하므로,
**manifest jsonl 에 `"lang": "ko-KR"` 을 넣어 `transcribe(manifest_path)` 로 호출**해야 한다.
`scripts/smoke-test-models.py` 가 이 방식이다.

## 실측 lookahead (2026-09-03)

[[output-encoder-causality-audit]]: `[56,0]` ≤80 ms, `[56,1]` ≤160 ms, `[56,3]` ≤320, `[56,6]` ≤480,
`[56,13]` ≤880. 문서의 chunk 크기 = 최대 lookahead 로 확인. **turn-taking 에는 80/160 ms chunk 만 사용.**

## 주의

- **frame rate 가 80 ms 로 원 VAP(20 ms / 50 Hz)의 4배 거칠다.**
  → [[streaming-causality-and-latency-budget]]
- ko-KR WER 7.12% 는 **1.12 s chunk** 기준이다. 80–320 ms chunk 에서는 더 나쁠 것이며,
  turn-taking 에 쓰는 chunk 에서의 WER 을 따로 측정해야 한다.
- 라이선스가 OpenMDW-1.1 이므로 파생 모델 배포 조건을 논문 제출 전에 확인해야 한다.
