---
type: source
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: Sesame TurnBench — 30h dual-channel 영어 dyadic turn-taking 벤치마크와 104h otoSpeech 학습셋
url: https://arxiv.org/abs/2608.25218
observed: 2026-09-03
---

# TurnBench

## 무엇인가

Sesame 이 공개한 turn-taking 벤치마크. 30시간 dual-channel 영어 dyadic 음성
(154 dialogue, 106 voice actor, 53 pair). 3인 독립 어노테이션 + 2/3 consensus,
Fleiss κ = 0.78. **영어 전용, studio 녹음.**

학습셋 **otoSpeech** 는 약 104시간 full-duplex dialogue 로 평가셋과 화자가 겹치지
않으며 Hugging Face 에 공개. 라이선스는 voice cloning 을 금지하는 custom
non-commercial.

## 평가 프로토콜 (그대로 채택할 것)

- 매칭 윈도우: gold 시점 `t` 기준 **[t−0.25s, t+3.0s]**
- 윈도우 내 **가장 이른 미청구 예측** 이 TP
- negative span 내에서는 몇 번 발화해도 **FP 최대 1회**
- Recall = TP/(TP+FN), FPR = FP/(FP+TN), latency = p10/p50/p90

이 프로토콜은 [[turn-taking-evaluation-protocol]] 에 정리했고,
한국어 benchmark 도 동일 규약을 따른다.

## Baseline 수치

| 모델 | EOT Recall | EOT FPR | EOT p50 latency |
|------|-----------:|--------:|----------------:|
| **VAP** | **0.845** | 0.055 | 368 ms |
| Mimi-EP | 0.782 | 0.078 | — |
| Kyutai SVAD | 0.773 | 0.059 | — |

Interruption: VAP 0.945 recall @ 0.107 FPR, 994 ms latency.

**dev split 재현 (2026-09-03)**: oto 체크포인트 EOT 0.841 / 0.045 / p50 463 ms, INT 0.957 / 0.100 / 896 ms —
[[output-vap-turnbench-baseline-reproduction]]. 위 표는 test 기준이라 수치가 다르다.

VAP 가 두 트랙 모두 최강이지만, **빠르면서 recall 높고 FP 낮은 시스템은 아직 없다.**

## 결정적 참조점

> 사람은 floor-taking interruption 을 제외해도 현재 turn 종료 **중앙값 −151 ms**
> 시점에 이미 turn transfer 를 시작한다.

VAP 의 368 ms 와 사람의 −151 ms 사이 **약 520 ms** 가 이 연구가 노리는 격차다.
[[streaming-causality-and-latency-budget]] 참조 — 80 ms frame backbone 으로
이 격차를 메울 수 있는지가 실현 가능성의 핵심이다.

## 코드 저장소 (2026-09-03 확인) — `SesameAILabs/turnbench`, MIT

`/data3/tskim/third_party/turnbench` 에 클론, `vapasr` env 에 설치.

- **scorer**: `python -m turnbench.score predictions.json` (기본 dev 셋). 임계값 sweep: `python -m turnbench.sweep probs.json`.
- **제출 형식** (`docs/SUBMISSION_FORMAT.md`): conversation 별·화자별 `eot`/`interruption` 시각(초) 리스트.
  **causality 규약**: 타임스탬프는 "그 판단에 쓰인 오디오를 모두 들은 시각" — lookahead 를 시각에 접어 넣는다.
  [[streaming-causality-and-latency-budget]] 의 회계 원칙과 동일하다.
- **baseline 20종 동봉**: `vap`, `dualturn`, `wavlm_base_causal`, `wavlm_large_causal`, `wavlm_large_anchor`,
  `espnet_turntaking`, `rms_vad`, `smart_turn_v3`, `kyutai_semantic_vad`, `mimi_endpointer`, `moshi(_vad)`,
  `openai_*`, `gemini_*`, `asr_floor.py`, `oracle_annotator`. → [[task-add-missing-baselines]] 의 상당수를 재사용 가능.
- **VAP baseline 상세** (`baselines/vap`): VapGPT + CPC 50 Hz, 25 s 윈도(20 s context + 5 s step),
  `p_now` 임계. **리더보드 수치(0.845 / 0.055 / 368 ms)는 otoSpeech fine-tune 체크포인트(`viks66/VAP_checkpoints` 의
  `oto.ckpt`)** 이며 θ_eot = 0.9161, θ_int = 0.8591 (fp ≤ 0.1 sweep). 사전학습 원본은 `--pretrained`.
  `predictions-dev.json` 이 동봉되어 있어 **데이터만 있으면 즉시 재채점 가능**.
- 데이터: dev 38 대화(라벨 포함, parquet 3 shard), test 116(오디오만), otoSpeech 104 h (폴더당 `speaker_{1,2}_audio.wav`
  + `*_annotation_a.srt` + `metadata.json`). **세 데이터셋 모두 gated(auto) — HF 토큰 필요.**

## 한계

- 영어 전용 → [[question-korean-corpus-licensing]] 과 한국어 benchmark 필요성의 근거.
- studio 녹음 dyadic → 실제 전화·회의 환경과 domain gap.
