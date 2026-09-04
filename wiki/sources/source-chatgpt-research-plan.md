---
type: source
status: active
created: 2026-09-03
updated: 2026-09-03
summary: ChatGPT 로 작성한 streaming ASR + VAP 통합 연구 초안 — 이 볼트 연구 방향의 출발점
raw_path: raw/sources/ChatGPT_Research_Plan.md
raw_authors:
  - tskim
observed: 2026-09-03
---

# ChatGPT 연구 계획 초안 (Streaming Conversational Projection ASR)

## 무엇인가

사용자(tskim)가 ChatGPT 와 대화하며 정리한 연구 방향 초안. 15개 절로 구성되며
주제 제안, 관련 연구 비교, backbone 선정, architecture, objective, dataset,
6단계 로드맵, 평가 지표, 한국어 benchmark 신설, ablation 축을 다룬다.

## 핵심 주장

- **주제**: 하나의 streaming speech representation 에서 transcription 과 future
  conversational dynamics 를 **동시에** 예측한다. "VAP + ASR 병렬 결합" 이 아니다.
- **novelty 축**: [[source-muse-voice-transcribe]] 의 endpointing 은 "지금 끝났는가"
  이고 [[voice-activity-projection]] 은 "앞으로 2초간 누가 말할 것인가" 다. 이 둘은
  다른 문제이며, 그 차이가 novelty 라는 것이 초안의 중심 논지다. **이 판단은 타당하다.**
- **backbone**: Nemotron 3.5 로 feasibility 확인 → Qwen3-ASR-0.6B 로 main model.
- **objective**: VAP 256-class + time-to-next-turn + semantic event head.
- **contribution 후보**: 한국어 turn-taking benchmark 신설.

## 검증 결과 (2026-09-03, 웹 확인)

초안이 인용한 자료는 **전부 실재**했다. 다만 계획을 바꿔야 하는 사실이 나왔다:

| 인용 | 실재 | 계획에 영향을 주는 확인 사항 |
|------|------|------------------------------|
| Muse Voice Transcribe | ✅ | **closed-weights, API 전용** ($0.18/h). fine-tune·ablation 불가 |
| Nemotron 3.5 ASR streaming 0.6B | ✅ | 스펙 일치. 라이선스는 Apache 아닌 **OpenMDW-1.1** |
| Qwen3-ASR (arXiv 2601.21337) | ✅ | 스펙 일치. **causality/lookahead 미문서화** |
| JAL-Turn (arXiv 2603.26515) | ✅ | 2026-03-27. speech-only, cross-attention, hold/shift |
| Next-Turn (arXiv 2606.18094) | ✅ | 2026-06-16. 320ms 내 endpoint accuracy +25.9%p |
| TurnBench | ✅ | 인용 수치 정확. **영어 전용**, otoSpeech 104h 공개 |
| AI Hub 감정 태깅 자유대화 | ✅ | **stereo 확인**. 단 내국인 한정 + 재배포 제약 |
| CANDOR | ✅ | 화자별 분리 채널 ✅, **CC BY-NC** |

## 초안에서 누락된 것

- **[[source-turn-taking-related-work-2026]] 의 DualTurn** (arXiv 2603.08216).
  dual-channel generative pretraining 으로 VAP 를 크게 앞선다 (weighted F1
  0.633 vs 0.389, 0.5B). 제안 방향과 직접 경쟁하는 가장 중요한 baseline인데
  초안에 없다.
- **[[streaming-causality-and-latency-budget]]**: encoder lookahead 를 latency 에
  합산하지 않으면 VAP 와의 비교가 성립하지 않는다.
- **baseline 부족**: VAD+threshold, cascade(ASR→text turn model) 실측치가 없다.
- **라이선스로 인한 benchmark 배포 제약** 미고려.

## 개선된 계획

→ [[output-streaming-vap-research-plan]] (v2)

## 원본

`raw/sources/ChatGPT_Research_Plan.md` — 원문 그대로 보존. 수정하지 않는다.
