---
type: source
status: active
created: 2026-09-09
updated: 2026-09-09
summary: SoulX-Duplug(arXiv 2603.14877) — Qwen3-0.6B 위 interleaved 청크 streaming ASR + 대화 상태 토큰. 우리 설계와의 공통점·차이·시사점
observed: 2026-09-09
---

# SoulX-Duplug (arXiv 2603.14877, 2026-03-16)

Soul-AILab 외(Xie Chen 그룹). "Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation".
코드: https://github.com/Soul-AILab/SoulX-Duplug (공개 약속).

## 설계

| 항목 | SoulX-Duplug | VAPKT(우리) |
|---|---|---|
| 백본 | Qwen3-0.6B(텍스트 LLM) | Qwen3-ASR-0.6B thinker |
| 음성 입력 | GLM-4-Voice 이산 토큰 12.5 Hz(동결), 블록 12 · look-back 960 ms · look-ahead 40 ms | Nemotron 3.5 streaming 0.6B 연속 특징([56,0], 동결) + adapter |
| 청크 | 160 ms(오디오 토큰 2 개) | 80 ms |
| 시퀀스 | `{A_t, T_t, S_t}` 반복: 청크 오디오 → 그 청크 텍스트 → 상태 토큰 | `[AUDIO_k] 텍스트 <NEXT_AUDIO>` 반복 + `<DELAY_δ>` 조건 |
| 상태 토큰 | idle / nonidle / backchannel / complete / incomplete (5 종, 청크마다 1 개) | 없음(VAP 층은 후속) |
| 손실 | 토큰 종류별 가중 CE | 텍스트 + `<NEXT_AUDIO>` 가중(EN 0.3 / KO 0.15) |
| 타임스탬프 | Paraformer(중) · WhisperX(영) 정렬 → 청크 배치 | 자체 강제 정렬(align-asr-tn-v1) → `k=⌊t_end/80ms⌋+δ` |
| 학습 | 3 단계: 비스트리밍 ASR 사전학습(47k h 중 + 31k h 영, LLM·adapter full FT) → 스트리밍 ASR 적응 → 상태 예측 SFT(LoRA r=32, Fisher 천 h + 자체 중국어 만 h, 상태 라벨은 Qwen2.5-72B 로 생성) | C2: full FT 1,930 h → D2: 5.6k h |
| 추론 | **자체 스트리밍 ASR 을 쓰지 않고** SenseVoice Small/Paraformer 가 청크마다 텍스트를 teacher-forcing | 자체 디코드 |
| 지연 | 이론 240 ms(80 + 160), 실측 205–250 ms(L20) | δ=2 p50 209 ms(dev-clean), δ=4 357 ms |

## 결과(요지)

- Easy Turn(Zh) ACC 84.3 % (complete 89.3 / incomplete 79.3), EN 83.3 %. Full-Duplex-Bench turn-taking TOR 0.93(EN)·0.99(ZH), 응답 지연 0.69 s / 0.85 s.
- 절제: ASR 사전학습 제거 −3.8 pt, teacher-forcing 제거(자체 스트리밍 ASR 사용) **−10.8 pt** (Table 5).
- **스트리밍 ASR 의 WER/CER 은 보고하지 않는다.** 6.3 절: "매우 작은 청크의 스트리밍 ASR 은 본질적으로 어렵다 … 영어에서 단어가 청크 경계에서 조각나 인식이 불안정 … LLM 기반 ASR 이 속도–정확도 균형에서 RNN-T 같은 전통 구조를 반드시 능가하지는 않는다."

## 시사점

1. 우리 파일럿 결론(δ=2 에서 RNN-T 대비 열세, δ=4 에서 근접)과 같은 관찰을 독립적으로 보고했다. 그들은 이 문제를 **외부 ASR teacher-forcing** 으로 우회했고, 상태 예측만 LLM 이 담당한다.
2. 청크 160 ms(우리 80 ms 의 2 배) + look-ahead 40 ms 로도 어렵다고 했으므로, 우리의 `<DELAY_δ>` 조건(δ 로 지연–정확도 교환)은 그들에게 없는 손잡이다.
3. 상태 토큰 5 종·청크마다 1 개 예측·토큰 종류별 손실 가중은 VAP 층 설계에 그대로 참고할 만하다. 상태 라벨을 LLM(Qwen2.5-72B)으로 만든 점은 우리 대화 코퍼스(NIKL·71631)에 적용 가능.
4. 이산 토큰(12.5 Hz) 대신 연속 특징을 쓰는 우리 쪽이 음향 정보 손실은 적다. 대신 그들은 텍스트 LLM 을 그대로 써 ASR 사전학습에 78k h 를 썼다.

관련: [[output-vapasr-model-and-sequence]], [[output-stage2-c2-final-eval]], [[source-conversation-corpora]]
