---
type: output
status: active
created: 2026-09-04
updated: 2026-09-04
summary: Stage 1 frozen encoder probing 결과 — 프레임율 효과 > 인코더 효과, H1 미지지, Qwen AuT 실외 취약, INT는 사전학습 의존
sources:
  - [[task-stage1-encoder-probing]]
  - [[output-feature-cache-and-compute-budget]]
  - [[output-vap-turnbench-baseline-reproduction]]
---

# Stage 1 — frozen encoder probing (잠정 판정, 2026-09-04)

설정: 인코더 7종 freeze(캐시), 동일 causal probe head(d256 · 2층 · 2.83 M, 입력 projection 만 인코더별), 학습 otoSpeech 105 h + AI Hub 실내 50 h,
4 epoch, TurnBench dev 38 대화(공식 sweep/scorer, lookahead 접기), 고정 FP 예산에서 최대 recall. 코드 `experiments/train_probe.py`, 결과 `show_probe_results.py`.

## 결과

| encoder | Hz | causal | EOT R@fp≤0.045 / p50 | EOT R@fp≤0.10 / p50 | INT R@fp≤0.10 / p50 | KO val CE 실내 / 실외 |
|---|---:|---|---|---|---|---|
| fbank (사전학습 없음) | 50 | ✓ | 0.799 / 988 ms | 0.838 / 761 | 0.533 / 1462 | — |
| **cpc** | 50 | ✓ | **0.880 / 499** | **0.893 / 448** | 0.922 / 987 | 2.90 / 3.18 |
| cpc → 12.5 Hz | 12.5 | ✓ | 0.867 / 527 | 0.873 / 486 | 0.914 / 1037 | — |
| nemotron-c0 | 12.5 | ✓ | 0.868 / 553 | 0.868 / 528 | 0.916 / 1338 | 3.11 / 3.30 |
| qwen-aut-causal | 13 | ✓ (OOD 마스크) | 0.862 / 444 | 0.865 / 423 | **0.945 / 909** | 2.88 / **5.81** |
| qwen-aut-cc1s (블록 끝 접음) | 13 | 블록 | 0.834 / 456 | 0.841 / 1517 | 0.841 / 1517 | 2.54 / **6.52** |
| wavlm-base / large | 50 | ✗ | 0.80 / — | 0.81 / — | 0.90 / — | — |
| VAP oto fine-tune (기준) | 50 | ✓ | 0.841 / 463 | — | 0.957 / 896 | — |

## 판정

1. **H1 미지지.** ASR 사전학습 표현(Nemotron 12.5 Hz)은 같은 12.5 Hz 의 CPC 와 동급(0.868 vs 0.867)이고 CPC 50 Hz(0.880)보다 낮다.
   **프레임율 효과가 인코더 효과보다 크다.** → IS-SLM 의 12.5 Hz audio clock 만으로는 부족 — U3 의 50 Hz 사이드 브랜치 하이브리드 근거.
2. **frozen probe 가 fine-tune VAP 보다 EOT recall 이 높다** (CPC 0.880 vs 0.841 @ FP 0.045), latency 는 40–90 ms 느리다. 학습 데이터가 더 많다는 점(155 h vs ~104 h) 명시.
3. **INT 는 사전학습 유무에 크게 의존** (fbank 0.53 vs 0.92–0.95). qwen-aut-causal 이 INT 최강(0.945) 이나 run 간 분산이 크다(CPC 1-epoch 0.902 vs 4-epoch 0.816 @0.045) → seed 반복 필요.
4. **Qwen AuT 는 잡음에 취약**: 실내(학습 도메인) CE 는 CPC 와 같거나 낫지만 실외(SNR +5/+10 dB)에서 붕괴(5.8–6.5 vs 3.2–3.3). 스튜디오 벤치마크만으로는 안 보인다.
5. INT latency 1.0–1.6 s 로 VAP(896) 보다 느림 — probe 구조(20 s 창·2층)/손실에서 볼 지점.
6. lookahead 를 정직하게 접으면 cc1s 의 이점이 사라진다(371 → 456 ms, INT 1.5 s). WavLM 은 비인과 참조(20 s 창) — latency 무의미.

## 남은 것

- seed 3회 반복(특히 INT), EN-only 학습 조건, DualTurn encoder 확보, probe 창/층 ablation. → [[task-stage1-encoder-probing]]
- 원본 수치: `/data4/tskim/VAPASR/experiments/probe/*/results.json`.
