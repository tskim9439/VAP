---
type: task
status: done
owner: tskim
due: 2026-10-08
priority: p1
created: 2026-09-03
updated: 2026-09-03
summary: GPU 예산 산정과 frozen encoder 출력 디스크 캐싱으로 probing 실험 가속
sources:
  - [[output-streaming-vap-research-plan]]
---

# 컴퓨트 예산과 특징 캐싱

## 배경

초안은 컴퓨트를 다루지 않았다. frozen encoder probing 은 **인코더 출력을 미리
디스크에 캐시하면 며칠 → 분 단위**로 줄어든다. Stage 1 에서 여러 encoder ×
여러 하이퍼파라미터를 돌려야 하므로 효과가 크다.

## 완료 조건

- [x] 가용 GPU 및 스토리지 확인 — GPU 1(배정), /data3 2.2 TB 여유, 64 CPU / 503 GB RAM
- [x] 특징 크기 산정 — 214 h 기준 총 ≈ 430 GB (WavLM 50 Hz 가 276 GB) → [[output-feature-cache-and-compute-budget]]
- [x] 추출·캐싱 파이프라인 — `vapasr/features/`, 대화별 npy + index.jsonl, 재개 가능. 세그먼트 이어붙이기 fp32 exact 검증(함정 4개 해결)
- [x] **RTF·GPU memory** (stereo 2채널 배치): cpc 0.0018 / nemotron 0.0052 / qwen 0.018 / wavlm 0.017–0.019, peak 1.4–2.4 GB — 채널 2회 비용은 RTF 에 포함되어도 실시간 대비 50×+ 여유 ([[streaming-causality-and-latency-budget]] 문제 4 해소)
- [x] 예산 초과 없음 — 대안 불필요

## 진행 기록

- 2026-09-04: **완료.** 7 인코더 × 816 대화(205 h, TurnBench dev 38 포함) = **447 GB**. 실측 RTF(공유 GPU): cpc 0.0025 · fbank 0.0002 · nemotron 0.010 · qwen-cc1s 0.005 · qwen-causal 0.014 · wavlm-base 0.013 · wavlm-large 0.018; peak GPU nemotron 13 GB(78 min 파일), wavlm-large 18 GB. 총 소요 ≈ 26 h(3 작업 병행). fbank floor 추가.
- 2026-09-03: 생성.
