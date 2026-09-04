---
type: task
status: doing
owner: tskim
due: 2026-10-22
priority: p0
created: 2026-09-03
updated: 2026-09-03
summary: frozen encoder 비교 실험 — CPC/WavLM/FastConformer/AuT/DualTurn + random floor, 교란 통제
sources:
  - [[output-streaming-vap-research-plan]]
---

# Stage 1 — Encoder probing

## 배경

→ [[question-asr-representation-vs-ssl-for-vap]]

**이 연구의 첫 논문 가치가 여기 있다.** 동시에 가장 망치기 쉬운 실험이다 —
encoder 들이 frame rate·차원·causality·pretraining 규모에서 모두 다르기 때문에
통제 없이는 "ASR 이 좋다" 와 "크고 데이터 많다" 를 구분할 수 없다.

## 완료 조건

- [x] 후보 확보(캐시 중): CPC, WavLM Base/Large, Nemotron-c0/c1, Qwen3 AuT(cc1s/causal), **fbank(사전학습 없음 floor)**. DualTurn encoder 는 미확보 (turnbench baselines/dualturn 확인 필요)
- [x] **probe 학습기 완성** — `vapasr/probe/` + `experiments/train_probe.py`: 캐시 특징 → 고정 용량 **causal** Transformer head(d256, 2층, 2.83M; 입력 projection 만 인코더별) → VAP 256 + VAD 손실. 학습 otoSpeech + AI Hub TS_01_5 50 h, val AI Hub VS_02, **TurnBench dev 를 공식 sweep/scorer 로 자동 채점** (20 s context + 5 s step, baseline 과 동일 규약). 1 epoch ≈ 3 분(GPU 단독).
- [ ] 모든 encoder freeze, 동일 용량 head — **head 학습 데이터는 otoSpeech(+AI Hub)** 로 통일. 사전학습 원본 VAP 는 oto fine-tune 대비 recall −0.05 이므로 ([[output-vap-turnbench-baseline-reproduction]]) 데이터 차이가 결과를 지배한다
- [ ] 공통 frame rate 로 리샘플한 조건 + 원 해상도 조건 **둘 다** 보고
- [ ] encoder 별 pretraining 시간·파라미터·**실효 lookahead** 를 결과표에 명시
- [ ] random-init floor 대비 상대 이득으로 보고
- [ ] 데이터는 **50–100시간이면 충분** — 순위가 안정되는지 학습곡선으로 확인
- [ ] **판정**: H1 지지 / 기각 / 불확실 중 하나를 명시적으로 결론

## 진행 기록

- 2026-09-04 13:15 **매트릭스 중간 (6/14 run, 4 epoch, dev, 고정 FP 예산 기준 최대 recall)**:
  | encoder | Hz | EOT R@fp≤0.045 / p50 | EOT R@fp≤0.10 / p50 | INT R@fp≤0.10 / p50 |
  |---|---:|---|---|---|
  | fbank (floor) | 50 | 0.799 / 988 ms | 0.838 / 761 | 0.533 / 1462 |
  | cpc | 50 | **0.880 / 499** | 0.893 / 448 | 0.922 / 987 |
  | cpc (12.5 pool) | 12.5 | 0.867 / 527 | 0.873 / 486 | 0.914 / 1037 |
  | nemotron-c0 | 12.5 | 0.868 / 553 | 0.868 / 528 | 0.916 / 1338 |
  | VAP oto ft (기준) | 50 | 0.841 / 463 | — | 0.957 / 896 |
  읽기: (1) 어제의 "FP 0.045 에서 recall 급락" 은 nearest-θ 뷰의 착시 — 예산 기준으로는 frozen probe 가 fine-tune VAP 보다 recall 이 높다(latency 는 40–90 ms 느림).
  (2) **Nemotron ≈ CPC@12.5 Hz, CPC@50 Hz 보다 낮다** → 현재까지 H1 지지 없음; frame rate 효과가 인코더 효과보다 크다.
  (3) INT 는 사전학습 유무에 크게 의존(fbank 0.53 vs 0.92) 하지만 run 간 분산이 크다(CPC 1-epoch run 0.902 vs 4-epoch 0.816 @0.045) — **seed 반복 필요**.
  (4) INT latency 가 1.0–1.6 s 로 VAP 896 ms 보다 훨씬 느림 — probe 의 p_now 가 interruption 초기에 느리게 반응.
- 2026-09-04: **첫 결과 (CPC frozen, 1 epoch, causal head 2.83M, TurnBench dev 38 대화)**: EOT recall **0.899 @ FP 0.058**, lat p10/50/90 −57/**456**/1715 ms
  (tp 1711 fn 193 fp 62 tn 1001); INT 0.948 @ 0.098, 867 ms. 기준 VAP(oto fine-tune) 0.841 @ 0.045 / 463 ms — fp≤0.1 operating point 에서
  frozen CPC + 작은 head 가 이미 recall 을 넘고 latency 동급. **주의**: sweep 이웃 θ 에서 FP 0.045 로 내려가면 recall 0.119 로 급락 —
  확률이 0.9–0.95 에 몰려 sweep 격자가 거칠다. 같은 FP 비교를 위해 세밀한 θ 격자 필요. 학습 데이터(otoSpeech 104 h + AI Hub 50 h) 가
  VAP(oto ~100 h) 보다 많다는 점도 명시할 것.
- 2026-09-04: 학습기 작성·스모크 통과. `experiments/run_stage1.sh <encoders>` 로 (고유율, 공통 12.5 Hz) 매트릭스 실행. 첫 CPC 1-epoch 결과 채점 진행 중.
- 2026-09-03: 생성.
