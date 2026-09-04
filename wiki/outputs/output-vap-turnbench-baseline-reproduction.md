---
type: output
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: TurnBench dev에서 VAP baseline 재현 — oto fine-tune 체크포인트는 공식 수치와 정확히 일치, 사전학습 원본은 recall −0.05·latency +150ms
sources:
  - [[source-turnbench]]
  - [[voice-activity-projection]]
---

# VAP baseline 재현 (TurnBench dev)

원본: `raw/sources/experiments/2026-09-03-vap-turnbench-repro/` (점수 3종 + 예측 JSON 2종),
절차: `experiments/reproduce_vap_turnbench.sh`, 환경: rack4 `vapasr`, GPU 1.

## 결과 (dev 38 대화, [[turn-taking-evaluation-protocol]] 규약)

| 설정 | EOT recall | EOT FP | EOT lat p10/50/90 | INT recall | INT FP | INT lat p50 | θ_eot / θ_int |
|---|---:|---:|---:|---:|---:|---:|---|
| 동봉 `predictions-dev.json` (oto) | 0.841 | 0.045 | −34 / **463** / 1657 | 0.957 | 0.100 | 896 | 0.9161 / 0.8591 |
| **oto 체크포인트 직접 예측** | **0.841** | **0.045** | −34 / **463** / 1657 | **0.957** | **0.100** | 896 | **0.91615 / 0.85913** (sweep) |
| 사전학습 원본 (`VAP_3mmz3t0u`, fine-tune 없음) | 0.793 | 0.094 | −89 / 613 / 2000 | 0.865 | 0.099 | 928 | 0.905 / 0.91 (sweep) |

혼동행렬(oto, EOT): tp 1601 / fn 303 / fp 48 / tn 1015. INT: tp 332 / fn 15 / fp 373 / tn 3360.

## 판정

1. **재현 성공.** 직접 예측이 동봉 예측과 임계값·점수 모두 동일하다. 스코어러·데이터·모델 파이프라인이
   공식과 같은 상태이며, 이후 모든 비교의 **기준선이 확보되었다**.
2. **리더보드 수치는 dev 가 아니라 test 다.** 리더보드(test) 0.845 / 0.055 / 368 ms 와 dev 0.841 / 0.045 / 463 ms 는
   같은 모델의 다른 split 결과. 논문에서 어느 split 인지 명시해야 한다. test 예측은 라벨이 없어 제출로만 채점된다.
3. **otoSpeech fine-tune 의 효과가 크다.** 사전학습 원본은 같은 FP 예산에서 recall −0.05, p50 latency +150 ms.
   → Stage 1 encoder probing 의 CPC 조건은 **사전학습 원본 CPC 를 freeze** 한 것이므로, 공정한 비교를 위해
   probe head 를 otoSpeech 에서 학습해야 한다 (fine-tune 된 VAP 와 비교하는 게 아니라 동일 데이터로 head 를 학습).
4. **INT 는 FP 가 문제.** FP 373 vs TP 332 — interruption 은 recall 은 높지만 오경보가 같은 수준.
   [[turn-taking-objectives]] 의 event head 가 개선할 여지가 가장 큰 지점.
5. EOT latency p10 이 **음수(−34 ms)** — VAP 가 실제 종료 전에 예측하는 사례가 10 % 이상 존재한다.
   projection 의 본질이며, 사람의 −151 ms 에 다가가는 방향이다.

## 재사용

- `bash experiments/reproduce_vap_turnbench.sh` 한 번으로 전 과정 재실행 (약 25 분, dev).
- 다른 모델은 `baselines/<name>/predict.py` 형식으로 `predictions.json` 을 내면 같은 스코어러로 비교된다.
  → [[task-add-missing-baselines]] (rms_vad, dualturn, wavlm_*_causal 은 저장소 동봉).
