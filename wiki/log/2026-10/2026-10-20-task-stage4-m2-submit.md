## [2026-10-20] task | st4-m2a(전화 대역)·st4-m2b(전화 대역 + CallHome 독일어) 제출(apex job 93404·93405), V2 M0 화자 분리도 측정 시작

- Changed: [[output-stage4-prep-plan-20261019]](다음 학습), `experiments/spk_probe.py`(새 — 인코더 층별 화자 분리도)
- Reason:
  - 사용자 제출(st4-m2a 93404, st4-m2b 93405). 평가 큐 6 개(두 run × ck1500·ck2000·final, `TRANS_LEAD_SPACE=1`, 6 단계 CallHome 포함)를 걸었다.
  - 사용자 "여태까지의 프레임워크를 V1 로 두고 V2 을 실험해볼까" → 외부 모듈 없이 하나의 end-to-end 모델(학습 때만 교사) → "그래 M0 측정부터 시작해."
- 결과: 진행 중
  - M0: Stage 3 회귀 창에서 턴(≥ 1 s, 겹침 ≤ 0.2)의 인코더 층별 평균 특징으로 같은 창 두 화자 분리(EER·자기 뺀 중심 최근접 정확도)를 Nemotron 원본 → Stage 1 → 2 → 3 → 4(V1) 로 잰다. ECAPA 기준선은 vapasr env 에 speechbrain 이 없어 뺐다.
- Next: m2a·m2b 평가, M0 결과로 V2 화자 가지의 층·방향 결정
- By: tskim
