## [2026-10-27] query | E5 한 스트림 + 정답 슬롯 활동 입력 결과

- Changed:
  - [[output-diarization-integration-20261025]] 6 절, [[task-data-expansion-20261026]]
- Reason:
  - E5(q17-e5, F1 레시피 + `--slot-input`, 95933 PARITY OK)의 AMI 정답 활동 입력 평가(amiact)가 나왔다.
  - F1 final 대비 회의 단위 짝 bootstrap 결과다.
    - δ10 SDM eval: 53.94 → 29.24(−24.7*), IHM-mix eval: 49.43 → 23.39(−26.0*). dev 도 −29.7 ~ −30.1 이다.
    - 귀속 몫은 1.0–2.1 이다.
  - E2 화자별 디코드 상한(SDM eval 28.6) 수준이라, 한 스트림 형식이 통한다(계획 6 절 판단 기준).
- Next:
  - E6(인과 헤드 예측 활동, 95935 학습 중)이 상한을 얼마나 지키는지 본다.
  - E1c DER 도 잰다(`e1c_eval.sh` 진행 중).
  - E5 의 활동 없는 평가(ami 4 몫)로, 입력이 없을 때 얼마나 무너지는지 본다.
- By: tskim
