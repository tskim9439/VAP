## [2026-10-12] query | 스트리밍 번역 베이스라인 xCOMET(XCOMET-XL) 채점

- Changed:
  - 위키: [[output-simulst-baselines-20261012]] 에 xCOMET 절
  - 원자료: `raw/sources/experiments/2026-10-12-simulst-baselines/` 에 `scores-xcomet-xl.json`, `xcomet_smoke2.py`, README xCOMET 절
  - mxc: `results/xcomet-20261012/`(캐시·로그)
- Reason:
  - 사용자 요청: xCOMET 수치. 사용자가 HF 약관에 동의하고 파일을 올렸다.
- 결과:
  - XCOMET-XL 은 빈 번역에 1.0 을 준다. 그래서 빈 번역은 0 점으로 뒀다.
  - **독→영:** s3 δ10 은 FLEURS 84.5 다. Seamless(85.2~86.2)와 0.7~1.7 점 차이로, BLEU 차이(3.6~3.8)보다 작다.
  - **한→영:** s3 δ10 은 75.4 로, 스트리밍 시스템 중 가장 높다.
  - **영→독:** 가장 큰 격차다. s3 δ10 FLEURS 78.3, CoVoST 81.3 이고, Seamless 는 90.4~91.4 다. 오프라인 88.3 대비 스트리밍에서 7~10 점을 잃는다.
  - **영→한:** 오프라인 76.4 는 모든 스트리밍 시스템보다 높지만, 스트리밍 δ10 은 58.9 다.
- Next:
  - 영→독·영→한의 스트리밍 손실을 오류 구간으로 나눠 본다. s4(CV4 영어 추가) 평가에도 xCOMET 을 함께 낸다.
- By: tskim
