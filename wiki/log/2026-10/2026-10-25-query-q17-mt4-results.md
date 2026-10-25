## [2026-10-25] query | 1.7B R1x 대 회의 + 4 슬롯 결과

- Changed: `raw/sources/experiments/2026-10-22-q17-chain/README.md`(결과 절), `raw/sources/experiments/2026-10-21-v3-attribution/v3_runs_summary.py`(noeot_summary.py 를 run 인자로 — 계산 같음, 기준 대비 단일 화자 가드), [[status]]
- Reason:
  - 사용자 "진행 상황 알려줘" — 1.7B 학습 둘과 평가 둘(95328·95329)이 끝났다.
  - 회의 + 4 슬롯은 AMI 창 cp 를 모든 분할에서 5–8 pt 낮췄다(받아쓰기 덕, 귀속 몫은 +1–3).
  - TAXI 는 δ10 −2.0(final, 경계선)·−2.7(avg3, 유의, 귀속 −1.1 유의)이다.
  - 대가로 단일 화자 영어 LibriSpeech other 가 +0.55–0.88 이다.
  - dev 귀속은 모든 판이 0 근처라 선택 기준으로 쓸 수 없다.
- Next: 1.7B 대화 모델로 회의 + 4 슬롯(avg3)을 채택할지 사용자가 정한다. 0.6B 회의 + 4 슬롯 시험(`submit-q06-mt4.sh`)을 낼지도 정한다.
- By: tskim
