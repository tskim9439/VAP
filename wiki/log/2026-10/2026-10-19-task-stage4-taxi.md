## [2026-10-19] task | Stage 4 S4 TAXI 어댑터 — 실제 독↔영 전화 대화를 교차 언어 prefix 로 디코드, cstbench 채점(출발점: 영→독 BLEU 18·독→영 4)

- Changed:
  - `experiments/cst_stage4_taxi.py`(신규) — 세션 창 나누기, 슬롯 규칙, cstbench 가설 쓰기, 채점(번역·받아쓰기 WER·슬롯 순열 WER), `--asr-only`
  - raw [[2026-10-19-stage4-dialogue-translation]]: TAXI 절, `taxi_p11.sh`·`taxi_p11_asronly.sh`, 보고서 집계(TAXI 텍스트는 mxc 에만)
  - [[output-stage4-prep-plan-20261019]]: S4 TAXI, S7 결정 필요 절
- Reason: 사용자 "그리고 Stage 4 준비를 이어서 수행해." — 주 언어쌍(EN↔DE)의 실제 교차 언어 평가
- 결과(Stage 3 최종 p11 ck3000 그대로, δ10):
  - 영→독 BLEU 18.0·chrF 35.7, 독→영 BLEU 3.7·chrF 15.8, 빈 턴 독→영 155/349
  - 받아쓰기 WER 영 102·독 99 %. 받아쓰기 prefix 로도 슬롯 순열 허용 53 % — 전화 음성·교차 언어 대화가 처음이다
- Next: Stage 4 파일럿 평가에 TAXI 를 넣는다. 전화 대역 증강 검토. S7 결정(엔진·원천 대화·규모)은 사용자
- By: tskim
