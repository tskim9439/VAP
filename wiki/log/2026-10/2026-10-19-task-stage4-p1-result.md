## [2026-10-19] task | Stage 4 파일럿 st4-p1 결과 — TAXI δ10 독→영 BLEU 3.7 → 22.7·영→독 18.0 → 26.4·cpWER 93.5 → 43.2(ck2000), Stage 3 회귀 없음

- Changed:
  - raw [[2026-10-19-stage4-dialogue-translation]]: "S6 파일럿 st4-p1 결과" 절, 집계 `st4-p1/`(대화·합성 summary.json, TAXI 점수), `cmp_st4p1.py`·`st4p1-vs-p11.txt`, `mask_bins_st4p1.py`·`st4p1-mask-bins.txt`
  - [[output-stage4-prep-plan-20261019]]: S6 상태·결과 절, summary
  - [[status]]
- Reason: 평가 큐(ck2000·ck3000·final) 완료. 사용자 질문 "Wer 과 cer 이 왤케 박살이 났지?", "stage 2 에서는 한국어 cer 이 8%, 영어 wer 이 3% 정도 나왔는데, 지금은 30% 이상이 나온다는게 이상해."
- 결과:
  - 학습 4,000 step·65 분, 재로드 parity 이상 없음
  - TAXI(실제 독↔영 전화): ck2000 이 가장 좋고 step 이 늘수록 내려간다(독→영 22.7 · 20.1 · 18.1)
  - Stage 3 회귀(같은 창 200): δ10 cp 25.28 → 26.16 · 24.95 · 25.09. 짝 bootstrap 구간은 한국어 ck2000 cp(+1.70)와 agn(+0.2–0.9) 외에는 0 을 포함한다
  - 단일 화자: Kspon·VoxPopuli·LibriSpeech·YODAS 는 그대로, FLEURS 한국어 CER 9.1 → 11.0–11.8(ck3000·final)
  - 영↔독 합성의 영어 δ10 WER 36–52 가 δ4(27–32)보다 나쁘다(원인 미확인)
  - 대화 cpWER 25 % = agn 18 + 귀속 7. 마스크 많은 창에서 오류·hyp/ref 가 커진다(71631 ≥ 40 % 창 cp 64.9). 번역 평가 창 100 의 71631 43.7 은 표본 차이였다(같은 창에서 32.6)
- Next: 체크포인트 선택(사용자), 영어 δ10 이상·FLEURS 한국어 확인, 대화 발화 단독 디코드 대조(제안)
- By: tskim
