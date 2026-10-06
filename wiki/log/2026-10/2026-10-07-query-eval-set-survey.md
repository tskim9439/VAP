## [2026-10-07] query | 스트리밍 ASR·동시 음성번역·2인 대화 표준 평가셋 조사

- Changed:
  - 원자료: `raw/sources/research/2026-10-07-eval-set-survey/`(조사 보고 4 건 + README)
  - 위키: `wiki/outputs/output-eval-set-plan-asr-simulst-20261007.md`
- Reason: 사용자가 최종 모델의 스트리밍 ASR·Simul-S2TT 평가를 필수로 두고, 관련 논문을 탐색해 필요한 평가셋을 조사하라고 요청했다.
- 결과: 영어 ASR 은 LibriSpeech, 독·한 스트리밍 ASR 은 FLEURS(+Kspon), 영→독 동시 번역은 IWSLT26 장문(MCIF, XCOMET-XL, LongYAAL, 저 ≤2 s·고 2–4 s) + MuST-C, 독→영은 FLEURS·CoVoST 2, 한↔영은 FLEURS(+kosp2e·EnKoST-C). 두 언어 2인 대화 번역 공개 세트는 없음. 가장 가까운 선행 모델 Factorized DSM(2610.04333) 발견.
- Next: 데이터 확보(사용자), 채점기에 XCOMET-XL·LongYAAL 추가, 논문 6 절 "Standard tasks" 표, mxc 리전과 AI Hub 국외 반출 조항 확인
- By: tskim
