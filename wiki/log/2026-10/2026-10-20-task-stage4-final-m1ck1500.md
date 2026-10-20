## [2026-10-20] task | Stage 4 최종을 st4-m1 checkpoint-1500 으로 확정, CallHome 독일어 단일 화자 셋 dry-run 통과

- Changed: [[output-stage4-prep-plan-20261019]] S6·summary, [[status]], raw [[2026-10-19-stage4-dialogue-translation]](st4-m1 결과 절, VibeVoice 절에 ck1500 기준 비교·TAXI 언어 확인)
- Reason: 사용자 "체크포인트는 st4-m1 ck1500 으로 정하자.", "VibeVoice 는 교차 언어를 정상적으로 인지해서 두 개 언어를 전사했어?"
- 결과:
  - **Stage 4 최종 = `/soundai/Model/VAPASR/stage3-st4-m1/checkpoint-1500`**
  - ck1500 대 VibeVoice 스트리밍(δ10 cpWER): 한국어 19.0–35.2 대 52.0–105.1, 독일어 20.7–29.6 대 28.2–34.3, 영어 27.0 대 26.3, TAXI 40.9 대 19.2
  - VibeVoice 는 TAXI 에서 두 언어를 각자 그 언어로 받아썼다(언어별 WER 영 21.3·독 16.7, 두 언어가 다 나온 세션 76/86)
  - `--replay-set rep_chde`(CallHome 독일어 감독 발화 6,174) 학습 dry-run: 32 항목 × δ 7 개 불일치 0
- Next: 다음 학습에 rep_chde·전화 대역 증강, 영어 반복 생성 원인
- By: tskim
