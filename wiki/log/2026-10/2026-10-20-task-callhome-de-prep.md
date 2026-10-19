## [2026-10-20] task | CallHome 독일어 두 교사 의사 전사 — WER < 3 % 발화 25.6 %(3.06 h), 대화 창 418 개는 마스크 79 %

- Changed: raw [[2026-10-19-stage4-dialogue-translation]] "CallHome 독일어 준비" 절·`callhome-de/`·`callhome_de_post.sh`, `experiments/p2_make_split.py`(callhome-de 통화 단위), `experiments/stage3_train.py`(DLG_LANG), mxc `phase2/{callhome-de.*, windows-chde/, splits/v3.json}`
- Reason: 사용자 "독일어 CallHome 만 준비해줘. 그리고 Whisper 와 Qwen3-ASR 두개 모델로 전사하고 서로 WER 또는 CER 이 3% 미만인 샘플만 학습에 활용하도록 하자."
- 결과:
  - 120 통화 18.4 h, 발화 27,967 중 교사 WER < 3 % 7,149(25.6 %, 3.06 h). 교사 WER ≥ 50 % 가 40 % 다(전화 음질·상대 말 섞임).
  - 기준 완화 시 < 10 % 4.96 h, < 20 % 8.33 h, < 30 % 11.25 h
  - 3 % 기준 대화 창 418 개(3.6 h), 창 마스크 비율 중앙 0.83, 마스크 절반 넘는 창 94 %
  - split v3 = v2 + callhome-de 통화 12 개(1.9 h) held-out
- Next: 쓰는 방식 결정(사용자) — 대화 창 그대로 / 감독 발화를 단일 화자 항목으로 / 기준 완화 / TalkBank 사람 전사
- By: tskim
