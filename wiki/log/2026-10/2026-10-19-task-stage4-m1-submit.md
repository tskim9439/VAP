## [2026-10-19] task | Stage 4 본 학습 st4-m1 제출(apex job 92394), CallHome 독일어 의사 전사 시작, VibeVoice 서버 직접 다운로드

- Changed:
  - [[output-stage4-prep-plan-20261019]] S6(제출)
  - `experiments/callhome_de_pseudo_label.py`(새): HF talkbank/callhome deu → 두 교사 의사 전사 → `callhome-de.dialogues.jsonl`
  - raw [[2026-10-19-stage4-dialogue-translation]]: `hf_fetch.py`(고정 리비전·sha256 대조 다운로드, 토큰은 표준 입력), `vibevoice_fetch_server.sh`, `callhome_de_asr.sh`
- Reason:
  - 사용자 제출(st4-m1, 92394). 평가 큐 ck1500·ck2000·final 을 `TRANS_LEAD_SPACE=1` 로 걸었다.
  - 사용자 "독일어 CallHome 만 준비해줘. 그리고 Whisper 와 Qwen3-ASR 두개 모델로 전사하고 서로 WER 또는 CER 이 3% 미만인 샘플만 학습에 활용하도록 하자."
  - 사용자 "이번엔 서버에서 바로 받아줘" — 이번 건(VibeVoice·CallHome)만 mxc 에서 직접 받았다. 상시 규칙(T5 경유)은 그대로다.
- 결과:
  - CallHome 독일어: 120 통화 18.4 h(mono 16 kHz, 화자 구간 27,967, 2 인 104·3–4 인 16 통화), sha256 일치. 라이선스 CC BY-NC-SA 4.0.
  - 한 통화 스모크: 248 발화 중 두 교사 WER < 3 % 는 68 개(27 %, 발화 시간 약 12 %), 발화 WER 중앙값 0.33
  - Whisper 실행 환경(hibiki overlay venv)은 호스트 /tmp 가 비워져 휠로 다시 만들었다.
- Next: CallHome 전체 의사 전사 → 정렬·정제·mono·창·split, st4-m1 평가, VibeVoice 스모크
- By: tskim
