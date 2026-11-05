## [2026-11-05] task | Qwen3.8-LiveTranslate 를 우리 AMI·TAXI 채점으로 — 맥 로컬 평가 경로 준비(API 키 대기)

- Changed:
  - raw `raw/sources/experiments/2026-11-05-qwen-livetranslate/`(신규): `qlt_client.py`(실시간 WebSocket 클라이언트, 원본 이벤트 로그), `qlt_events.py`, `qlt_ami.py`(baseline_ami_ext 경로 재사용 — 표준 cpWER·DER·지연), `qlt_taxi.py`(목표 de·en 두 번, cst_stage4_taxi score 그대로), `qlt_tc.py`(meeteval tcpWER·tcORC-WER), `qlt_selftest.py`(가짜 서버·정답 로그), README
  - [[source-qwen38-livetranslate]]: 할 수 있는 것 → 진행 상태
- Reason: 사용자 "Qwen 3.8 live translate 도 동일하게 평가해보고, TAXI 에 대해서도 번역 성능을 평가해보고 싶은데 … 절대 mxc 서버에 올리지말고 로컬에서만 진행해야해." TAXI 전송은 사용자 판단 "추론만 하는거고 공개적으로 배포하는 것이 아니니 상관없을거 같은데?"
- 결과(API 없이)
  - 평가 입력은 mxc 에서 T5 로 읽어 오기만 했다(rsync pull, md5 대조). Qwen 코드·로그·결과는 mxc 에 두지 않는다.
  - 정답 로그 점검: AMI 표준 cpWER 0.40 · DER 0.61 · tcORC-WER 0.43, TAXI BLEU·chrF 100 · WER 0.
  - 우리 agn 은 API 에 단어 시각이 없어 정답 로그에서도 9.5 % 가 나온다 → 화자 무시 비교는 tcORC-WER(collar 5 s)로 한다. 같은 셈으로 E13 + G2b eval tcORC-WER 22.34 / 29.38(IHM-mix / SDM), 맥에서 표준 cpWER 24.48 / 36.10 재현.
- Next: 사용자가 `.env.local` 에 `DASHSCOPE_API_KEY` 를 넣으면 AMI 한 회의 시험 → AMI 전체(오디오 18.1 h, 동시 8 이면 약 2.3 h) → TAXI(1.5 h), 결과를 외부 기준 표에 넣는다.
- By: tskim
