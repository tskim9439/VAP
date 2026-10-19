## [2026-10-19] task | 대화 발화 단독 디코드 대조 — 한국어는 화자 채널만 주면 CER 8–11 %(단일 화자 수준), 영독은 13.5–18 %

- Changed:
  - `experiments/cst_isolated_asr.py`(새): Stage 3 회귀 창의 전사 발화를 mix·chan 두 조건으로 잘라 단일 화자 manifest 를 만들고, 대화 평가와 같은 단위로 채점한다.
  - raw [[2026-10-19-stage4-dialogue-translation]]: "대화 발화 단독 디코드 대조" 절, `isolated_chain.sh`, 집계 `isolated/`
  - [[output-stage4-prep-plan-20261019]] S6 결과 절의 원인 설명
- Reason: 사용자 질문 "stage 2 에서는 한국어 cer 이 8%, 영어 wer 이 3% 정도 나왔는데, 지금은 30% 이상이 나온다는게 이상해." → "그래 호스트 gpu 로 돌려"
- 결과(st4-p1 final, δ10):
  - 발화 12,053 개(마스크 3,586 개 제외), mxc 호스트 GPU 2·4 에서 두 모델 각 51 분
  - chan(화자 채널만): 한국어 8.4–11.3, 영어 16.5, 독일어 13.5–18.2
  - mix(혼합 오디오 구간): 한국어 14.8–19.2, 영어 25.2, 독일어 18.0–24.7
  - 대화 디코드 agn 은 71631 을 빼면 mix 보다 낮다. 귀속 몫(cp − agn)은 3.6–9.6 pt
  - Stage 2 최종 대비 st4-p1 은 영독 chan 에서 1.9–3.9 pt 낫고, 한국어는 같다
- Next: VibeVoice 외부 기준선(업로드 대기), 체크포인트 선택(사용자)
- By: tskim
