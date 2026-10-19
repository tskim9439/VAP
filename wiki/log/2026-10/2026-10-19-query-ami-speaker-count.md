## [2026-10-19] query | AMI 창별 화자 수 — eval 20 s 창의 58 % 가 3 명 이상, 상위 2 명 밖 단어 7.8–11.1 %(20–40 s 창)

- Changed: raw [[2026-10-19-stage4-dialogue-translation]] "AMI 창별 화자 수" 절, `ami_speaker_count.py`·`ami-speaker-count.json`
- Reason: 사용자 질문 "우리도 화자 슬롯을 최대 4개로 할까??" → "한번 쟤봐". 슬롯 수를 정하기 전에 회의 데이터에서 2 슬롯이 얼마나 덮는지 쟀다.
- 결과:
  - AMI eval(16 회의) 20 s 창: 말한 화자 1·2·3·4 명 = 16.6 · 25.0 · 33.9 · 24.4 %. 화자 ≤ 2 창에 든 단어는 35 % 뿐이다.
  - 창마다 상위 두 화자 밖 단어는 20·30·40 s 창에서 7.8 · 9.8 · 11.1 % 다. 2 슬롯 모델로 회의를 디코드할 때 cpWER 의 바닥이다.
  - 정확히 2 명인 창은 eval 20 s 기준 397 개(2.2 h)다. 지금 형식으로 비교할 수 있는 부분 셋이다.
- Next: 슬롯 수 결정은 사용자에게 있다(권고: CST 1 차는 2 슬롯 유지, 회의 벤치마크는 Stage 4 뒤 K 슬롯 갈래). 바꾸면 [[decision-cst-training-stages-and-speaker-slots]] 개정과 팀 검토가 필요하다.
- By: tskim
