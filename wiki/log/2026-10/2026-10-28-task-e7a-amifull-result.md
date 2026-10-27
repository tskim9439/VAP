## [2026-10-28] task | E7a 회의 통째 cpWER(96172) — 받아쓰기는 그대로, 귀속 무너짐

- Changed:
  - [[output-diarization-integration-20261025]] 7 절 결과 3
  - [[output-ami-external-comparison-20261027]] 회의 통째 절
  - mxc `results/stage3-eval-amifull-c30-aosc-e7a-q17-e6-final-wn`(표준 정규화 재채점)
- Reason:
  - 사용자 "96172 끝나면 비교 결과 정리해줘".
  - AMI eval δ10 회의 통째 cpWER(표준)는 IHM-mix 67.34·SDM 73.10 이다. agn 21.06·27.36 은 창 단위와 같고, 귀속이 46 pt 다.
  - E7a 헤드의 회의 단위 DER(32–40)보다 훨씬 크다. LLM 이 회의 단위 슬롯 입력을 따르지 않아 한 화자의 말이 슬롯 넷으로 흩어진 것으로 본다.
- Next:
  - 96461(E6 헤드로 조각마다 새로·AOSC)과 견준다.
  - 사후 진단으로 가설 단어를 헤드 활동으로 다시 귀속해 cp 를 잰다. 디코드 때 태그를 활동으로 이끄는 방식이나 E7b(LLM 회의 단위 슬롯 학습)를 정할 근거다.
- By: tskim
