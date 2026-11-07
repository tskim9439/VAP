## [2026-11-07] query | 같은 셈의 단일 화자 기준 — Nemotron 3.5 ASR 대 E13 (ESB AMI)

- Changed: [[output-ssa-papers-review-20261106]] 6 절 ③·판단 정정, summary. 코드 `raw/sources/experiments/2026-11-06-e15b/esb_ami_nemo.py`(신규 — NeMo 프롬프트 RNN-T 를 직접 forward).
- Reason: 사용자 "a, b, c 진행해보자"
- 발견
  - 같은 ESB AMI·같은 Whisper 셈으로 Nemotron 3.5 ASR(우리 인코더의 원 모델)은 청크 0.32 s 에서 16.69 %, 1.12 s 에서 15.04 % 다. E13 δ10 은 16.47 % 다.
  - 앞서 적은 "보고값 11.58 % 보다 약 5 pt 나쁘다"는 정규화·모델이 다른 수치와 견준 것이라 정정했다.
  - 삭제(8.7–9.4 %)는 두 모델에 공통이다. 우리만의 약점은 길게 통째로 빠뜨리는 삭제다. 오프라인(final)은 18.10 % 로 오히려 나쁘다.
- Next: 단일 화자 데이터 추가는 보류한다. 혼자 말 격차는 인코더 앞보기(0.32 대 1.12 s)·영어 전용 기반 모델 차이로 본다. 앞보기를 늘릴지는 지연과 바꾸는 결정이다.
- By: tskim
