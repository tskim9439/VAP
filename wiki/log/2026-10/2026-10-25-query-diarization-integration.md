## [2026-10-25] query | 화자 분리 통합 검토(Streaming Sortformer · Nemotron 3 Diarization · SSA · FastMSS)

- Changed: [[output-diarization-integration-20261025]](새), [[status]]
- Reason:
  - 사용자 "우리도 Diarization 기능을 학습해 추가하는 건 어떨까? 통합하는 방법이 있을까?" → "2605.15442 · 2507.18446 · nemotron 3 diarization 논문을 참고 해봐".
  - 원문을 읽고 정리했다: Streaming Sortformer(AOSC), Nemotron 3 Diarization 모델 카드(AMI SDM DER 12.8 @1.04 s), Multitalker Parakeet(AMI SDM cpWER 37.4, 화자별 인스턴스), SSA(arXiv:2506.22646, pre-encode 주입), FastMSS(arXiv:2605.15442).
  - AMI SDM 오류 세분(귀속은 화자 합치기, 받아쓰기 오류 절반은 겹침)과 맞춰, 화자 분리 가지 + SSA 주입 + 슬롯별 디코드 + AOSC 설계와 E0–E4 시험 순서를 제안했다.
- Next: 사용자가 E0(분석)·E1(화자 분리 가지)·E2(상한) 중 어디서 시작할지, 공개 가중치 초기화를 쓸지 정한다.
- By: tskim
