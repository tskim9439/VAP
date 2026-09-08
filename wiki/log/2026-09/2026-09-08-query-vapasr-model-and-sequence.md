## [2026-09-08] query | 현행 모델 구조와 시퀀스 규약 보고서

- Changed: `wiki/outputs/output-vapasr-model-and-sequence.md`
- Reason: 사용자 요청 — 현재 VAP-ASR 모델(인코더·adapter·thinker·특수 토큰)과 80 ms 청크 인터리브 시퀀스·라벨·손실·디코드·학습 설정을 코드 기준으로 한 곳에 정리. 인코더/thinker 설정은 서버의 .nemo·config.json, 스트림 통계는 정렬 캐시, 예시는 s1_show_sequence 출력으로 확인.
- Next: 인코더 해동·디코드 속도 개선 시 §2·§6 갱신.
- By: tskim
