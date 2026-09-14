## [2026-09-13] query | Phase 2 자동 EOT 검수 프레임워크 설계

- Changed: [[output-phase2-eot-review-framework]] 신규; [[output-phase2-real-sequence-probe]]의 확정 오답으로 읽힐 수 있는 표현 보완; [[output-phase2-streaming-asr-diarization-plan]] Q0 검수 설계 링크 추가.
- Reason: 자동 EOT 추가 검수를 수행 가능한 프레임워크로 설계해 달라는 요청. 파일 기반 CLI·청취 팩·독립 판정·누락 감사·잠금 평가·릴리스·event-complete 학습 계약을 정의했다.
- Next: F0–F1 구현 후 AMI 20개로 사람 검수 저장/재가져오기 루프 확인. 본 작업은 설계이며 검수 프로그램·사람 판정·학습 실행은 하지 않았다.
- By: tskim
