## [2026-09-14] query | 시퀀스 8개 비판 검증 및 정본 개정

- Changed: [[output-phase2-streaming-asr-diarization-plan]], [[output-phase2-eot-review-framework]], [[output-phase2-sequence-critique-response]], probe/replay 실험 문서와 코드의 역사적 역할 표시, superseded 블록 명세의 테스트 기준 표현 수정.
- Reason: 사용자 비판에 따라 Q1 C-mode·A/B ID 재사용·TN/registry와 공용 Dataset/16사례 테스트·KO 1대화 QA를 우선하고 QC를 TurnBench dev+AMI 200경계로 축소했다. 실제 tokenizer/embedding 헤더·TN 표본을 읽어 판단 근거를 확인했다.
- Next: 정본 순서대로 실제 구현. C의 3초 지연·coverage, dev 튜닝 오염, HF config 미확인, head/KO 파이프라인 미구현을 완료 주장과 구분한다. 관련 변경만 로컬 커밋하고 원 음성·무관한 dirty 변경은 제외한다.
- By: tskim
