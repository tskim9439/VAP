## [2026-09-22] ingest | speechlm_v1_encoder_alignment_2607 ASR 카탈로그 검토

- Changed: `wiki/sources/source-speechlm-v1-encoder-alignment-2607.md`, `wiki/outputs/output-speechlm-arrow-asr-review.md`, `wiki/status.md`
- Reason: 다른 프로젝트의 Arrow·tar ASR 자원을 현재 승인 2,356시간 학습셋과 겹치지 않게 재사용할 수 있는지 읽기 전용으로 조사했다. 동일 코퍼스 6종과 YODAS en129 부분 중복, 다수 AI Hub 8 kHz 사본, Arrow에 word alignment가 없다는 점, MLS 폴더의 비참조 Common Voice tar 혼재를 확인했다.
- Next: P0 source별 결정적 1,000행 probe에서 Arrow→tar resolve, PCM hash 중복, Qwen–Whisper 5% 합의, forced alignment 통과율을 측정한다.
- By: tskim

