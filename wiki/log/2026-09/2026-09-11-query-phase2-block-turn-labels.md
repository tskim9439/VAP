## [2026-09-11] query | Phase 2 무음·단독·겹침 블록 및 화자별 start/end 토큰 명세

- Changed: `output-phase2-block-and-turn-label-spec`, `output-phase2-streaming-asr-diarization-plan` v1.2, `output-phase2-plan`, `source-muse-voice-transcribe`
- Reason: 사용자 요청에 따라 18가지 블록 사례, 화자별 독립 OPEN/CLOSED 상태, start/end 공동 생성, 실제 label 생성·관측 시각·loss·EOF·부분 감독 계약을 구체화했다. Muse 공식 token 표기와 우리 확장을 구분했다.
- Next: Q0 라벨 생성기·serializer·상태 parser 및 fixture 구현, 신뢰도 높은 자연 turn 창 검수. 이번 작업은 계획·명세이며 코퍼스 전체 라벨 생성이나 모델 학습을 수행하지 않았다.
- By: tskim
