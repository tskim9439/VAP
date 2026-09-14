## [2026-09-13] query | AMI 실제 데이터의 Phase 2 블록 시퀀스 생성

- Changed: [[output-phase2-real-sequence-probe]], `experiments/p2_sequence_probe.py`, `raw/sources/experiments/2026-09-13-phase2-sequence-probe/` 신규 산출물; [[output-phase2-streaming-asr-diarization-plan]] §9에 검증 사례 링크 추가.
- Reason: 사용자 요청으로 AMI 4화자 38.4초를 480개 80ms 블록으로 직렬화하고 mono crop·실제 BPE ID·학습 라벨·블록 표를 생성했다. 전사 복원 검증과 자동 EOT의 marketing…expert 재개 오류 후보를 기록했다.
- Next: VAD/정렬 비교, 재개·짧은 겹침 EOT 검수, TN·특수 토큰·encoder 연결. 실제 모델 학습·추론 결과와 혼동하지 않는다.
- By: tskim
