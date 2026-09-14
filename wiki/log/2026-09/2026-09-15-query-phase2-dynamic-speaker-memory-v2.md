## [2026-09-15] query | 동적 화자 메모리 대안 실행안 v2

- Changed: [[output-phase2-dynamic-speaker-memory-plan-v2]] 신규; `raw/sources/experiments/2026-09-15-phase2-lane-sim/`(AMI·ICSI·NOTSOFAR-1 어노테이션 lane 시뮬레이션 스크립트·출력).
- Reason: 원안 [[output-phase2-dynamic-speaker-memory-plan]]을 구체화·검증해 실행 계획으로 만들라는 요청. lane 즉시 해제를 lazy-free로 바꿔 N≤R이면 정본 K슬롯과 동일 시퀀스가 되게 하고, 실측으로 R=6·EOT lane 결합 규칙(pointer head 보류)·단일 병렬 학습을 도출했다. NOTSOFAR-1의 화자별 close-talk·단어 시각을 확인해 A등급으로 편입했다.
- Next: D0 fixture·allocator 동등성 테스트, D2 E2 화자 표현 probe(SLURM 1 job), 정본 Q0 K 결정에 R=6 반영 여부는 사용자 결정. 정본·registry·코드는 변경하지 않았다.
- By: tskim
