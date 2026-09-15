## [2026-09-15] decision | Phase 2 정본을 lane 유지형 계획으로 교체

- Changed: [[output-phase2-lane-plan]] 신규(정본); [[decision-phase2-canonical-lane-plan]] 신규; [[output-phase2-streaming-asr-diarization-plan]] status superseded + 상단 안내(참조로 유효한 절 명시); [[output-phase2-dynamic-speaker-memory-plan-v2]]·[[output-phase2-lane-proposal-report]]·[[decision-eot-immediate-soft-label]] 정본 링크; 아티팩트 갱신.
- Reason: 사용자 결정 "이번 계획은 마음에 들어. 이걸 정본으로 설정해줘". lazy-free R=6, EOT 즉시 soft label, 외부 재매핑, 모델 내 화자 메모리 범위 밖.
- Next: D0(allocator·parser·soft collator·fixture)와 D2(E2 화자 표현 probe) 착수. 다른 세션의 untracked 페이지(외부 재매핑 두 편·원안·비교)는 분석 기록으로 남기며 그쪽 세션이 커밋한다.
- By: tskim
