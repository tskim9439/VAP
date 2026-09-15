## [2026-09-15] decision | EOT 즉시 방출 + soft label

- Changed: [[decision-eot-immediate-soft-label]] 신규; [[output-phase2-dynamic-speaker-memory-plan-v2]] §2 갱신·§11 추가(SEG_END 제거, EOT 후보 위치·p_end 표·손실·평가); [[output-phase2-lane-proposal-report]]·아티팩트 갱신; `raw/sources/experiments/2026-09-15-phase2-lane-sim/segment_end_outcomes.out`(AMI·ICSI·NOTSOFAR·71631 구간 끝 결과 분류).
- Reason: 사용자 결정 — C-mode 3 s 대기가 너무 길다. SEG_END 를 EOT 로 치환하고 soft label 로 학습해 lane 이 끊어지면 바로 EOT 로 인식한다.
- Next: 정본 §6.3·§4.4·§6.1·§8 개정(사용자 확인 후), Q0 에서 p_end 표 동결, D1 에 유지 구간 내 오방출률 관문 추가.
- By: tskim
