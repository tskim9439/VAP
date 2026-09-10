## [2026-09-11] query | Phase 2 실행 요약 — 일정·데이터 보유 현황·결정 사항

- Changed: `wiki/outputs/output-phase2-plan.md` 생성(정본 [[output-phase2-streaming-asr-diarization-plan]] 의 보조 페이지), `output-phase1-report.md` §7 이월 항목에 두 링크
- Reason: 사용자의 Phase 2 계획 요청에 두 세션이 동시에 계획을 썼다. 설계·관문은 정본으로 일원화하고, 이 페이지에는 정본이 "실측 전" 으로 남긴 서버 가용량(71631 248 h·otoSpeech 104.9 h·TurnBench dev 보유, CANDOR 없음)·달력 일정(Q0–Q4, 09-15 → 11-14)·혼합 비율·학습 비용·사용자 결정 4 건만 둔다
- Next: 사용자 결정(EN 대화 코퍼스, 71631 추가 반입, 단일 모델·항상 태그, turn 출력 형식) → Q0 착수. Switchboard/CallHome 시간축 복원 가능 여부 확인
- By: tskim
