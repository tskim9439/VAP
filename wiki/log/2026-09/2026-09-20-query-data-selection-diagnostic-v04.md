## [2026-09-20] query | 방송·AMI·VoxPopuli 우선 진단 실행

- Changed: `output-data-selection-diagnostic-v04.md`, 실행 계획 역링크, `wiki/status.md`; `selection_review.py`, `selection_report.py`, `selection_diagnose_v04.py`, 보호 테스트; 신규 raw 진단/청취/재판정 산출물.
- Reason: 사용자의 진행 요청으로 세 진단을 순서대로 수행했다. 방송 반복의 원 발음전사 기원, AMI 메타데이터 일치, VoxPopuli 알려진 저장소의 원 세션 미발견을 확인했다. 사람 실패 11개를 실제 재판정에서 보류했다. wiki-query 절차로 근거와 한계를 연결했다.
- Next: 정확한 정정문과 AMI 증폭/문맥 검수, 원 세션 필요 범위 확인. 향후 전량 판정에도 검수 인자 필수 연결. 기존 학습 입력은 교체하지 않음.
- By: tskim
