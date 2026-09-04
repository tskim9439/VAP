## [2026-09-03] task | 연구 실행 체크리스트 TODO.md 와 환경 태스크 2건

- Changed: `TODO.md`(신규, 저장소 루트), `wiki/tasks/task-setup-training-environment.md`(신규),
  `wiki/tasks/task-checkpoint-retention-policy.md`(신규), `wiki/todo.md`·`wiki/index.md` 재생성,
  `wiki/status.md` 카운트 갱신.
- Reason: 사용자가 연구 수행용 ToDo 리스트 파일을 요청했다. 생성 대시보드 `wiki/todo.md` 는
  owner 별 표라 단계·관문·의존성이 보이지 않아, 루트에 Phase 0~5 + 논문 + 결정 관문을 담은
  실행 체크리스트를 별도로 두었다. 상세는 태스크 파일이 정본이며 체크리스트는 그 링크다.
  실측에서 드러난 환경 공백(torch 미설치, /data4 여유 575G)이 태스크로 빠져 있어 2건을 추가했다.
- Next: `TODO.md` 는 Directory Contract 에 없으므로 `AGENTS.md` 유지보수 PR 에 함께 반영.
  Phase 0 p0 5건 착수.
- By: tskim
