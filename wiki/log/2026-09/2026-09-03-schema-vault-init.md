## [2026-09-03] schema | 볼트 스키마 초기화

- Changed: `raw/{inbox,sources,meetings,assets}`, `wiki/` 전체 디렉토리와 시드 페이지,
  `scripts/{init-local-user,pull-safe,sync-user-branch,install-skills}.sh`,
  `.skills/{wiki-ingest,wiki-query,wiki-lint,wiki-merge,wiki-task}/SKILL.md`,
  `.gitignore`, `.gitattributes`
- Reason: `AGENTS.md` 의 Directory Contract, Agent Skills, Git Collaboration Policy 에
  따라 빈 저장소에 볼트 골격을 구성했다. 스킬은 `.skills/` 에 에이전트 중립 정본으로
  두고, 런타임 경로(`.claude/skills/`, `.codex/skills/`)는 `install-skills.sh` 가
  만드는 git-ignored 링크다.
- Next: 팀원별 `.llm-wiki-local/user.yaml` 생성, 원격 저장소 연결과 `main` 보호 설정,
  대용량 바이너리 정책 결정, 첫 원천 자료 ingest
- By: unknown
