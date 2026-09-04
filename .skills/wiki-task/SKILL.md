---
name: wiki-task
description: 팀 태스크를 wiki/tasks/ 에 파일 단위로 만들고 상태를 갱신한다. 태스크 생성·할당·완료·조회, "할 일 추가", "태스크 상태 바꿔줘", "내 할 일 뭐 있어", 회의 액션 아이템 등록 시 사용한다. 조회는 읽기, 쓰기는 task/ 브랜치를 통한다.
---

# Task Workflow

`AGENTS.md` 의 Team Execution Layer 를 실행하는 스킬이다. **정본은 `AGENTS.md`** 이다.

태스크는 별도 시스템이 아니라 지식이다. 한 태스크 = 한 파일이며, 파일 샤딩 덕분에
여러 사람이 동시에 상태를 갱신해도 충돌하지 않는다.

## 조회 (읽기 — 브랜치 불필요)

1. `wiki/todo.md` 를 먼저 읽는다 (생성된 대시보드).
2. 최신 상태가 필요하면 `wiki/tasks/*.md` 의 frontmatter 를 직접 읽는다.
   `todo.md` 는 병합 시점에 생성되므로 브랜치 작업 중에는 뒤처질 수 있다.

```bash
grep -H -E '^(status|owner|due|priority|summary):' wiki/tasks/*.md
```

## 생성 / 갱신 (쓰기 — `task/` 브랜치)

1. 브랜치: `scripts/sync-user-branch.sh task <task-slug>`
2. `wiki/tasks/<task-slug>.md` 를 만들거나 수정한다.
3. `wiki/log/YYYY-MM/YYYY-MM-DD-task-<task-slug>.md` 로그 샤드를 추가한다.
4. **`wiki/todo.md` 는 직접 편집하지 않는다** (생성물 — 병합 시 재생성).

## 태스크 파일 형식

```markdown
---
type: task
status: open        # open | doing | done | blocked
owner: member-id
due: YYYY-MM-DD
priority: p1        # p0 | p1 | p2 | p3
created: YYYY-MM-DD
updated: YYYY-MM-DD
summary: 한 줄 요약 (최대 120자, todo/index 생성에 사용)
sources:
  - [[source-note]]
---

# 태스크 제목

## 배경
왜 필요한가. 관련 개념·출처·회의·결정을 위키링크로 연결한다.

## 완료 조건
- [ ] 확인 가능한 조건

## 진행 기록
- YYYY-MM-DD: 상태 변화와 사유
```

파일명은 lowercase kebab-case, 내용을 식별할 수 있게 짓는다 (`task-3` 금지).

## 규칙

- **파일을 삭제하지 않는다.** 완료는 `status: done` 으로 표시해 이력을 남긴다.
  취소도 마찬가지로 `done` 또는 `blocked` + 사유를 기록한다.
- 상태를 바꾸면 `updated` 를 갱신하고 진행 기록에 한 줄 남긴다.
- 태스크는 의존하는 개념·출처 페이지를 위키링크한다. "왜" 가 한 홉 거리에 있어야 한다.
- 회의에서 나온 액션 아이템은 회의록 페이지를 `sources` 에 넣고 회의록에서도 역링크한다.
- owner 는 `.llm-wiki-local/user.yaml` 의 `member_id` 형식을 쓴다. 모르면 `unknown`.

## 로그 샤드

```markdown
## [YYYY-MM-DD] task | 태스크 제목

- Changed: wiki/tasks/<task-slug>.md
- Reason: 생성 사유 또는 상태 변경 사유
- Next: 다음 단계
- By: member-id
```
