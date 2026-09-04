---
name: wiki-merge
description: 풀 리퀘스트 병합 전 절차. 브랜치를 main 으로 갱신하고, 충돌은 텍스트가 아니라 의미 단위로 재합성하며, 리뷰 체크리스트를 확인하고, 파생 파일(index.md, log.md, todo.md)을 재생성해 커밋한다. 병합·PR 준비·충돌 해결·인덱스 재생성 요청 시 사용한다.
---

# Pre-merge Procedure

`AGENTS.md` 의 Pre-merge procedure 와 Merge And Conflict Resynthesis 를 실행한다.
**정본은 `AGENTS.md`** 이다.

## 순서

1. 브랜치를 최신 `main` 으로 갱신한다 (`git fetch origin && git merge origin/main`
   또는 rebase). `main` 체크아웃·푸시는 사용자가 `main` 을 명시적으로 지목했을 때만 한다.
2. 충돌이 나면 아래 **충돌 재합성** 을 수행한다.
3. 리뷰 체크리스트를 확인한다.
4. **파생 파일을 재생성** 하고 커밋한다.
5. 병합한다. 그 결과 `main` 은 항상 일관된 인덱스와 로그 다이제스트를 갖는다.

## 리뷰 체크리스트

- [ ] `raw/` 원천 파일이 보존되었고, 명시적 요청 없이 재작성되지 않았다.
- [ ] 위키 페이지의 provenance, 인용, 유용한 역링크가 있다.
- [ ] 변경된 페이지에 끊어진 위키링크가 없다.
- [ ] 자격증명, 사적 정보, 불필요한 대용량 파일이 없다.
- [ ] `.llm-wiki-local/user.yaml` 등 무시 대상 로컬 파일이 스테이징되지 않았다.
- [ ] 생성 파일을 손으로 편집하지 않았다.

추가 리뷰가 필요한 변경 (다른 팀원 승인 필수):
`AGENTS.md`, `scripts/`, `.skills/`, `wiki/decisions/` 하위 페이지, 재합성된 페이지.

## 충돌 재합성

마크다운 충돌은 텍스트가 아니라 **의미** 다. 충돌 마커를 줄 단위로 손으로 풀지 않는다.

1. 충돌 페이지마다 세 버전을 읽는다:
   ```bash
   git show :1:<path>  # merge base
   git show :2:<path>  # ours
   git show :3:<path>  # theirs
   ```
2. 양쪽의 모든 주장, provenance 링크, frontmatter 를 보존하는 **하나의 페이지** 로 합성한다.
   양쪽이 모순되면 페이지 안에 불확실성 노트로 명시하고, 필요하면
   `wiki/questions/` 항목을 새로 만든다.
3. 재합성된 페이지는 병합 전 다른 팀원의 리뷰가 필요하다.
4. `resynth` 로그 샤드를 추가한다 — 양쪽 소스 브랜치명과 건드린 페이지를 적는다.

## 파생 파일 재생성

`wiki/index.md`, `wiki/log.md`, `wiki/todo.md` 는 빌드 산출물이다.
스크립트 없이 **에이전트가** 재생성한다. 각 파일은 다음 줄로 시작한다:

```
<!-- generated: do not edit -->
```

### wiki/index.md

각 페이지의 `summary` frontmatter 로 조립한다 (없으면 본문 첫 문장, 120자에서 절단).
Directory Contract 순서대로 디렉토리별 그룹핑:
overview → status → concepts → entities → sources → questions → outputs →
meetings → decisions → tasks → templates → _maintenance.

```markdown
<!-- generated: do not edit -->
# 인덱스

마지막 생성: YYYY-MM-DD

## Concepts
- [[page-slug]] — 한 줄 요약
```

`wiki/index.md` 가 약 300줄을 넘으면 계층 인덱스로 전환한다: 최상위 인덱스는
디렉토리당 한 줄로 디렉토리별 인덱스 파일을 링크한다. 전환은 `schema` 로그 샤드로 기록한다.

### wiki/log.md

`wiki/log/` 의 샤드를 **최신순** 으로 이어붙인 다이제스트.

### wiki/todo.md

`wiki/tasks/` frontmatter 에서 `open` / `doing` / `blocked` 태스크를 모아
owner 별로 그룹핑하고 due 오름차순 정렬. `done` 은 제외한다 (파일은 삭제하지 않는다).

```markdown
<!-- generated: do not edit -->
# TODO

마지막 생성: YYYY-MM-DD

## member-id
| 상태 | 우선순위 | 마감 | 태스크 |
|------|----------|------|--------|
| doing | p1 | YYYY-MM-DD | [[task-slug]] — 요약 |
```

## 주의

- 생성 파일을 손으로 편집하지 않는다. 재생성이 정답이다.
- `main` 은 운영상 보호된다. 사용자가 `main` 과 수행할 작업을 명시적으로 지목했을 때만
  checkout / pull / merge into / push 한다.
