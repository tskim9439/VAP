---
name: wiki-ingest
description: 새 원천 자료(문서, 링크, 트랜스크립트, 회의록, PDF)를 raw/ 에서 읽어 wiki/ 로 합성한다. 사용자가 자료를 추가하거나 가리키며 "ingest", "정리해줘", "위키에 넣어줘", "회의록 정리"라고 할 때 사용한다. raw/inbox/ 미분류 파일 분류도 이 스킬이 처리한다.
---

# Ingest Workflow

`AGENTS.md` 의 Ingest Workflow 를 실행하는 스킬이다. **정본은 `AGENTS.md`** 이며,
충돌 시 `AGENTS.md` 가 이기고 이 파일을 유지보수 PR 로 고친다.

## Pre-flight

1. `.llm-wiki-local/user.yaml` 을 읽어 `member_id` 와 `branch_prefix` 를 확보한다.
   없으면 `scripts/init-local-user.sh` 실행을 사용자에게 요청한다.
2. 작업 브랜치를 만든다: `scripts/sync-user-branch.sh ingest <source-name>`
   (`main` 에서 직접 작업하지 않는다.)
3. `raw/inbox/` 에 파일이 있으면 **Raw File Classification** 을 먼저 수행한다 (아래).
4. `wiki/index.md`, `wiki/status.md` 를 읽어 현재 상태를 파악한다.

## Raw File Classification

`raw/inbox/` 의 미분류 파일을 내용 기반으로 분류해 이동시킨다.
판단 순서는 확장자 → 파일명 → 파일 내용.

| 유형 | 판단 기준 | 이동 경로 |
|------|-----------|-----------|
| 회의록 / 토론 덤프 | 회의, 안건, 참석자, 액션 아이템 키워드 또는 대화체 구조 | `raw/meetings/` |
| 연구 자료 / 기사 | 논문, 기술 문서, 웹 기사, 링크, 슬라이드, 정리 노트 | `raw/sources/` |
| 이미지 / 미디어 / 바이너리 | `.png .jpg .gif .mp4 .mp3 .wav .pptx .xlsx` 등 | `raw/assets/` |
| 신규 유형 | 위 세 범주에 맞지 않음 | `raw/<유형>/` 새 폴더 (lowercase kebab-case) |

절차:

1. `raw/inbox/` 파일 목록을 확인한다.
2. 텍스트는 전체를 읽고, 바이너리/미디어는 확장자와 파일명으로 판단한다.
3. 이동 경로를 결정한다. 맞는 폴더가 없으면 새로 만든다.
4. 불확실한 파일은 이동 전에 사용자에게 확인한다.
5. 결과를 보고한다 — 이동 경로와 신규 폴더 목록 포함.
6. 이어서 ingest 를 진행할지 사용자에게 묻는다.

Hard constraints:

- 파일 내용은 읽기만 하고 수정하지 않는다.
- `raw/` 외부로 이동하지 않는다.
- 복사가 아니라 **이동** 이다. 분류 후 원본은 한 경로에만 존재해야 한다 (`git mv` 또는 `mv`).

## 합성 절차

1. 원천 자료를 읽는다. 30페이지(또는 수 시간 트랜스크립트) 이상이면 파트로 나눠
   세션을 걸쳐 ingest 하고, 분할 사실을 source note 에 기록한다.
2. `wiki/sources/` 에 source note 를 생성하거나 갱신한다
   (템플릿: `wiki/templates/source-note.md`).
3. 지속적 개념을 `wiki/concepts/` 로 추출한다.
4. 반복 등장할 인물·조직·제품·프로젝트를 `wiki/entities/` 로 추출한다.
5. 지식을 복제하지 말고 **기존 페이지를 갱신** 한다.
6. 유용한 곳에 양방향 교차 링크를 추가한다.
7. 큰 그림이 바뀌면 `wiki/overview.md` 를 갱신한다.
8. 회의록/토론 덤프인 경우 추가로:
   - `wiki/meetings/` 에 정리된 회의록 생성
   - 지속적 결정을 `wiki/decisions/` 로 분리 (회의록 안에 결정을 묻지 않는다;
     회의록에서 링크한다). 결정 페이지는 병합 시 메인테이너 리뷰 필요.
   - 액션 아이템마다 `wiki/tasks/` 에 태스크 파일 생성 (owner, due 포함).
9. 신규/변경 페이지의 `summary` frontmatter 를 정확히 유지한다.
   **`wiki/index.md` 는 직접 편집하지 않는다** (생성물).
10. `wiki/status.md` 에 다음 액션과 미해결 이슈를 갱신한다.
11. `wiki/log/YYYY-MM/YYYY-MM-DD-ingest-<topic-slug>.md` 로그 샤드를 추가한다.

## 로그 샤드 형식

```markdown
## [YYYY-MM-DD] ingest | 원천 자료 제목

- Changed: 변경된 파일 또는 페이지 그룹
- Reason: 갱신 이유
- Next: 후속 작업
- By: member-id
```

제목 줄의 `ingest` 키워드는 파싱 대상이므로 영어로 유지하고, 본문은 한국어로 쓴다.

## 원칙

- `raw/` 는 진실의 원천이다. 사용자가 명시적으로 요청하지 않는 한 재작성·재편성하지 않는다.
- 주장에는 출처가 필요하다. source note, raw 파일, 외부 URL 로 되돌아 링크하고
  불확실성은 명시한다.
- 긴 원문 인용을 피하고 자기 말로 요약한다. 웹 자료는 관측 일자와 URL 을 기록한다.
- 하나의 원천이 여러 페이지를 건드리는 것은 정상이다. 고립된 요약 하나보다
  정확성과 통합을 우선한다.
- 한국어로 쓴다. 파일명, frontmatter 필드명, raw 경로, URL, 논문 제목, 모델명,
  벤치마크명, 위키링크 슬러그는 원형을 유지한다.
