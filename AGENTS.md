# LLM-Wiki-Collab Schema

This vault implements Andrej Karpathy's LLM Wiki pattern: raw sources are kept
immutable, while the LLM maintains a persistent, interlinked Markdown wiki.

The vault is operated entirely by **CLI agents on flat-rate subscriptions**
(e.g. codex Code) plus Git hosting. There is no server, no database, no
embedding index, and no extra local tooling beyond Git and the agent itself.
Retrieval is navigation: read `wiki/index.md`, search the wiki with the
agent's built-in text search, and follow wikilinks.

The workflows are packaged as agent skills under the agent-neutral
`.skills/` directory and ship with the repository, so every member's agent —
codex Code, Codex, or any SKILL.md-compatible runtime — runs the same
procedures (see Agent Skills for installation).

## Operating Principles

1. `raw/` is the source of truth. Read from it, but do not rewrite or reorganize
   source files unless the user explicitly asks.
2. `wiki/` is LLM-owned synthesis. Create, revise, link, and reorganize pages
   here as knowledge accumulates.
3. `AGENTS.md` is the schema. Follow it before changing the wiki, and update it
   when the workflow itself improves. Schema changes go through a pull request
   and a `schema` log entry.
4. `wiki/index.md` is content-oriented and **generated**. Never edit it
   directly. Instead, keep every page's one-line `summary` frontmatter
   accurate; the index is assembled from those summaries (see Derived Files).
5. The activity log is **sharded**. Append-only entries live as individual
   files under `wiki/log/`; `wiki/log.md` is a generated digest. Never edit
   generated files directly.
6. Claims need provenance. Link back to source notes, raw files, or external
   URLs. Mark uncertainty explicitly.
7. Humans should generally write rough notes, transcripts, and pasted
   discussions into `raw/` first. The LLM then curates those rough inputs into
   clean `wiki/` pages. Direct human edits to `wiki/` are allowed for small
   corrections, but the default collaboration model is human-provided raw
   evidence plus LLM-maintained synthesis.
8. No external infrastructure. Do not introduce databases, embedding stores,
   background servers, paid per-call APIs, or required local tools into the
   workflow. If a future scale problem appears, solve it with vault
   conventions first (e.g. hierarchical indexes) and record the change as a
   `schema` log entry.

## Language Policy

- Maintained wiki synthesis, reports, source notes, status pages, health reports,
  templates, and durable outputs should be written in Korean by default.
- Keep filenames, frontmatter field names, raw paths, URLs, paper titles, model
  names, benchmark names, and Obsidian wikilink slugs in their original form
  unless there is a clear reason to change them.
- English source material should be summarized in Korean. Avoid long verbatim
  quotes; preserve precise terms in parentheses when useful.
- Log headings should keep the parseable English operation keyword
  (`ingest`, `query`, `task`, `lint`, `schema`, `resynth`) while the entry
  body is written in Korean.

## Agent Runtime And Cost Policy

The team uses flat-rate subscription CLI agents. Marginal LLM calls cost
nothing, but subscriptions have rate limits and external infrastructure costs
real money and maintenance. Therefore:

- Retrieval is **navigation, not embedding search**: `wiki/index.md` first,
  then the agent's built-in text search, then read pages and follow wikilinks.
  Do not generate embeddings or call external retrieval APIs.
- Prefer extra agent turns over extra tooling. Re-reading an index twice is
  free; installing and maintaining tools is not.
- **Chunk large ingests.** Split sources larger than roughly 30 pages (or
  several hours of transcript) into parts and ingest them across sessions to
  stay within rate limits. Note the split in the source note.
- **Lint rotation.** Run the Lint Workflow about once a week. The team assigns
  a rotating member, recorded in `wiki/status.md`; the assignee runs it with
  their own agent and opens a maintenance PR with findings and fixes.
- **Index scaling.** When `wiki/index.md` grows past roughly 300 lines, switch
  to hierarchical indexes: the top index lists one line per directory linking
  to per-directory index files, which are also generated. Record the switch as
  a `schema` log entry.

## Agent Skills

The four operating procedures are registered as project-level agent skills,
versioned in this repository:

- `.codex/skills/wiki-ingest/SKILL.md` — Ingest Workflow
- `.codex/skills/wiki-query/SKILL.md` — Query Workflow
- `.codex/skills/wiki-lint/SKILL.md` — Lint Workflow
- `.codex/skills/wiki-merge/SKILL.md` — Pre-merge procedure, including
  conflict resynthesis and derived-file regeneration

Rules:

- Skills are the **executable packaging** of the workflows; this `AGENTS.md`
  is the **canonical definition**. On any conflict, `AGENTS.md` wins, and the
  skill must be fixed via a maintenance PR.
- Because skills live in the repository, registration is automatic: cloning
  the vault installs them for every member. No per-member setup is needed.
- Changing a skill is a schema change: it requires a pull request reviewed by
  another team member and a `schema` log shard.
- Agents that do not support the skill format should follow the workflow
  sections of this file directly; the skills add no behavior beyond them.

## Git Collaboration Policy

Git collaboration is an operational concern, not knowledge content. This section
is the canonical Git collaboration plan for the vault.

- Use a private Git repository as the canonical remote unless the project owner
  explicitly decides otherwise.
- Repository visibility and large binary policy are project-owner decisions.
- Treat `main` as the shared, lint-clean branch. Protect it on the Git host:
  no direct pushes, pull requests required.
- Store local user identity in `.llm-wiki-local/user.yaml`. This file is
  machine-local and must not be committed or pushed.
- If `.llm-wiki-local/user.yaml` is missing during project initialization, ask
  the user for the minimum local identity fields and create it. Prefer
  `scripts/init-local-user.sh` for this setup.
- When updating the local repository from Git, preserve local user identity.
  Prefer `scripts/pull-safe.sh`, which refuses to pull if `.llm-wiki-local/`
  is tracked by Git locally, is present in the upstream tree, or if local
  identity has not been initialized.
- Use user-prefixed small branches for normal work. Read `branch_prefix` from
  `.llm-wiki-local/user.yaml` and create branches as:
  - `<branch_prefix>/ingest/<source-name>` for source ingestion, including
    meeting-note synthesis.
  - `<branch_prefix>/query/<topic>` for durable output pages created from
    questions.
  - `<branch_prefix>/task/<task-slug>` for task creation and status updates.
  - `<branch_prefix>/maintenance/<topic>` for schema, guide, lint, or cleanup
    work.
- When syncing with Git, push the current user-prefixed branch to `origin` by
  default. Do not push directly to `main`. Prefer
  `scripts/sync-user-branch.sh <work-type> <topic>` for this operation.
- `main` is protected operationally. Only checkout, pull, merge into, or push
  `main` when the user gives an explicit command that names `main` and the
  requested operation. The sync script requires `--allow-main` for an explicit
  main push.
- Changes to `AGENTS.md`, `scripts/`, or `.skills/`, any page under
  `wiki/decisions/`, and any resynthesized page (below), require review by
  another team member before merge.
- Keep large binary policy explicit. PDFs can stay in Git while the corpus is
  small; audio, video, and large artifacts should use Git LFS or separate
  storage after the team decides.
- `wiki/` may mention that Git collaboration exists, but detailed Git workflow
  belongs in this `AGENTS.md` section rather than a maintained knowledge page.

### Pre-merge procedure

Whoever merges a pull request follows this order, using their own agent:

1. Update the branch with the latest `main`.
2. If conflicts appear, perform Merge And Conflict Resynthesis (below).
3. Review checklist:
   - `raw/` source files were preserved and not rewritten unless explicitly
     requested.
   - Wiki page provenance, citations, and useful backlinks are in place.
   - No dead wikilinks in changed pages.
   - No accidental credentials, private data, or unnecessary large files.
   - `.llm-wiki-local/user.yaml` and other ignored local files are not staged.
   - Generated files were not hand-edited.
4. Ask the agent to regenerate the derived files (see Derived Files) on the
   branch and commit the result.
5. Merge. `main` therefore always carries a consistent index and log digest.

### Merge And Conflict Resynthesis

Markdown conflicts are semantic, not textual. Do not hand-resolve conflict
markers line by line.

1. For each conflicting page, read three versions: merge base, ours, theirs.
2. Ask the agent to synthesize a single page that preserves all claims,
   provenance links, and frontmatter from both sides; contradictions between
   the two sides are recorded explicitly in the page with an uncertainty note
   and, if needed, a new `wiki/questions/` entry.
3. Resynthesized pages require review by another team member before merge.
4. Append a `resynth` log entry naming both source branches and the pages
   touched.

## Local User Identity And Attribution

Use `.llm-wiki-local/user.yaml` as the machine-local identity file for the
current collaborator. This file is operational state, not knowledge content, and
must be ignored by Git.

Identity goals:

- Provide a stable `member_id` for raw notes, branch names, and Git review.
- Avoid syncing personal information into the repository.
- Keep attribution consistent even when Git author names or chat display names
  differ.

Minimum local identity fields:

```yaml
member_id: lowercase-kebab-case
display_name:
git_username:
git_user_name:
git_user_email:
role:
timezone:
attribution_name:
branch_prefix:
```

Privacy rules:

- Do not commit `.llm-wiki-local/user.yaml`.
- Do not overwrite `.llm-wiki-local/user.yaml` during clone, pull, checkout, or
  branch switch. If it is missing, ask the user and create it locally.
- Do not store phone numbers, private addresses, personal IDs, API keys, or
  secrets in this repository.
- If a collaborator wants to be credited differently in reports, use
  `attribution_name` in the local identity file.

Attribution workflow:

1. Each collaborator keeps their own `.llm-wiki-local/user.yaml`.
2. On first project initialization, if the file does not exist, ask for
   `member_id`, `display_name`, `git_username`, `git_user_name`,
   `git_user_email`, `role`, `timezone`, `attribution_name`, and
   `branch_prefix`.
3. Rough raw notes should include an author header when practical:

   ```yaml
   ---
   author_id: member-id
   created: YYYY-MM-DD
   source_type: source-drop | rough-note
   ---
   ```

4. When the LLM synthesizes raw notes into `wiki/`, it should preserve
   authorship in useful fields such as `raw_authors` or `contributors`.
5. For log entries, include who supplied the raw input or requested the
   synthesis when that information is available.
6. For ambiguous authorship, write `author_id: unknown` rather than guessing.

## Directory Contract

- `raw/inbox/`: 분류 전 스테이징 영역. 폴더를 모를 때 여기 넣으면 에이전트가
  내용을 읽고 알맞은 하위 폴더로 자동 이동한다 (Raw File Classification Workflow).
- `raw/sources/`: ingested or staged text, links, transcripts, exports, PDFs,
  and other source documents.
- `raw/meetings/`: rough meeting notes, transcripts, and discussion dumps
  waiting for synthesis. Preserved like any other raw source.
- `raw/assets/`: images, media, screenshots, and binary attachments.
- `raw/<custom>/`: 기존 4개 폴더에 맞지 않는 자료가 있으면 에이전트가 내용 유형에
  맞는 새 하위 폴더를 생성한다. 폴더명은 lowercase kebab-case.
- `wiki/index.md`: navigation map and page catalog. **Generated — do not edit.**
- `wiki/log/`: sharded chronological activity entries, one file per operation
  (see Log Format).
- `wiki/log.md`: generated digest of `wiki/log/`. **Generated — do not edit.**
- `wiki/overview.md`: current high-level synthesis of the vault.
- `wiki/status.md`: current maintenance state, next actions, and the lint
  rotation.
- `wiki/concepts/`: stable ideas, mechanisms, themes, and abstractions.
- `wiki/entities/`: people, organizations, products, projects, and named systems.
- `wiki/sources/`: one note per important source or source bundle.
- `wiki/questions/`: open questions and research tasks.
- `wiki/outputs/`: durable answers, briefs, comparisons, and reports.
- `wiki/meetings/`: cleaned meeting minutes — agenda, discussion, decisions,
  action items, follow-up links.
- `wiki/decisions/`: durable decision records, one file per decision —
  context, options, decision, consequences, owner, review date. Requires
  maintainer review at merge.
- `wiki/tasks/`: team tasks, **one file per task** with owner / status /
  due / priority frontmatter. Sharding per task keeps parallel status
  updates conflict-free.
- `wiki/todo.md`: dashboard of open tasks grouped by owner.
  **Generated — do not edit.**
- `wiki/templates/`: reusable note templates.
- `wiki/_maintenance/`: lint reports, orphan checks, contradiction reviews, and
  other internal maintenance artifacts.
- `scripts/`: local setup, safe Git sync, and skill installation tools only
  (`init-local-user.sh`, `pull-safe.sh`, `sync-user-branch.sh`,
  `install-skills.sh`).
- `.skills/`: the five workflow skills — agent-neutral canonical copy (see
  Agent Skills). Changes require team review. Runtime paths
  (`.codex/skills/`, `.codex/skills/`) are git-ignored links created by
  `scripts/install-skills.sh`.

## Filenames And Links

- Use lowercase kebab-case filenames: `example-topic.md`.
- Use Obsidian wikilinks for internal links: `[[example-topic]]`.
- If display text is needed, use `[[example-topic|Readable Name]]`.
- Avoid ambiguous page names. Prefer `source-karpathy-llm-wiki` over `notes`.
- Keep pages focused. Split broad pages once they contain multiple durable
  concepts.

## Frontmatter

Every maintained wiki page should begin with YAML frontmatter:

```yaml
---
type: concept | entity | source | question | output | maintenance | overview | status | meeting | decision | task
status: seed | active | stable | stale | superseded
created: YYYY-MM-DD
updated: YYYY-MM-DD
summary: 한 줄 요약 (최대 120자, index 생성에 사용)
sources:
  - [[source-note]]
---
```

The `summary` field feeds index generation. Keep it accurate whenever the page
materially changes; if it is missing, the index falls back to the first
sentence of the body. Use only fields that are meaningful.

For execution-layer pages, use these additional fields:

```yaml
# type: task
status: open | doing | done | blocked   # replaces the generic status enum
owner: member-id
due: YYYY-MM-DD
priority: p0 | p1 | p2 | p3

# type: decision
decision_status: proposed | accepted | superseded
owner: member-id
review: YYYY-MM-DD

# type: meeting
attendees:
  - member-id
```

## Derived Files

`wiki/index.md`, `wiki/log.md`, and `wiki/todo.md` are build artifacts of the
vault. They are regenerated **by the agent** (no script, no extra tooling)
during the pre-merge procedure, and any time a fresh local copy is useful:

- `wiki/index.md`: assembled from each page's `summary` frontmatter (fallback:
  first sentence, truncated at 120 chars), grouped by directory in the
  contract order above.
- `wiki/log.md`: shard entries from `wiki/log/` concatenated newest-first.
- `wiki/todo.md`: dashboard of `open` / `doing` / `blocked` tasks from
  `wiki/tasks/` frontmatter, grouped by owner and sorted by due date.
- Each generated file starts with `<!-- generated: do not edit -->`.

Because the index is assembled from page-level summaries, the log from
per-operation shards, and the todo dashboard from per-task files, parallel
work by multiple members never conflicts on these shared files.

## Team Execution Layer

Collaboration artifacts — meetings, decisions, tasks — are knowledge, not a
separate subsystem. They follow the same raw → synthesis model as everything
else, so no dedicated "project" command exists:

- **Meetings**: humans drop rough notes or transcripts into `raw/meetings/`
  and ask for ingest. The Ingest Workflow recognizes the source type and
  produces a cleaned minutes page in `wiki/meetings/`, extracts durable
  decisions into `wiki/decisions/`, and creates task files for action items.
- **Decisions**: one file per decision in `wiki/decisions/` with context,
  options, decision, consequences, owner, and review date. Never bury a
  decision inside meeting minutes — link it from there instead. Decision
  pages require maintainer review at merge.
- **Tasks**: one file per task in `wiki/tasks/` (see the wiki-task skill).
  Reads are queries; writes go through a `task/` branch. Completing a task
  flips its `status` — files are never deleted, preserving history.
- **Dashboard**: `wiki/todo.md` is generated at merge, like the index.
- Keep execution linked to knowledge: tasks and decisions should wikilink the
  concept and source pages they depend on, so "why" is always one hop away.

This layer is what makes handovers cheap: meeting context, decision
rationale, task history, and the knowledge behind them accumulate in one
navigable place.

## Ingest Workflow

Use this when the user adds or points to a new source.

1. `raw/inbox/` 에 미분류 파일이 있으면 Raw File Classification Workflow 를 먼저
   수행해 파일을 적절한 하위 폴더로 이동한 뒤 ingest 를 계속한다.
2. Read `wiki/index.md`, `wiki/status.md`, and the source.
3. Create or update a source note in `wiki/sources/`.
4. Extract durable concepts into `wiki/concepts/`.
5. Extract named people, organizations, products, and projects into
   `wiki/entities/` when they are likely to recur.
6. Update existing related pages instead of duplicating knowledge.
7. Add cross-links in both directions where useful.
8. Update `wiki/overview.md` if the source changes the big picture.
9. If the source is a meeting note or discussion dump: also create the
   minutes page in `wiki/meetings/`, extract decisions into
   `wiki/decisions/` (these require maintainer approval), and create task
   files under `wiki/tasks/` for action items with owners and due dates.
10. Keep the `summary` frontmatter of new or changed pages accurate (the index
    is generated from it — do not edit `wiki/index.md` itself).
11. Update `wiki/status.md` with next actions and unresolved issues.
12. Append a log shard under `wiki/log/` (see Log Format).

One source may legitimately touch many wiki pages. Prefer correctness and
integration over a single isolated summary. For large sources, follow the
chunking rule in Agent Runtime And Cost Policy.

## Query Workflow

Use this when the user asks a question about the knowledge base.

1. Read `wiki/index.md` first.
2. Search the wiki for relevant terms with the agent's built-in text search.
3. Read relevant pages and their cited source notes.
4. Answer with citations to wiki pages and source notes.
5. If the answer is durable or reusable, create an output note in
   `wiki/outputs/` with an accurate `summary` field.
6. Append a `query` log shard when a durable page is created or materially
   updated. Pure Q&A with no page changes needs no log entry and no branch.

## Lint Workflow

This is the wiki health check, run by an agent roughly weekly on a
rotating-member basis, or whenever the user asks for cleanup.

Check for:

- contradictions between pages
- stale claims that newer sources have superseded
- orphan pages with no inbound links or unclear place in the wiki
- important concepts mentioned repeatedly but lacking their own page
- missing cross-references between related pages
- dead wikilinks and broken frontmatter
- source-backed claims that need stronger provenance
- data gaps that could be filled by a new source or web search
- unanswered questions that now have enough evidence
- useful new questions, comparisons, or source targets suggested by the wiki
- inaccurate or missing `summary` fields
- tasks overdue or stalled without a status update
- decisions still `proposed` past their review date

Perform lint as an LLM health-check, not as a purely mechanical format check:

1. Read `wiki/index.md`, `wiki/overview.md`, `wiki/status.md`, and recent
   entries under `wiki/log/`.
2. Search the wiki for recurring terms, wikilinks, unresolved questions,
   TODOs, and uncertainty markers.
3. Read the relevant pages together and compare their claims, dates, sources,
   and cross-links.
4. Write findings to `wiki/_maintenance/wiki-health.md` with severity,
   evidence, affected pages, and recommended fixes.
5. Fix straightforward synthesis and linking issues when the evidence is clear.
6. Create or update `wiki/questions/` entries for gaps that require research.
7. Append a `lint` log shard and open a maintenance PR.

## Log Format

The log is sharded. Each operation writes **one file**:

```text
wiki/log/YYYY-MM/YYYY-MM-DD-<operation>-<topic-slug>.md
```

`<operation>` is one of `ingest`, `query`, `task`, `lint`, `schema`,
`resynth`.
Inside the file, keep the parseable heading format:

```markdown
## [YYYY-MM-DD] ingest | Source Title
```

Each entry must include:

- `Changed:` files or page groups touched
- `Reason:` why the update happened
- `Next:` follow-up work
- `By:` member_id of the requester or author (use `unknown` if ambiguous)

Because each operation owns its own file, parallel work never conflicts on the
log. `wiki/log.md` is the generated newest-first digest of all shards.

## Raw File Classification Workflow

원천 자료를 `raw/` 에 추가할 때 적절한 하위 폴더가 불분명한 경우 이 워크플로를 사용한다.
파일을 `raw/inbox/` 에 넣으면 에이전트가 내용을 읽고 분류해 자동으로 이동시킨다.

### 분류 기준

에이전트는 확장자 → 파일명 → 파일 내용 순서로 참조해 분류한다:

| 유형 | 판단 기준 | 이동 경로 |
|------|-----------|-----------|
| 회의록 / 토론 덤프 | 회의, 안건, 참석자, 액션 아이템 등 키워드 또는 대화체 구조 | `raw/meetings/` |
| 연구 자료 / 기사 | 논문, 기술 문서, 웹 기사, 링크, 슬라이드, 정리 노트 | `raw/sources/` |
| 이미지 / 미디어 / 바이너리 | `.png .jpg .gif .mp4 .mp3 .wav .pptx .xlsx` 등 | `raw/assets/` |
| 신규 유형 | 위 세 범주에 명확히 맞지 않는 자료 | `raw/<유형>/` 새 폴더 생성 |

### 절차

1. `raw/inbox/` 에서 미분류 파일 목록을 확인한다.
2. 각 파일을 읽는다:
   - 텍스트 파일: 내용 전체를 읽고 유형을 판단한다.
   - 바이너리 / 미디어: 확장자와 파일명만으로 판단한다.
3. 분류 기준에 따라 이동 경로를 결정한다.
   - 기존 디렉토리가 적합하면 해당 위치로 이동한다.
   - 맞는 디렉토리가 없으면 `raw/<유형>/` 을 새로 만들고 이동한다.
     새 폴더명은 내용 유형을 명확히 나타내는 lowercase kebab-case 로 짓는다.
4. 분류가 불확실한 파일은 이동 전에 사용자에게 확인한다.
5. 분류 결과를 보고한다 — 이동 경로, 신규 생성된 폴더 목록 포함.
6. 분류 완료 후 즉시 ingest 를 진행할지 사용자에게 묻는다.

### 호출 방법

- 명시적 요청: "inbox 정리해줘", "raw 파일 분류해줘", "classify 해줘"
- Ingest Workflow Pre-flight 중 `raw/inbox/` 에 파일이 있으면 자동으로 이 절차를
  먼저 수행한 뒤 ingest 로 진행한다.
- 사용자가 경로 지정 없이 파일만 전달하면 `raw/inbox/` 에 저장 후 이 절차를 실행한다.

### Hard constraints

- 파일 내용은 읽기만 하고 수정하지 않는다.
- `raw/` 외부로 파일을 이동하지 않는다.
- 분류 후 원본 파일은 반드시 하나의 경로에만 존재해야 한다 (복사 후 삭제 금지, 이동만).

## Source Handling

- Preserve raw files and original URLs.
- Avoid long verbatim quotes from copyrighted sources.
- Summarize in your own words.
- For web sources, record observed date and URL.
- For images or PDFs, keep attachments in `raw/assets/` and summarize what was
  actually inspected.

## Human Collaboration

The human curates sources, asks questions, and steers emphasis. The LLM handles
summarizing, linking, bookkeeping, contradiction checks, and log updates.
Index assembly is a regeneration task performed at merge time, not a page the
LLM curates by hand.

Each team member runs their own subscription CLI agent on their own machine
with their own `.llm-wiki-local/user.yaml`. The shared state is the Git
repository and nothing else.