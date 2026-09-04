#!/usr/bin/env python3
"""파생 파일 재생성: wiki/log.md(샤드 다이제스트), wiki/todo.md(열린 태스크 대시보드).
index.md 는 설명 문구가 큐레이션되므로 여기서 다루지 않는다(수동 갱신)."""
import re, glob, datetime, pathlib
R = pathlib.Path(__file__).resolve().parent.parent / "wiki"
today = datetime.date.today().isoformat()
# --- log.md
shards = sorted(glob.glob(str(R / "log" / "*" / "*.md")), reverse=True)
body = "\n".join(pathlib.Path(p).read_text().strip() + "\n" for p in shards)
(R / "log.md").write_text(f"<!-- generated: do not edit -->\n# 활동 로그\n\n마지막 생성: {today}\n\n"
    "`wiki/log/` 의 샤드를 최신순으로 이어붙인 다이제스트다. 직접 편집하지 않는다.\n\n" + body)
# --- todo.md
def fm(p):
    t = pathlib.Path(p).read_text(); m = re.match(r"---\n(.*?)\n---", t, re.S); d = {}
    for l in (m.group(1) if m else "").splitlines():
        if re.match(r"^[a-z_]+:", l): k, v = l.split(":", 1); d[k] = v.strip()
    return d
rows = {}
for p in sorted(glob.glob(str(R / "tasks" / "*.md"))):
    d = fm(p)
    if d.get("status") not in ("open", "doing", "blocked"): continue
    rows.setdefault(d.get("owner", "unassigned"), []).append((d.get("due", "9999"), d))
    d["name"] = pathlib.Path(p).stem
out = ["<!-- generated: do not edit -->", "# TODO", "", f"마지막 생성: {today}", "",
       "`wiki/tasks/` 의 `open` / `doing` / `blocked` 태스크를 owner 별로 모은 대시보드.",
       "직접 편집하지 않는다. 단계별 실행 체크리스트는 저장소 루트 `TODO.md` 를 본다.", ""]
for owner, items in sorted(rows.items()):
    out += [f"## {owner}", "", "| 상태 | 우선순위 | 마감 | 태스크 |", "|------|----------|------|--------|"]
    for due, d in sorted(items, key=lambda x: (x[1].get("priority", "p9"), x[0])):
        st = d["status"]; st = f"**{st}**" if st != "open" else st
        pr = d.get("priority", ""); pr = f"**{pr}**" if pr == "p0" else pr
        out.append(f"| {st} | {pr} | {d.get('due','')} | [[{d['name']}]] — {d.get('summary','')} |")
    out.append("")
(R / "todo.md").write_text("\n".join(out))
print("regen ok:", len(shards), "shards,", sum(len(v) for v in rows.values()), "open tasks")
