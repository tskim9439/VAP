#!/usr/bin/env bash
# .skills/ (에이전트 중립 정본) 를 각 런타임 스킬 경로에 링크한다.
# 런타임 경로는 .gitignore 로 무시된다 (AGENTS.md: Directory Contract).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

[[ -d .skills ]] || { echo ".skills/ 가 없습니다." >&2; exit 1; }

targets=(".claude/skills" ".codex/skills")

for target in "${targets[@]}"; do
  mkdir -p "$target"
  for skill_dir in .skills/*/; do
    name="$(basename "$skill_dir")"
    link="$target/$name"
    rel="$(printf '%s' "$target" | sed 's:[^/]*:..:g')/.skills/$name"
    if [[ -L "$link" ]]; then
      rm "$link"
    elif [[ -e "$link" ]]; then
      echo "건너뜀 (심볼릭 링크가 아님): $link" >&2
      continue
    fi
    ln -s "$rel" "$link"
    echo "링크됨: $link -> $rel"
  done
done

echo
echo "설치 완료. 등록된 스킬:"
ls -1 .skills
