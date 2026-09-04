#!/usr/bin/env bash
# 로컬 신원을 보존하는 안전한 pull.
# 다음 경우 pull 을 거부한다 (AGENTS.md: Git Collaboration Policy):
#   1) .llm-wiki-local/ 이 로컬에서 Git 에 추적되고 있을 때
#   2) 업스트림 트리에 .llm-wiki-local/ 이 존재할 때
#   3) 로컬 신원이 초기화되지 않았을 때
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

remote="${1:-origin}"
branch="${2:-$(git rev-parse --abbrev-ref HEAD)}"

fail() { echo "pull 중단: $*" >&2; exit 1; }

[[ -f .llm-wiki-local/user.yaml ]] || \
  fail "로컬 신원이 없습니다. 먼저 scripts/init-local-user.sh 를 실행하세요."

if git ls-files --error-unmatch .llm-wiki-local >/dev/null 2>&1; then
  fail ".llm-wiki-local/ 이 Git 에 추적되고 있습니다. 'git rm -r --cached .llm-wiki-local' 로 해제하세요."
fi

git fetch "$remote" "$branch"

if git ls-tree -r --name-only "$remote/$branch" | grep -q '^\.llm-wiki-local/'; then
  fail "업스트림 트리에 .llm-wiki-local/ 이 포함되어 있습니다. 원격에서 제거한 뒤 다시 시도하세요."
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "경고: 작업 트리에 커밋되지 않은 변경이 있습니다." >&2
  git status --short >&2
fi

echo "==> git merge --ff-only $remote/$branch"
git merge --ff-only "$remote/$branch" || {
  echo "fast-forward 불가. 브랜치를 rebase 하거나 병합 절차(Pre-merge procedure)를 따르세요." >&2
  exit 1
}
echo "완료. 로컬 신원 보존됨: .llm-wiki-local/user.yaml"
