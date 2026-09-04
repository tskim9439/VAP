#!/usr/bin/env bash
# 사용자 접두사 브랜치를 생성/체크아웃하고 origin 에 푸시한다.
#
#   scripts/sync-user-branch.sh <work-type> <topic> [--push] [--allow-main]
#
# work-type: ingest | query | task | maintenance
# 브랜치명: <branch_prefix>/<work-type>/<topic>
# main 직접 푸시는 --allow-main 없이는 거부한다 (AGENTS.md: Git Collaboration Policy).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

allow_main=0
do_push=0
args=()
for a in "$@"; do
  case "$a" in
    --allow-main) allow_main=1 ;;
    --push) do_push=1 ;;
    *) args+=("$a") ;;
  esac
done

usage() {
  echo "사용법: scripts/sync-user-branch.sh <ingest|query|task|maintenance> <topic> [--push] [--allow-main]" >&2
  exit 2
}

user_file=".llm-wiki-local/user.yaml"
[[ -f "$user_file" ]] || { echo "로컬 신원이 없습니다. scripts/init-local-user.sh 를 먼저 실행하세요." >&2; exit 1; }
branch_prefix="$(sed -n 's/^branch_prefix:[[:space:]]*//p' "$user_file" | head -1)"
[[ -n "$branch_prefix" ]] || { echo "$user_file 에 branch_prefix 가 없습니다." >&2; exit 1; }

if [[ $allow_main -eq 1 && ${#args[@]} -eq 0 ]]; then
  current="$(git rev-parse --abbrev-ref HEAD)"
  [[ "$current" == "main" ]] || { echo "--allow-main 은 main 브랜치에서만 사용합니다 (현재: $current)." >&2; exit 1; }
  echo "==> git push origin main (명시적 승인)"
  git push origin main
  exit 0
fi

[[ ${#args[@]} -eq 2 ]] || usage
work_type="${args[0]}"
topic="${args[1]}"

case "$work_type" in
  ingest|query|task|maintenance) ;;
  *) echo "알 수 없는 work-type: $work_type" >&2; usage ;;
esac

slug="$(printf '%s' "$topic" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//;s/-$//')"
[[ -n "$slug" ]] || { echo "topic 슬러그가 비었습니다." >&2; exit 1; }
branch="$branch_prefix/$work_type/$slug"

if [[ "$branch" == "main" || "$branch" == */main ]]; then
  echo "main 브랜치는 이 스크립트로 다룰 수 없습니다." >&2; exit 1
fi

if git show-ref --verify --quiet "refs/heads/$branch"; then
  echo "==> git checkout $branch"
  git checkout "$branch"
else
  echo "==> git checkout -b $branch"
  git checkout -b "$branch"
fi

if [[ $do_push -eq 1 ]]; then
  echo "==> git push -u origin $branch"
  git push -u origin "$branch"
else
  echo "브랜치 준비 완료: $branch"
  echo "푸시하려면: scripts/sync-user-branch.sh $work_type $topic --push"
fi
