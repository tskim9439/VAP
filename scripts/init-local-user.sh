#!/usr/bin/env bash
# .llm-wiki-local/user.yaml 을 대화식으로 생성한다.
# 이 파일은 머신 로컬 상태이며 Git 에 커밋되지 않는다 (AGENTS.md: Local User Identity).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_dir="$repo_root/.llm-wiki-local"
user_file="$local_dir/user.yaml"

if [[ -f "$user_file" ]]; then
  echo "이미 존재합니다: $user_file"
  echo "덮어쓰지 않습니다. 수정하려면 파일을 직접 편집하세요."
  exit 0
fi

ask() {
  local prompt="$1" default="${2:-}" reply
  if [[ -n "$default" ]]; then
    read -r -p "$prompt [$default]: " reply
    printf '%s' "${reply:-$default}"
  else
    while :; do
      read -r -p "$prompt: " reply
      [[ -n "$reply" ]] && { printf '%s' "$reply"; return; }
      echo "  (필수 항목입니다)" >&2
    done
  fi
}

git_name_default="$(git -C "$repo_root" config user.name || true)"
git_email_default="$(git -C "$repo_root" config user.email || true)"

echo "로컬 사용자 신원을 설정합니다. (Git 에 커밋되지 않습니다)"
member_id="$(ask 'member_id (lowercase-kebab-case)')"
display_name="$(ask 'display_name' "${git_name_default}")"
git_username="$(ask 'git_username (GitHub 등 호스트 계정)')"
git_user_name="$(ask 'git_user_name' "${git_name_default}")"
git_user_email="$(ask 'git_user_email' "${git_email_default}")"
role="$(ask 'role' 'contributor')"
timezone="$(ask 'timezone' 'Asia/Seoul')"
attribution_name="$(ask 'attribution_name (보고서 표기명)' "$display_name")"
branch_prefix="$(ask 'branch_prefix' "$member_id")"

mkdir -p "$local_dir"
cat > "$user_file" <<YAML
member_id: $member_id
display_name: $display_name
git_username: $git_username
git_user_name: $git_user_name
git_user_email: $git_user_email
role: $role
timezone: $timezone
attribution_name: $attribution_name
branch_prefix: $branch_prefix
YAML

cat > "$local_dir/README.md" <<'MD'
이 디렉토리는 머신 로컬 상태입니다. Git 에 커밋하지 마세요.
`.gitignore` 에 등록되어 있으며, pull-safe.sh 가 추적 여부를 검사합니다.
MD

echo
echo "생성 완료: $user_file"
echo "브랜치 접두사: $branch_prefix"
