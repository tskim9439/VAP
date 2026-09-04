---
name: wiki-lint
description: 위키 건강 점검. 모순, 낡은 주장, 고아 페이지, 끊어진 위키링크, 누락된 summary, 지연된 태스크, 리뷰 기한이 지난 결정을 찾아 wiki/_maintenance/wiki-health.md 에 기록하고 명확한 문제는 고친다. 주 1회 로테이션 또는 사용자가 "린트", "정리", "위키 점검"을 요청할 때 사용한다.
---

# Lint Workflow

`AGENTS.md` 의 Lint Workflow 를 실행하는 스킬이다. **정본은 `AGENTS.md`** 이다.

이것은 **LLM 건강 점검** 이지 기계적 포맷 검사가 아니다. 페이지들을 함께 읽고
주장·날짜·출처·교차링크를 비교하는 것이 핵심이다.

## 절차

1. 브랜치: `scripts/sync-user-branch.sh maintenance wiki-health-YYYY-MM-DD`
2. `wiki/index.md`, `wiki/overview.md`, `wiki/status.md`, 그리고 `wiki/log/` 의
   최근 항목을 읽는다.
3. 반복되는 용어, 위키링크, 미해결 질문, TODO, 불확실성 표시를 검색한다.
4. 관련 페이지들을 **함께** 읽고 주장·날짜·출처·교차링크를 비교한다.
5. `wiki/_maintenance/wiki-health.md` 에 심각도, 근거, 영향 페이지, 권장 수정을 기록한다.
6. 근거가 명확한 합성·링크 문제는 직접 고친다.
7. 조사가 필요한 공백은 `wiki/questions/` 항목으로 만들거나 갱신한다.
8. `lint` 로그 샤드를 추가하고 유지보수 PR 을 연다.

## 점검 항목

- 페이지 간 **모순**
- 새 출처가 대체한 **낡은 주장**
- 인바운드 링크가 없거나 위치가 불분명한 **고아 페이지**
- 반복 언급되지만 자기 페이지가 없는 **중요 개념**
- 관련 페이지 사이 **누락된 교차 참조**
- **끊어진 위키링크** 와 깨진 frontmatter
- 출처 보강이 필요한 주장 (**provenance**)
- 새 원천이나 웹 검색으로 메울 수 있는 **데이터 공백**
- 이제 근거가 충분해진 **미답 질문**
- 위키가 시사하는 유용한 **새 질문·비교·출처 후보**
- 부정확하거나 누락된 **`summary` 필드**
- 기한이 지났거나 상태 갱신 없이 멈춘 **태스크**
- 리뷰 기한이 지났는데 아직 `proposed` 인 **결정**

## 유용한 사전 스캔

기계적 스캔은 시작점일 뿐이며, 반드시 읽기로 검증한다.

```bash
# 끊어진 위키링크 후보
grep -rhoE '\[\[[^]|]+' wiki --include='*.md' | sed 's/\[\[//' | sort -u \
  | while read -r p; do [ -f "$(find wiki -name "$p.md" | head -1)" ] || echo "DEAD? $p"; done

# summary 누락 페이지
grep -rLE '^summary:' wiki --include='*.md'

# 고아 후보 (다른 페이지에서 링크되지 않은 페이지)
for f in $(find wiki -name '*.md' -not -path 'wiki/log/*'); do
  s=$(basename "$f" .md)
  grep -rq "\[\[$s\([|]\|\]\)" wiki --include='*.md' || echo "ORPHAN? $f"
done
```

## 건강 리포트 형식

`wiki/_maintenance/wiki-health.md`:

```markdown
---
type: maintenance
status: active
created: YYYY-MM-DD
updated: YYYY-MM-DD
summary: YYYY-MM-DD 위키 건강 점검 결과 — 심각도별 발견 사항과 권장 수정
---

# 위키 건강 리포트 (YYYY-MM-DD)

담당: member-id

## High

### 1. 제목
- 근거: 무엇을 어디서 확인했는가
- 영향 페이지: [[page-a]], [[page-b]]
- 권장 수정: 구체적 조치
- 처리: 수정함 / 질문 생성 / 보류(사유)

## Medium
## Low
```

## 로그 샤드

```markdown
## [YYYY-MM-DD] lint | 주간 건강 점검

- Changed: wiki/_maintenance/wiki-health.md 및 수정한 페이지
- Reason: 로테이션 주간 점검 (또는 사용자 요청)
- Next: 남은 권장 수정과 담당
- By: member-id
```

## 로테이션

담당자는 `wiki/status.md` 에 기록된다. 점검 후 다음 담당자로 갱신한다.
