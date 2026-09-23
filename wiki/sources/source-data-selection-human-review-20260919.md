---
type: source
status: active
created: 2026-09-19
updated: 2026-09-19
summary: 사용자의 NIKL 48 kHz 진단 16개 검수: 전사 15 통과·1 실패, 경계·화자 모두 통과. 의미 보정과 발음 전사를 구분한다.
raw_path: raw/sources/experiments/2026-09-19-data-selection-review/human-review-20260919-130219.json
raw_authors:
  - tskim
sources:
  - "[[output-data-selection-review-v02]]"
---

# NIKL 48 kHz 진단 표본: 사람 청취 검수

## 출처와 검증

사용자가 `/Users/taesookim/Downloads/human-review.json`으로 전달한 검수 내보내기다.
Downloads 원본은 그대로 두고, [원형 사본](../../raw/sources/experiments/2026-09-19-data-selection-review/human-review-20260919-130219.json)을 보존했다.
검수 시각은 `2026-09-19T04:02:19.073Z`(KST 13:02:19)다.

원본 SHA256: `4fa57def437c7c09413b7f647eef139b824325bd56a80ac5f026b8ff8135bbea`.

16개 key·source가 모두 [48 kHz 진단 manifest](../../raw/sources/experiments/2026-09-19-data-selection-review/rate-probe.jsonl)와 일치한다.
886개 전체 표본 검수가 아니라, NIKL 2022의 `SDRW2200000598`, `SDRW2200000635`
두 세션에서 고른 진단 표본 16개의 검수다. 파일의 `training_eligible`은 `false`다.
내보내기 자체는 파형 hash를 포함하지 않으므로 key로 연결된 진단 manifest의 원음·파형
hash와 진단 조건을 근거로 사용한다. 독립적인 검수자 서명 또는 청취 행동 로그는 아니다.

## 원래 판단을 그대로 집계

| 검수 축 | pass | fail | uncertain | unreviewed |
|---|---:|---:|---:|---:|
| 전사 | 15 | 1 | 0 | 0 |
| 경계/잘림 | 16 | 0 | 0 | 0 |
| 채널/화자 | 16 | 0 | 0 | 0 |
| 턴 | 5 | 0 | 2 | 9 |

## 메모가 있는 표본과 해석

| 발화 ID | 현재 타깃 / Qwen 차이 | 사용자 판단 | 반영 방식 |
|---|---|---|---|
| SDRW2200000635.1.1.35 | `긍까 연기를` / `그러니까 용기를` | text=fail. 발음은 현재 타깃이 정확하나 의미는 Qwen이 알맞다는 메모 | 실패 라벨 보존. 발음 충실도와 의미 보정 기준이 달라 추가 판정 필요. 자동 교정·폐기 금지 |
| SDRW2200000598.1.1.336 | `떠올리면서…끄트마리에 여게` / `또 올리면서…끝말이에요 이게` | 현재 타깃이 맞고 Qwen이 틀림 | 원전사 유지; 교사의 유창한 출력이 정답이라는 가정 금지 |
| SDRW2200000635.1.1.68 | `손석구` / `손석호` | 현재 타깃이 맞음 | 고유명사 교사 치환 오류 사례로 보존 |
| SDRW2200000598.1.1.152 | `쫌` / `좀` | text=pass. 발음은 현재 타깃, 문법은 Qwen이 맞다는 메모 | 원전사 유지; 의미·문법 정리는 별도 display 목적과 구분 |
| SDRW2200000598.1.1.116 | `우리가 뭐…거뿐이지` / `우리가 막…그뿐이지` | 현재 타깃이 맞음 | 교사와 다르다는 이유만으로 원전사를 교체하지 않음 |

이 표는 사용자의 청취 판단을 요약한 것이며, 에이전트가 음성을 직접 듣고 판정했다는 뜻이 아니다.
원문의 철자·메모·축별 판정은 수정하지 않았다.

## 유효 범위와 다음 조치

- 진단 CER 개선과 함께 **검수된 16개의 48 kHz 해석을 지지하는 청취 근거**가 추가됐다.
  다만 사용자가 샘플레이트 자체를 별도 필드로 승인한 것은 아니다. 해당 두 세션의 나머지
  파일 및 NIKL 다른 세션까지 일괄 변경하는 근거로 확대하지 않는다.
- 경계 pass는 청취한 crop의 잘림 점검이다. 단어별 정렬 오차, 원대화상의 시작 offset,
  80 ms streaming 지연 정확도를 승인한 것으로 해석하지 않는다.
- 화자 pass는 해당 crop의 채널/화자 청취 판단이다. 원대화 전체 speaker identity 매핑이나
  화자 분리 정확도까지 승인한 것은 아니다.
- turn pass 5개는 원형대로 보존하지만, 내보내기에 종료 시각·주변 문맥이 없다.
  시각이 붙은 `<EOT>` 또는 VAP/hazard 감독으로 자동 변환하지 않는다.
- 다음은 두 세션의 파일별 포맷 검사와 명시적 sample-rate 예외표다. 전사 실패 1개는
  lexical/display 기준을 구분해 재판정하고, 남은 DB의 시간축·채널 검수도 계속 필요하다.

검수 당시 기록: [[output-data-selection-review-v02]].
후속 실행: [[output-data-selection-repair-v03]] — 두 세션 681개 파일 검사 후 기존 후보 625개만
새 WAV·포맷 예외표로 복구했다. 원전사와 사람 검수 쟁점 1개는 유지했다.
