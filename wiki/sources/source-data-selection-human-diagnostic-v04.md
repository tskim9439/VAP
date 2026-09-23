---
type: source
status: active
created: 2026-09-20
updated: 2026-09-20
summary: 후속 진단 8개 수신: 청취문 4개 입력, 상태는 전부 unreviewed. AMI great와 VoxPopuli in 생략을 확인 전까지 정정 후보로 보존.
raw_path: raw/sources/experiments/2026-09-20-data-selection-diagnostic/human-diagnostic-20260920-003716.json
raw_authors:
  - tskim
sources:
  - "[[output-data-selection-diagnostic-v04]]"
  - "[[source-data-selection-human-review-v03]]"
---

# 후속 진단 v0.4: 사람 청취문 수신

## 원형 보존과 대응 검증

사용자가 전달한 `/Users/taesookim/Downloads/human-diagnostic-v04.json`은 그대로 유지하고
[원형 사본](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/human-diagnostic-20260920-003716.json)을 보존했다.
기록 시각은 `2026-09-19T15:37:16.202Z`, KST **2026-09-20 00:37:16**이다.

- 원본·사본 SHA256: `d35b7e8c3ead10682564f51afe7aee40c404f95901f61a0f16e4d01582fd7a74`.
- `purpose=diagnostic-followup-v04`, `training_eligible=false`.
- 8개 key가 [진단 카드](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/cards.json)와 정확히 일치하며 중복·누락 없음.
- 카드 SHA256: `2639c25fc11ef39fd09524b27fdbdb9589831d7550bd5070ec74a99be8f309f7`.
- 연결된 13개 오디오의 파형 hash와 기존 검수 예외표 지문을 다시 확인했다.
- `resolution`은 8개 모두 **unreviewed**, `heard_text`는 4개 비어 있지 않음/4개 빈 문자열,
  메모는 전부 빈 문자열이다. 입력 내용이 있다는 사실과 최종 해결 상태를 구분한다.

## 입력된 청취문 4개 — 원형 그대로

| 사례 | 현재 타깃 | 사용자가 입력한 `heard_text` | 해석과 남은 확인 |
|---|---|---|---|
| 방송 `vr_m_001_370_049_0348` | `네 네 네` | `네 네` | 앞선 사람 검수와 두 번 반복이라는 판단이 일치. 국소 전사 정정 후보이며 TN 오류라는 뜻은 아님 |
| AMI `ES2010d / A_00018` | `right` | `great` | 원주석과 세 교사의 `Right`에 반하는 새 청취 근거. 해당 Headset-0 crop인지 확인 필요 |
| VoxPopuli `20110511-0900-PLENARY-2-en_20110511-10:15:44_19` | `i understand that too` | `And i understand that too` | 앞의 `and` 추가를 다시 지지. 대소문자를 포함한 입력 원형을 유지 |
| VoxPopuli `20090423-0900-PLENARY-3-en_20090423-09:23:18_9` | `let us keep this in perspective` | `let's keep this perspective` | 축약형 변경 외에 **`in` 삭제**가 추가됨. 의도적 생략인지 입력 누락인지 미확정 |

현재 근거는 사용자의 청취문이며, 에이전트가 음성을 직접 듣고 독립 판정했다는 뜻이 아니다.
자연스러운 영어 문법이나 교사 합의를 이유로 `great`를 `right`로, 또는 생략된 `in`을
임의 복원하지 않는다. 반대로 문자열 입력만으로 기존 사람 fail을 자동 해제하지도 않는다.

## 빈 청취문 4개

`getting so much`, `attending ministers…`, ECR 문장, fisheries 긴 문장은 새 청취문·메모가 없다.
**빈 값은 무음·빈 전사 정답·문제 해결 또는 원타깃 동의가 아니다.** 기존 검수와 보류 상태를 유지한다.
앞선 메모에 있던 정정 의견도 이번 빈 값으로 취소된 것으로 해석하지 않는다.

## 필요한 확인은 두 가지

후속 사용자 지시로 두 확인을 반복 요청하지 않는다. 기존 모호성은 보류 근거로 남기되, [[output-data-selection-automatic-framework]]의 자동 전량 선별을 막지 않는다. 아래는 당시 남은 쟁점의 기록이다.

1. **AMI `great`**: Headset-0의 원 crop 또는 증폭 crop에서 들은 내용인가?
   카드에는 A/B/C/D의 앞뒤 문맥도 있었으므로 다른 화자·다른 시각의 단어인지 구분해야 한다.
   내보내기 형식에는 사용자가 어느 재생 파일을 근거로 했는지 기록되지 않는다.
2. **VoxPopuli `let's keep this perspective`**: `in`이 들리지 않아 의도적으로 뺀 것인가,
   아니면 입력 중 빠진 것인가? 기존 타깃과 세 교사에는 모두 `in`이 있다.

모든 상태가 unreviewed인 것은 UI 기본값을 그대로 둔 결과일 수도 있지만 확정할 수 없다.
사용자가 대화에서 정확한 청취문을 확정하면 추가 증거로 기록할 수 있으므로,
8개를 모두 다시 듣거나 파일을 전부 다시 작성할 필요는 없다.

## 이번 반영 범위

원형 사본·대응 검증·후속 source note·보고서/상태 링크를 반영했다.
4개 입력은 **정정 후보**로 보존했으며, 최종 승인 0개다.
학습 타깃·TN·원음·기존 검수 예외표는 변경하지 않았다. 원래 사람 실패 11개 보류를 유지하고,
추론·학습 GPU 작업이나 전량 screening을 새로 시작하지 않았다.

확인 이후에도 전사 확정과 경계·화자·턴 자격은 축별로 판단한다. 필요한 행만 새 override로
연결하고, 텍스트 수정 시 정렬·시퀀스를 재생성해야 한다. 기존 검수 기록은 덮어쓰지 않는다.
