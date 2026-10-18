---
type: source
status: active
created: 2026-09-25
updated: 2026-09-25
summary: 사용자가 speechlm 19개 DB의 1차 청취 표본 57개를 모두 통과로 판정했다. 미청취 확대 표본과 QC 보류 행은 포함하지 않는다.
sources:
  - '[[output-speechlm-semcommit-approval-audit-20260925]]'
---

# speechlm 57개 1차 청취 표본의 사용자 판정

원문 기록: `raw/sources/experiments/2026-09-25-speechlm-human-review-57.json`. 사용자 답변은 “57개 모두 통과”이다. 앞선 요청은 [1차 검토 페이지](</Users/taesookim/Downloads/speechlm-semcommit-review-20260925/review-bundle-smoke-20260925-v1/index.html>)의 음성을 듣고 승인 판정을 알려 달라는 것이었다.

검토 묶음은 19개 DB마다 8초 미만·8초 이상·0길이 정렬 표본을 각 1개 포함한다. 로컬과 mxc의 57개 오디오 파일 목록·내용 SHA-256, 검토 항목, HTML, 묶음 요약이 동일함을 확인했다. 묶음 요약 SHA-256은 `3e7aa1d0d70b9a8a7a0d490d75186280053bcde5e07ad108bbcd7ff2480d6ca1`, 오디오 파일 해시 목록의 SHA-256은 `6b55e1bc16301f6ab9158cbf77d0afae49a8f4d09340ccc694909bab21240b3c`이다.

이 판정의 범위는 **57개 1차 표본의 사람 검토 통과**이다. 확대 묶음 285개는 사람 검토됐다고 보지 않으며, `REVIEW_ZERO_DURATION` 모집단을 자동 통과시키지 않는다. 통과 표본만으로 880만 행 각각의 전사 정확도를 증명하지는 않는다.

관련 범위와 후속 관문: [[decision-speechlm-all19-qc-candidate-scope]], [[output-speechlm-semcommit-approval-audit-20260925]], [[task-semcommit-main-labeling]].
