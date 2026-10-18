---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-10-02
created: 2026-09-25
updated: 2026-09-25
summary: speechlm QC 통과 19개 DB를 편입 범위로 택했고, 57개 청취 표본 통과 후 라벨링 입력을 승인했다. 교사 관문 전 학습은 보류한다.
sources:
  - '[[output-speechlm-semcommit-approval-audit-20260925]]'
  - '[[output-speechlm-arrow-asr-review]]'
  - '[[task-semcommit-main-labeling]]'
  - '[[source-speechlm-human-review-57-20260925]]'
---

# 결정: speechlm QC 통과 19개 DB 전체를 편입 후보로

## 맥락과 선택지

부분 정렬을 종료한 speechlm 자료의 CPU QC와 보호 평가셋 감사를 마쳤다. `QC_PASS_CANDIDATE`는 19개 원천, 8,807,815건, 약 12,786.2시간이다. 대안은 P0 7개 원천에 시간 상한을 거는 것이었다. 이 대안의 사전 제안 뷰는 477,380건, 961.475시간이며 삭제하지 않고 비교용으로 남긴다. 수치와 감사 범위는 [[output-speechlm-semcommit-approval-audit-20260925]]에 기록했다.

## 결정

2026-09-25 사용자 선택에 따라 **19개 DB의 QC 통과분 전부**를 이번 후보 범위로 한다. 원천별 또는 길이별 시간 상한은 적용하지 않는다. 8초 미만 발화도 제외하지 않고 별도 풀에 두며, 이후 학습에서 짧은 발화의 step 비중을 **10%**로 계획한다. 이 비중은 데이터·교사 라벨링 물량의 10% 상한이 아니다.

이 결정은 처음에 **후보 범위 선택**이었다. 이후 사용자가 [[source-speechlm-human-review-57-20260925|57개 1차 청취 표본을 모두 통과]]로 판정하여, 별도 뷰에서 QC 통과 행을 `labeling_eligible=true`로 승인했다. 0길이 정렬 보류 행은 제외하고 `training_eligible=false`를 유지한다. 보호 평가셋 대조의 한계 기록, 교사 골드 관문, 코퍼스별 구두점/단어 타깃 점검은 여전히 필요하다. [[task-semcommit-main-labeling]]의 GPU 라벨링을 이 결정만으로 시작하지 않는다.

## 결과와 검토 조건

- 기존 [[output-speechlm-arrow-asr-review]]의 P0 우선순위와 `000009/000017/000021/000026` 보류 권고는 **이번 후보 범위에 한해서** 대체된다. QC 통과가 이 도메인들의 전사·정렬 품질을 보증하지는 않는다. 정확 중복·평가 보존에 따른 앞선 QC 탈락분은 되살리지 않는다.
- 전체 후보를 한 번에 교사 라벨링하면 처리 비용과 출처 편중이 커진다. 사람 검수 후 코퍼스·언어·8 kHz/16 kHz 구성과 처리량을 확인하고 학습 샘플링을 별도로 결정한다.
- 2026-09-25에 1차 청취 57개 통과와 라벨링 입력 승인 뷰 생성을 확인했다. 2026-10-02에는 교사·학습 관문 진행 여부를 검토한다. 결정 페이지는 병합 전 다른 팀원의 리뷰를 받는다.
