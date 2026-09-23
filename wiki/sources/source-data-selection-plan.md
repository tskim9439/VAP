---
type: source
status: active
created: 2026-09-18
updated: 2026-09-18
summary: DataSelectionPlan 원문을 보존하고 다중 ASR 합의 기반 선별의 장점·위험·실행 범위를 정리한다.
sources:
  - "raw/inbox/DataSelectionPlan.md"
raw_authors:
  - unknown
---

# 고품질 데이터 선별 원안

사용자가 분석·구체화·DB별 실행을 요청한 [원문](../../raw/inbox/DataSelectionPlan.md).
지정된 원본 경로와 내용을 그대로 보존했다. 문서 안 대화의 개별 작성자는 확정하지 않는다.

## 원안에서 채택한 내용

- DB adapter → 공통 manifest → Qwen/Whisper 전사 → GT 포함 3자 비교 → 음향·정렬 QC → 규칙 판정 → 검수.
- 원전사와 교사 전사를 별도로 보존하고, 학습용 TN과 비교용 정규화를 구분한다.
- 교사 두 개가 원전사와 다르게 동의해도 자동으로 원전사를 덮어쓰지 않는다.
- 문장별 판정 이유·버전·출처를 보존하며 사람 표본 검수로 임계값을 보정한다.

## 구체화가 필요한 내용

원문의 통과 비율과 5%/15% 기준은 실측 결과가 아니라 제안값이다. forced alignment는
주어진 글을 음성에 배치하는 도구이지 전사의 진위를 독립적으로 판정하는 심판이 아니다.
교사 합의와 높은 정렬 성공률만으로 GOLD를 선언하지 않는다.

Phase 2에서는 overlap·짧은 맞장구·긴 무음이 필요한 학습 상황이다. 이를 일괄 제거하는
단일 음향 점수 대신 전사·시각·화자·턴 품질을 분리해야 한다. 발화 조각을 지워 대화
시간축을 압축하거나 미확인 구간을 무음 정답으로 바꾸면 안 된다.

실행 설계와 관측 결과: [[output-data-selection-execution-plan]].
기존 정규화 계약: [[output-asr-tn-v1-spec]].
단일 NIKL 표본의 Whisper 비교는 [[output-phase2-d1b-sample-inference]]에 있으며,
그 표본만으로 v2/v3의 전체 한국어 성능 순위를 확정할 수 없다.
