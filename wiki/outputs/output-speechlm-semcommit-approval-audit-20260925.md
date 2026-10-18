---
type: output
status: active
created: 2026-09-25
updated: 2026-09-25
summary: speechlm 19개 DB의 QC 통과 880.8만 행은 57개 청취 표본 통과로 라벨링 입력 승인됐으나, 교사 관문 전 학습에는 사용하지 않는다.
sources:
  - '[[output-speechlm-arrow-asr-review]]'
  - '[[output-speechlm-asr-alignment-preparation]]'
  - '[[task-semcommit-main-labeling]]'
  - '[[decision-speechlm-all19-qc-candidate-scope]]'
  - '[[source-speechlm-human-review-57-20260925]]'
---

# speechlm 부분 정렬본의 semcommit 학습 승인 감사 (2026-09-25)

## 결론

**19개 DB의 QC 통과분을 semcommit 라벨링 입력으로 승인했다. 실제 학습은 아직 아니다.** 사용자가 1차 청취 표본 57개를 모두 통과로 판정했고, 검토 묶음과 고정 QC·평가셋 감사의 지문을 대조해 별도 승인 스냅샷을 만들었다. `data_approved=true`, `labeling_eligible=true`, `training_eligible=false`이다. 원천별 시간 상한은 적용하지 않는다. 사람 판정의 출처는 [[source-speechlm-human-review-57-20260925]], 범위 선택은 [[decision-speechlm-all19-qc-candidate-scope]], 이후 교사·학습 관문은 [[task-semcommit-main-labeling]]을 따른다.

## 고정한 입력과 전수 결과

- 원천: `/soundai/users/tskim/VAPKT-data/data/speechlm-asr-align-v1-full-b128-20260923/results`에서 완료된 6,824 샤드. 실행 중이던 추가 정렬을 재개하지 않았다.
- QC 고정 스냅샷: `/soundai/users/tskim/VAPKT-data/data/semcommit-work/qc/speechlm-partial-20260925-parallel-v1/summary.json`. 10,443,715 정렬 행 중 `QC_PASS_CANDIDATE` 8,807,815, `REVIEW_ZERO_DURATION` 1,614,874, 샤드 내부 중복 제외 21,026. 0길이 단어 보류분은 자동 승인 대상이 아니다.
- 기존 승인 학습셋의 파형·오디오 locator와 정확 중복 감사가 QC에 포함됐다. 본 단계의 통과 행은 **후보**이며 전사 품질의 직접 측정치는 아니다.
- 보호 평가셋 인덱스: `/soundai/users/tskim/VAPKT-data/data/semcommit-work/heldout-index-20260925-v1/summary.json`. LibriSpeech dev/test·KsponSpeech dev/eval·semcommit GigaSpeech test 골드의 19,437개 고유 구간을 동일 16 kHz PCM 규약으로 해시했다. AI Hub 71631 dev의 66,365개 구간은 경로 대조 범위이며 PCM 해시 범위는 **아니다**. 인덱스의 복호화 오류 0.
- 전량 후보와 보호 평가셋을 대조한 결과는 `/soundai/users/tskim/VAPKT-data/data/semcommit-work/approval-audit-20260925-v1/summary.json`: 정확 파형·경로 충돌 0, 원천 계열 보류 0. 후보는 8초 미만 7,566,927건, 8초 이상 1,240,888건이다.
- 텍스트 근접 스크리닝은 10,443,715 정렬 행과 보호 전사 24,252개를 비교했다. 길이 20자 이상, 12자 anchor 공유, 문자 유사도 ≥0.90, 발화 길이 차이 ≤25% 또는 0.75초 규칙의 적중 0건. `/soundai/users/tskim/VAPKT-data/data/semcommit-work/text-near-audit-20260925-v1/summary.json`. **이 결과는 음향 근접 중복 부재의 증명이 아니다.**

## 표본과 규모 판단

- 원 오디오를 다시 복호화해 파형 해시가 정렬 기록과 같은지 확인한 1차 청취 카드 57개: [로컬 검토 페이지](</Users/taesookim/Downloads/speechlm-semcommit-review-20260925/review-bundle-smoke-20260925-v1/index.html>). 19개 원천 × 짧은/긴/0길이 정렬 그룹당 1개다. **2026-09-25 사용자 판정: 57개 모두 통과.** 로컬·서버의 57개 오디오와 목록·HTML 지문이 일치한다. 이 판정은 57개 표본에 대한 것이며 모집단 전수 청취가 아니다.
- 확대 카드 285개(그룹당 5개): [로컬 검토 페이지](</Users/taesookim/Downloads/speechlm-semcommit-review-expanded-20260925/review-bundle-20260925-v1/index.html>). 오디오 285개 연결과 항목·HTML 해시를 확인했다. 자동 음향 경고(RMS·clipping 등)는 0개다. **확대 묶음은 사람 검토를 받았다고 기록하지 않는다.**
- 사용자는 **QC 통과 19개 DB 전부**를 이번 편입 후보 범위로 선택했다. 총 **8,807,815건 / 약 12,786.2시간**이며 원천·길이별 시간 상한은 적용하지 않는다. 8초 미만 **7,566,927건**은 버리지 않고 별도 풀로 유지한다. 이 풀의 **학습 step 비중 10%**는 이후 학습 샘플러 설계값이며, QC·승인 행을 10%로 자르거나 라벨링 물량을 줄인다는 뜻은 아니다. 코퍼스별·언어별·8 kHz별 혼합 비율은 학습 전 별도 검토가 필요하다.
- 전체 후보 범위 스냅샷: `/soundai/users/tskim/VAPKT-data/data/semcommit-work/approval-scope-all19-20260925-v1/summary.json`. 19개 원천과 QC 결정 파일의 SHA-256을 고정하고 `approval_status=scope_chosen_human_review_pending`, `training_eligible=false`로 저장했다. 880만 개 키를 중복 복사하지 않고 지문이 있는 QC 결정 파일의 `state == QC_PASS_CANDIDATE`로 범위를 정의한다.
- **현재 라벨링 입력 승인 뷰:** `/soundai/users/tskim/VAPKT-data/data/semcommit-work/source-approval-all19-20260925-v1/summary.json`. 승인 행은 고정된 QC 결정 파일의 `state == QC_PASS_CANDIDATE`인 **8,807,815건 / 12,786.242시간**이다. 0길이 단어 정렬 `REVIEW_ZERO_DURATION` **1,614,874건**은 19개 청취 예시가 통과했어도 승인 행에서 제외한다. 뷰는 검토 항목·57개 WAV 해시 목록·사용자 판정 원문·범위/QC/감사 지문을 검증한다. `approval_status=source_approved_for_semcommit_labeling`, `training_eligible=false`다.
- 이전 [[output-speechlm-arrow-asr-review]]의 P0 우선순위와 일부 원천 보류 권고는 **이번 범위 선택에 한해 대체**된다. 특히 명령·주소·숫자 중심 `000009/000017/000021/000026` 및 저우선 원천도 QC 통과분이라면 포함한다. 이는 해당 도메인의 전사·정렬 품질이 사람에게 검증됐다는 의미는 아니다.
- 보존된 대안: P0 7개, 원천별 긴 발화 160시간·짧은 발화 40시간 상한으로 만든 제안 뷰는 **477,380건 / 961.475시간**이다. `/soundai/users/tskim/VAPKT-data/data/semcommit-work/approval-proposal-p0-160plus40-20260925-v1/summary.json`에 있으며 현재 선택안은 아니다. 삭제하거나 학습 승인으로 전환하지 않았다.

## 남은 라벨링·학습 관문

1. 승인 뷰의 QC 결정 필터를 speechlm `words.jsonl` 빌더에 연결한다. 단어/토큰 시각과 `punct_final` 품질을 DB별로 점검하고 8초 미만 풀을 보존한다. 57개 표본 통과를 행별 전사 정답 보증으로 해석하지 않는다.
2. 텍스트 근접 스크리닝은 음향 근접 중복의 증명이 아니며 AI Hub dev는 경로만 대조했다. 이 한계를 학습 기록에 유지하고 필요하면 추가 음향 근접 감사를 한다.
3. semcommit 교사 골드 관문과 shard 처리량 시험을 통과한 뒤 본 라벨링을 진행한다. 교사 라벨·shard QC가 끝나기 전 `training_eligible=false`를 유지한다.
4. 학습에서 짧은 발화 step 10%와 코퍼스·언어·8 kHz 혼합 비율을 별도로 구현·검증한다. 확대 285개는 문제가 발견될 때 추가 검토 자료로 쓰며 이미 통과한 것으로 간주하지 않는다.

## 재현·출처

직접 CPU 실행 코드: `experiments/speechlm_semcommit_qc.py`, `experiments/speechlm_semcommit_heldout_index.py`, `experiments/speechlm_semcommit_approval_audit.py`, `experiments/speechlm_semcommit_text_near_audit.py`, `experiments/speechlm_semcommit_review_bundle.py`, `experiments/speechlm_semcommit_approval_proposal.py`, `experiments/speechlm_semcommit_approval_finalize.py`. 사람 판정 원문은 `raw/sources/experiments/2026-09-25-speechlm-human-review-57.json`에 보존한다. 각 산출물의 `summary.json`이 입력·출력 SHA-256과 완료 상태를 보존한다. 원천 데이터와 기존 승인 뷰는 변경하지 않았다.
