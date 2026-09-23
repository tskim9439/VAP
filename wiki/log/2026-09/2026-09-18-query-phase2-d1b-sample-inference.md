## [2026-09-18] query | Phase 2 D1b 샘플 추론과 청취 시각화

- Changed: [[output-phase2-d1b-sample-inference]], `raw/sources/experiments/2026-09-18-phase2-d1b-sample-inference/`, D1b 샘플 청취·타임라인 뷰어, 동일 NIKL 원음의 Whisper-large-v2/v3 대조, [[task-phase2-data-prep]].
- Reason: 완료된 D1b 최종 체크포인트로 미학습 한국어·영어 대화 표본을 free-running 추론하고 원음·참조·예측·ONSET/EOT를 직접 검토하며, 한국어 내용 인식은 공개 Whisper 기준선 및 디코딩 경로와 비교하기 위해 생성했다.
- Next: 고정 manifest 50창 이상으로 확대하고 평가 v2의 pooled/ORC·cp/lane·event 축으로 정량화한다.
- By: tskim
