## [2026-09-20] query | ASR 원출력 보존과 대형 배치 추출

- Changed: [[output-data-selection-automatic-framework]], [[status]], 자동 후처리·전량 큐·교사 worker·출력 보존 테스트.
- Reason: 사용자는 메모리를 활용한 빠른 추출과 Qwen/Whisper 표기 보존을 요청했다. 저장 시 TN을 강제하지 않고 비교에만 적용하며 통과한 샘플의 Qwen 원출력을 명시적 pseudo-label 후보로 둔다.
- Next: 전량 입력 준비 완료 후 Qwen512/Whisper256·최대4 GPU 자동 시작. DB별 후보 수·처리 속도·peak 메모리·OOM을 점검한다. 원 학습 manifest는 유지하고 향후 export에서 명시적으로 타깃을 선택한다.
- By: tskim
