## [2026-09-21] query | 두 교사 일치 기준으로 유지 범위 확대

- Changed: [[output-data-selection-automatic-framework]], [[status]], `selection_pair.py`, `selection_reselect_pair.py`, 신규 테스트7개.
- Reason: 사용자는 원전사 일치 여부와 별개로 Qwen–Whisper 정규화 편집거리 ≤5%를 통과한 데이터까지 유지하도록 지시했다.
- Result: 전량3,757,805건·459shard 재선별 완료. 유지1,770,981건·2,359.703시간, v1 대비510,174건·722.486시간 증가. CPU8개·GPU0장·약153초. 원천·v1 결과 보존.
- Next: 두 유지 근거 집합을 분리한 채 학습 export와 경계/화자/턴 품질 관문에 연결한다. GPU 재추론·원전사 덮어쓰기·학습 제출은 하지 않았다.
- By: tskim
