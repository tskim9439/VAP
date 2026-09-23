## [2026-09-19] ingest | 데이터 선별 v0.3 추가 청취 32개

- Changed: `source-data-selection-human-review-v03.md`, `output-data-selection-repair-v03.md`, `wiki/status.md`; 원형 검수 JSON 사본.
- Reason: tskim이 전달한 `human-review-v03.json`을 32개 표본 key·source·라벨 스키마와 대조했다. 전사 실패 11·경계 실패 5·화자 실패 3을 원형대로 보존하고 실제 내용 오류와 표기 선호를 분리했다. wiki-ingest 절차에 따라 근거·상태를 연결했으며 기존 지정 경로의 DataSelectionPlan은 이동하지 않았다.
- Next: 실패/쟁점 keyed overlay와 생성 이력·원음 문맥 진단. lexical/display 범위가 명확해진 뒤 DB별 calibration 및 전량 선별. 자동 전사 교체·학습 승인·새 GPU 작업 없음.
- By: tskim
