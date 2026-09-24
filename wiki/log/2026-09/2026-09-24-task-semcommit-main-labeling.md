## [2026-09-24] task | semcommit 본 라벨링 실행 문서

- Changed: `wiki/tasks/task-semcommit-main-labeling.md`(신규), `wiki/outputs/output-semcommit-recipe-v0.3.md`(링크).
- Reason: 다른 세션에 mxc 대규모 라벨링을 맡기기 위한 절차서가 없었다(결과 페이지·골드 README·관문 스크립트에 흩어져 있고, 대화체 words 빌더·shard 실행·구두점 유무 점검·mxc 규칙은 문서화되지 않음).
  사용자 결정: 판정 LLM 은 EXAONE-4.0 과 Qwen3.8-27B 만 사용(Stage A·B·C 모두). 두 모델은 이 파이프라인에서 검증 전이라 어댑터 추가·스모크·골드 관문을 선행 단계로 넣었다.
- Next: 다른 세션이 이 문서대로 준비(코드·골드·모델 업로드) → 어댑터·스모크 → 관문 g1/g2 → words(새 빌더) → shard 라벨링 → 점검·기록.
- By: tskim
