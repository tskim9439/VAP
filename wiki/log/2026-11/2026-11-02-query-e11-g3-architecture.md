## [2026-11-02] query | E11·G3 모델 구조 설명(그림)

- Changed:
  - 위키: [[output-e11-g3-architecture-20261102]] 새 페이지(공통 슬롯 블록 LLM, E11, G3, 차이, 결과 요약 — Mermaid 그림 둘)
  - 위키: [[output-diarizer-replacement-20261031]] 10 절과 [[output-ami-results-summary-20261030]] 머리에 링크
  - 그림 페이지(비공개): https://claude.ai/artifact/GEYEpKf8DoeyxrH8LVxczS
- Reason:
  - 사용자 "G3와 E11 에 대한 모델 구조 설명을 그림과 함께 작성해줘."
  - 수치·구성은 E11 config, G2 graft.json, 코드(modeling_vapasr·n3d·n3d_graft·long_session)에서 확인했다. 이식 어댑터 크기(약 13M)만 설정으로 셌다.
- Next:
  - G3 를 한 forward 로 묶는 구현과 G2 다음 판이 정해지면 그림을 고친다.
- By: tskim
