## [2026-10-08] task | AST 단조 번역 데이터 결정과 파일럿 준비

- Changed:
  - `wiki/decisions/decision-ast-monotonic-data.md`(신규)
  - `wiki/outputs/output-monotonic-ast-translation-methods-20261008.md`(stable prefix 논문 추가)
  - `experiments/ast_mono_pilot.py`(신규)
- Reason: 사용자가 계획 보고서의 열린 질문에 답했다.
  - Qwen3.8-27B 만 쓴다.
  - 파일럿은 방향별 50 건, 규모는 전체 ASR 전사다.
  - 학습에는 단조 번역만 쓴다.
  - prefix 는 화자별 언어 → 화자별 목표 언어 순서다.
  - stable prefix 논문(arXiv 2609.05799)을 적용하라고 했다.
- 결과:
  - 파일럿 스크립트를 작성했다. 단조·일반 번역, 판정, prefix 번역으로 방출 시각과 AL 을 계산한다.
  - mxc 접속 중계 맥북(Tailscale)이 오프라인이 되어 실행은 대기 중이다.
- Next: 중계 연결 복구 → 표본 추출 → 4 방향 GPU 실행 → 사용자 검토
- By: tskim
