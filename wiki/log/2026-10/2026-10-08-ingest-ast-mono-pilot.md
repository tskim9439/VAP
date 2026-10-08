## [2026-10-08] ingest | AST 단조 번역 파일럿 v1·v2 결과와 검토표

- Changed:
  - `experiments/ast_mono_pilot.py`(v2)
  - `raw/sources/experiments/2026-10-08-ast-mono-pilot/`
  - `wiki/outputs/output-monotonic-ast-translation-methods-20261008.md`
- Reason: 사용자가 S2TT 데이터 작업을 이어 실행하라고 했다. 방향별 50 문장 파일럿을 돌리고 검토표를 만들었다.
- 결과:
  - v1 은 단조 번역의 관문 통과가 13–26/50 이었다. 원인은 문법이 깨진 것과 전체·앞부분 번역의 프롬프트가 달랐던 것이다.
  - v2 는 통과 36–44/50, AL 12–42 % 감소다.
  - 처리 속도는 GPU 당 초당 약 150 토큰이라 전체 규모에는 vLLM 이나 빠른 커널이 필요하다.
- Next: 사용자 검토(Claude Docs 검토표) → 프롬프트 확정 → 추론 엔진 결정(vLLM / flash-linear-attention) → 전체 생성
- By: tskim
