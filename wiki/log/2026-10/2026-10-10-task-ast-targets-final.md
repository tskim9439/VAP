## [2026-10-10] task | AST 번역 생성·수리 완료, 최종 번역 타깃 조립

- Changed:
  - mxc:
    - `VAPKT-data/data/ast-mono-v1/gen-v3-bf16/`(3,474 항목)
    - `fix-gen-v3-bf16/`(3,474 샤드)
    - `targets-fix-gen-v3-bf16/`(방향별 타깃 + summary.json)
  - 원자료: `raw/sources/experiments/2026-10-10-ast-targets-final/`
  - 위키: `wiki/outputs/output-ast-mono-generation-20261009.md`(최종 결과 절)
- Reason:
  - 사용자가 진행 현황을 물었다.
  - 생성(job 83563)과 수리(job 83564, 생성 뒤 의존성으로 시작)가 끝나 타깃을 조립했다.
- 결과:
  - 행 보존은 de-en 99.1 %, en-de 92.8 %, en-ko 86.5 %, ko-en 97.4 % 다(수리 전 영어 63·56 %).
  - 번역 토큰은 모두 1 억 4,516 만 개다.
  - 생성 255.8 + 수리 19.8 GPU 시간, 오류 0
- Next:
  - 평가 스크립트를 ast-v1 prefix(화자 태그·translate 줄)와 번역 평가에 맞춘다.
  - 0.6B(q06-s1m3-d10/final)에서 시작하는 ast-v1 학습 제출 래퍼를 만든다.
- By: tskim
