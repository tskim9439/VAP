## [2026-10-09] decision | 다음 단계는 0.6B(q06-s1m3-d10/final)에서 시작

- Changed: `wiki/decisions/decision-stage1-lookahead-320-1040ms.md`(재검토 항목에 크기 결정 추가)
- Reason:
  - 0.6B 평가([[output-stage1-q06-d10-eval-20261009]])를 보고 사용자가 정했다.
  - 실시간성을 우선한다. 영어는 1.7B 와 같거나 낫고, 한국어는 0.6–1.6 %p 뒤진다.
- Next:
  - 화자·언어 prefix 형식을 확정한다.
  - AST 생성(job 83350)이 끝나면 행 단위 AST 타깃을 조립하고, 이 체크포인트에서 학습을 준비한다.
- By: tskim
