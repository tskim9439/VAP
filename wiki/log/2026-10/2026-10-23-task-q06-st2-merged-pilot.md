## [2026-10-23] task | 0.6B 합친 Stage 2 시험 준비

- Changed: `slurm/submit-q06-st2-merged.sh`(새 파일), `slurm/submit-eval-ast.sh`(새 파일 — 단일 화자 AST·ASR 7 셋 평가 일반화), `raw/sources/experiments/2026-10-23-q06-st2-merged/README.md`, [[status]]
- Reason:
  - 사용자 "Ast 1-3 차는 뭐지?" → "화자 태그는 대화에서부터 하고, 한 단계로 통합하면 안 좋으려나?" → "Apex 노드 2대 더 사용할 수 있어".
  - Stage 2 의 세 차례(s2·s3·s4)는 평가 결과를 보고 덧붙인 것이다. 최종 설정으로 한 번 학습해도 s4 만큼 되는지 0.6B 로 확인한다.
  - 판단 기준은 README 에 미리 적었다.
  - 화자 태그는 대화 형식의 한 슬롯 경우이고 replay 데이터에도 필요해 그대로 둔다.
- Next:
  - 사용자 제출(학습 → 평가 afterok).
  - 평가 뒤 s4 와 짝 bootstrap 으로 비교한다.
  - 채택하면 대화 Stage 3·4 를 다시 내서 마지막 성능을 비교한다.
- By: tskim
