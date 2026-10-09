## [2026-10-09] task | AST 수리 단계(판정 v3 + 한 번 수리) 구현·검증

- Changed:
  - 코드:
    - `experiments/ast_mono_pilot.py`: `JUDGE_EXAMPLES_V3`, `p_judge_v3`, `p_repair_v3`
    - `experiments/ast_mono_generate.py`: `fix` 하위 명령(엄격 규칙: 엇갈리면 v3 "same" 만 인정)
    - `experiments/ast_build_targets.py`: `--fix-dir` 반영
    - `slurm/ast-mono-generate.sbatch`·`submit-ast-mono-generate.sh`: `--fix`(MODE=fix)
    - `tests/test_ast_mono_generate.py`: 가짜 엔진으로 수리 경로·타깃 반영을 시험
  - 원자료: `raw/sources/experiments/2026-10-09-ast-fix-validation/`
  - 위키: `output-ast-mono-generation-20261009`(수리 단계 절), `decision-ast-monotonic-data`(결정 10)
- Reason:
  - 영어 원문 행이 AST 타깃에서 많이 빠졌다(en-de 63 %·en-ko 56 % 만 남음).
  - 사용자가 생성이 끝나면 재판정·수리를 바로 진행하라고 했다.
- 결과(방향별 2 샤드 검증):
  - 단위 통과율은 v2 72–90 % 에서 엄격 규칙 추정 89–99 % 로 오른다.
  - v2 통과 단위의 98–99.7 % 는 v3 도 통과시킨다.
  - 표본 확인 결과 v3 단독 "minor" 에는 실제 누락이 섞여 있다. 그래서 엇갈린 단위는 "same" 만 인정한다.
- Next:
  - 생성(job 83350)이 끝나면 사용자가 `bash slurm/submit-ast-mono-generate.sh --nodes 2 --partition apex --job-name SA_SFT_FullDuplex --fix` 로 제출한다.
  - 제출 뒤 `ast_build_targets.py --fix-dir fix-gen-v3-bf16` 로 타깃을 조립한다.
- By: tskim

- 정정(2026-10-10 00:10):
  - sbatch 에 수리 모드를 넣으며 환경변수 줄(`\` 로 이어짐) 뒤에 `if` 문을 끼워 `bash -c` 안쪽이 문법 오류가 됐다. 그래서 재제출한 생성 job 83560(hpc 4 노드)의 워커 32 개가 바로 종료 코드 2 로 끝났다. 결과 파일 손상은 없다.
  - 환경변수를 `export` 로 바꿔 고쳤다. 안쪽 명령을 sbatch 에서 그대로 꺼내 가짜 python 으로 실행해 run·fix × bf16·fp8 의 인자를 확인했다.
  - `bash -n` 은 `bash -c '…'` 안의 문자열을 검사하지 않는다.
