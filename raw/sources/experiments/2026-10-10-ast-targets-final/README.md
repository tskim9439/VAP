---
author_id: tskim
created: 2026-10-10
source_type: source-drop
---

# AST 단조 번역 전체 생성·수리·타깃 조립 결과 (mxc, 2026-10-09 ~ 10-10)

- 생성:
  - `experiments/ast_mono_generate.py run`, 프롬프트 v3, bf16, 메모리 0.95·동시 1,024
  - job 83350(apex 2 노드, 10-09 16:59–23:55, 사용자가 파티션 변경을 위해 취소) → 83560(hpc 4 노드, sbatch 문법 오류로 워커 즉시 종료) → 83563(hpc 4 노드, 10-10 00:09–06:34, 02:19 선점 1 회 뒤 자동 재시작)
  - 3,474 항목 모두 완료, 오류 0, 합계 255.8 GPU 시간(항목 elapsed 합)
- 수리:
  - `ast_mono_generate.py fix`(판정 v3, 엇갈리면 v3 "same" 만 인정, 나머지 한 번 수리)
  - job 83564(hpc 4 노드, 10-10 06:37–07:35), 3,474 샤드, 오류 0, 19.8 GPU 시간
  - `fix-summary.txt`: 방향별 단위 수, v2 통과율, v3 "same" 로 인정된 수, 수리 성공/시도, 최종 탈락, 수리 뒤 단위 통과율
- 타깃 조립:
  - `experiments/ast_build_targets.py --fix-dir fix-gen-v3-bf16`(호스트 CPU 48 개, 병렬 단계 약 6.5 분 + 요약 단계)
  - 결과 위치: mxc `VAPKT-data/data/ast-mono-v1/targets-fix-gen-v3-bf16/`
  - `targets-summary.json`: 방향별 행 수·보존 행·탈락 사유·번역 토큰 수, 단위가 통과한 경로(v2/v3/repair/fail)
