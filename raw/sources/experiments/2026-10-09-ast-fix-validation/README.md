---
author_id: tskim
created: 2026-10-09
source_type: source-drop
---

# AST 수리 단계(판정 v3 + 한 번 수리) 검증 (mxc 호스트 H200, 2026-10-09)

- 코드: `experiments/ast_mono_generate.py fix --validate --limit 2`. 판정 v3·수리 프롬프트는 `experiments/ast_mono_pilot.py` 의 `p_judge_v3`·`p_repair_v3` 이다.
  - 공통 앞부분은 판정이 785–793 토큰, 수리가 784–786 토큰이다. vLLM 784 블록을 넘어 캐시가 걸린다.
- 대상: gen-v3-bf16 의 방향별 u-00000·u-00001(총 8 샤드, 15,121 단위).
  - 검증 모드라 v2 통과 단위도 v3 로 다시 판정했다.
  - 결과 폴더: mxc `VAPKT-data/data/ast-mono-v1/fix-val-gen-v3-bf16/`
- 실행 조건: 검증 때 코드는 v3 통과를 그대로 인정했다(현재 규칙보다 관대). `flip-stats.txt` 의 strict(est) 는 현재 규칙을 적용한 추정이다.
  - 현재 규칙: v2 탈락 단위는 v3 가 내용 "same" 일 때만 그대로 쓰고, "minor" 면 수리로 보낸다.
  - 추정 방법: "minor" 로 통과한 단위가 수리를 거치면 그 샤드의 수리 성공률만큼 살아난다고 보았다.
- 파일:
  - `flip-stats.txt`: 샤드별 v2 탈락 수, v3 로 바뀐 수(내용별), 수리 수·성공, 통과율(v2 · 검증 당시 규칙 · 엄격 추정)
  - `samples.txt`: en-de·en-ko·ko-en u-00000 표본. v2 탈락 → v3 통과 8 개, 수리 4 개, v2 통과 → v3 탈락 3 개
  - `summaries.json`: 샤드별 summary(v2/v3 교차표 포함)
- 한 GPU(de-en)는 다른 사용자 프로세스(약 16 GB)가 있어 vLLM 메모리 0.95 를 확보하지 못하고 멈췄다. GPU 0 에서 0.90 으로 다시 돌렸다.
