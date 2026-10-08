---
author_id: tskim
created: 2026-10-08
source_type: source-drop
---

# AST 단조 번역 파일럿 원자료 (mxc, 2026-10-08)

- 스크립트: `experiments/ast_mono_pilot.py`. Qwen3.8-27B 하나로 생성·판정하고, 생각 모드는 끄고 greedy 로 디코딩했다. mxc 호스트 GPU 0–2 에서 돌렸다.
- 표본: 방향별 50 문장(4–15 s, 단어 6–40 개).
  - 영어: YODAS·VoxPopuli 의 pnc_text
  - 한국어: speechlm display_text
  - 독일어: YODAS-Granary v1 display_text
  - 영어 50 문장은 영→독·영→한에 같이 쓴다.
  - 표본 파일: mxc `VAPKT-data/results/ast-mono-pilot-20261008/items-{en,ko,de}.jsonl`
- v1(커밋 cdbe719, ko-en 은 v2 로 넘어가며 중단):
  - 결과: mxc `VAPKT-data/results/ast-mono-pilot-20261008/`
  - 단조 번역 관문 통과 13–26/50
  - 전체 단조 번역을 JSON 조각 프롬프트로 만들어, 앞부분 번역과 표현이 달라 확정 시각이 늦어졌다.
- v2(커밋 2eb900a):
  - 결과: mxc `VAPKT-data/results/ast-mono-pilot-v2-20261008/` — 방향별 results.jsonl(앞부분 번역 포함), summary.json, review.tsv
  - 전체 번역과 앞부분 번역을 한 템플릿으로 만들었다.
  - 문법을 강조하고 방향별 예시를 넣었다.
  - 판정은 내용과 문법을 나눠 본다.
- 이 폴더:
  - `summaries.json`: v1·v2 방향별 요약
  - `v2-results-compact.jsonl`: v2 200 행. 앞부분 번역은 뺐고, 방출 시각·판정·두 번역은 들어 있다.
- 처리 속도: GPU 당 초당 약 140–160 토큰(transformers generate, 배치 32). 선형 attention 빠른 커널(flash-linear-attention)이 없어 torch 구현으로 돈다. 모델 로드는 427 s 다.
- 사용자 검토표: [Claude Docs](https://claude.ai/code/artifact/0c43a8c8-198e-44c1-a9e2-abbe889076f5)
