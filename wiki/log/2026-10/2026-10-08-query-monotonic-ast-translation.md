## [2026-10-08] query | 어순을 맞춘(단조) 번역으로 AST 데이터 만들기 — 선행 연구

- Changed: `wiki/outputs/output-monotonic-ast-translation-methods-20261008.md`(신규)
- Reason: 사용자가 기존 전사를 번역해 AST 데이터를 만들 때 원문 어순에 가깝게 번역하는(내용 일치·문법 유지) 방법이 제안된 적 있는지 물었다.
- 결과:
  - Simul-MuST-C(EMNLP 2024): GPT-4o + salami 기법. 영→일 단조성이 0.57 에서 0.82 로 오르고 품질·지연이 좋아졌다. 영→독은 이득이 작았다.
  - 그 밖의 확인: LLM-SI-Corpus, Chen 외 2021, Han 외 2021, Wang 외 2023, Simul-COMET
  - 제안: 후보 생성 → 내용·문법 관문 → 정렬 시각 기반 오라클 lag 최소 후보 선택
- Next: 방향별 1–2 천 발화 파일럿(사용자 결정)
- By: tskim
