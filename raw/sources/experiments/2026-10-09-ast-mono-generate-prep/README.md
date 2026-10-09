---
author_id: tskim
created: 2026-10-09
source_type: source-drop
---

# AST 단조 번역 전체 생성: 원문 단위 준비와 실제 크기 시험 (mxc, 2026-10-09)

- 스크립트: `experiments/ast_mono_generate.py`. mxc 결과 루트는 `VAPKT-data/data/ast-mono-v1/` 다.
- prepare(호스트 CPU 48 개, 704 s, 오류 0):
  - 입력은 0.6B Stage 1(q06-s1m3-d10) 학습 목록 네 개다. speechlm main·short, ASR words, YODAS-de tn2.
  - 행 id 중복은 0 이었다.
  - `manifest-stats.json`: 세트별 행·시간·단위·단어 수와 원문 출처(pnc_text/display_text/raw_text/words), 언어별 합계(샤드 2,749 개)
- 실제 크기 시험(호스트 H200, 작업 항목 하나씩):
  - `smoke-summaries.txt` 에 en-de u-00000 과 ko-en u-00000 의 summary 가 있다.
  - en-de: 1,024 단위, 호출 26,046, 770 s, 관문 통과 779
  - ko-en: 4,518 단위, 호출 29,518, 617 s, 관문 통과 3,778
  - prefix cache 적중은 0 토큰이었다(cached_tokens). 출력 한도에 걸린 호출은 8·20 개였다.
  - 두 결과는 `gen/` 에 그대로 두었다. 본 job 은 이 두 항목을 건너뛴다.
