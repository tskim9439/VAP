## [2026-10-01] query | 학습 단계 재설계(Stage 0/1/2)와 1.7B·final 모드 구현

- Changed:
  - `wiki/outputs/output-staged-training-plan-20261001.md`(신규)
  - 코드 커밋 df19d9a: final 모드 형식·디코드·평가·self 2-pass, from_qwen 1.7B
  - 제출 스크립트 단계 옵션
  - mxc 배포: 데이터셋·학습 스크립트의 --asr-only/--offline-frac/--freeze-thinker
- Reason: 사용자가 "WER/CER 이 가장 중요하다, 1.7B·AuT 인코더·단계 분리를 분석해 구현하라, δ 스트리밍과 final 학습 병행도 고민 중"이라고 했다.
- Next:
  - 사용자가 0.6B·1.7B 파일럿(Stage 0 → Stage 1)을 제출한다.
  - single-turn 평가(--final 포함)로 크기를 결정한다.
  - 다음은 Stage 2 SEM 이다.
  - Codex 미커밋 short 풀 변경의 커밋 여부를 결정해야 한다.
- By: tskim
