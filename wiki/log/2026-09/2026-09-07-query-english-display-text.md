## [2026-09-07] query | 영어 대소문자·문장부호 출력 규약

- Changed: `wiki/outputs/output-asr-tn-v1-spec.md`, `wiki/concepts/asr-text-normalization.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/sources/source-asr-output-style-probe.md`, `wiki/status.md`
- Reason: 영어 대소문자와 문장부호를 평가에서 제거하는 데 그치지 않고 최종 ASR 출력 능력으로 학습·평가해야 한다는 요구를 TN v1에 반영했다.
- Next: lexical audit 동결 후 1,930 h alignment를 시작하고, display 평가 subset과 provenance, Qwen exact-match 필터·masked loss·지표를 full-FT 전에 별도로 동결한다.
- By: tskim
