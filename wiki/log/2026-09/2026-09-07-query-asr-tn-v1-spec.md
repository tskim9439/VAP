## [2026-09-07] query | asr-tn-v1.0.0 동결 후보 규약

- Changed: `wiki/outputs/output-asr-tn-v1-spec.md`, `wiki/concepts/asr-text-normalization.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/status.md`
- Reason: 1,930 h manifest·alignment 생성 전에 LibriSpeech·KsponSpeech의 target/score 규약, 지원·미지원 숫자 패턴, quarantine, golden test, fingerprint와 버전 조건을 고정할 문서 정본이 필요했다.
- Next: TN v1 구현, 전체 transcript audit, 기존 230 h target/token-ID diff, 목적 표본 검토 후 frozen 판정.
- By: tskim
