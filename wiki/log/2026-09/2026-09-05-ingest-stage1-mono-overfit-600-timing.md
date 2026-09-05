## [2026-09-05] ingest | Stage 1 mono overfit @600 타이밍

- Changed: `raw/sources/experiments/2026-09-05-stage1-mono-overfit-600-timing.md`, `wiki/sources/source-stage1-mono-overfit-600-timing.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/status.md`
- Reason: KO·EN 16개 overfit의 내용·방출 타이밍·처리시간 결과를 보존하고, EN evidence-time 위반에서 실제 선행 방출과 반복 토큰 매칭 오류를 분리하며, 라벨 이월과 런타임 처리 backlog를 구분하기 위해 합성했다.
- Next: @900–1500 위반 토큰 문맥 감사, leading-silence shift test, deferred `<NEXT_AUDIO>`+다음 audio forward 최적화, 새 KsponSpeech 발음형 `align2/` QC와 overfit 재현, end-to-end chunk service time 기반 deadline miss·누적 runtime lag 측정.
- By: tskim
