## [2026-09-10] query | Stage 2 run D2 최종 평가 보고서

- Changed: `wiki/outputs/output-stage2-d2-final-eval.md` 신규(§4 스윕·C2 δ4 는 측정 중), `vapasr/uslm/mono_data.py` 결함 수정 기록
- Reason: 확장 DB run D2 의 최종 성능을 C2 와 같은 표본으로 비교. 평가 중 71631 오디오 조립 버그(src_offset_s 누락)를 발견해 D2 를 오염 run 으로 판정
- Next: D3(수정 로더) 학습 → 같은 보고서 형식으로 비교, E1 인코더 해동은 D3 위에서
- By: tskim
