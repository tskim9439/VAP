## [2026-09-14] query | 고정 화자 ID와 A/B 교대 표식 비교

- Changed: [[output-phase2-speaker-representation-comparison]] 신규 분석 보고서.
- Reason: 최대 화자 수·뒤쪽 슬롯 학습 희소성·3중 겹침을 고려한 표현 선택 질문. 세션 N·동시 S·전사 채널 R을 구분하고 t-SOT/t-vector/Streaming Sortformer 원문과 대조했다.
- Next: 현행 K baseline의 실제 Dataset 검증과 N/S별 병목 분리. 동적 speaker memory는 미채택 제안이며 정본·registry·결정·코드는 변경하지 않았다.
- By: tskim
