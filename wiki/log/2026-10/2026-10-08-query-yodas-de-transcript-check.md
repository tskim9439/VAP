## [2026-10-08] query | YODAS-Granary 독일어 전사 품질 점검

- Changed:
  - 코드: `experiments/yodas_de_qwen_check.py`, `experiments/yodas_de_whisper_check.py`
  - 원자료: `raw/sources/experiments/2026-10-08-yodas-de-transcript-check/`
  - 위키: `wiki/outputs/output-yodas-granary-de-transcript-check-20261008.md`
- Reason: 사용자가 독일어 실시간 ASR 학습용 YODAS-Granary 전처리 전에, 소규모로 Qwen 전사를 뽑아 원 전사 품질을 보고 ASR 필터 필요 여부를 정하자고 했다.
- 결과: 원 전사는 Whisper-v3 결과 그대로(재실행 CER 중앙값 0)라 v3 재실행은 무의미. 원 전사와 Qwen 일치는 시간 기준 약 절반이고, 불일치의 주원인은 정리된(말 그대로가 아닌) 전사·누락이라 ASR 필터는 필요. WER ≤ 3 % 원 전사↔Qwen 일치만으로 약 2,400 h 유지 예상.
- Next: 판정 규칙·문턱·학습 타깃 표기 결정(사용자) → 전량 전처리 파이프라인(사전 점검·평가셋 대조·Qwen 전사·판정·정렬·패키징)
- By: tskim
