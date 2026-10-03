## [2026-10-03] query | CST-Bench 합성음 계획 개정 — 원어민 검수 없음·상용 TTS 허용·평가셋만

- Changed: `wiki/outputs/output-cst-bench-synthetic-plan-20261003.md`
  - 내용: 사람 번역 병렬 대화(XDailyDialog·BConTrasT)를 뼈대로 사용
  - 참조: 한국어·생성 발화는 다중 LLM(EXAONE·Qwen3.8·상용 MT) 합의 silver 참조
  - TTS: CosyVoice3 와 상용 TTS 로 두 판을 전부 합성
  - 범위: 평가셋만
- Reason: 사용자 결정 — 원어민 번역 검수는 불가, 상용 TTS 사용 가능, 평가 전용 벤치마크.
- Next: 상용 TTS 약관·엔진 선택, S0 TTS 파일럿, QE 모델(CometKiwi 등) T5 확보
- By: tskim
