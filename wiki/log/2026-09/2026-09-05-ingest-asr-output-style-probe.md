## [2026-09-05] ingest | Qwen3-ASR · Nemotron 출력 표기 실측

- Changed: `wiki/sources/source-asr-output-style-probe.md`, `wiki/concepts/asr-text-normalization.md`, `wiki/sources/source-qwen3-asr.md`, `wiki/sources/source-nemotron-3-5-asr-streaming.md`, `wiki/status.md`
- Reason: 두 사전학습 ASR의 실제 영어·한국어 숫자 출력과 구두점·태그 동작을 근거로, 늘어나는 코퍼스의 학습 타깃과 채점 표기를 하나의 규약으로 관리하기 위해 합성했다. 12개 목적 표본의 일반화 한계와 구현 감사에서 드러난 `num2words` 재현성 위험도 함께 기록했다.
- Next: `num2words` 의존성과 실패 정책을 고정하고 숫자 문맥별 단위 검사를 추가한 뒤, 새 KsponSpeech manifest·`align2/`로 overfit을 재검증한다.
- By: tskim
