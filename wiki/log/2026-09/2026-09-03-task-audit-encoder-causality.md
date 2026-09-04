## [2026-09-03] task | Encoder causality·lookahead 감사 완료 — Qwen 조건 위반

- Changed: `experiments/causality_audit.py`(신규), `raw/sources/experiments/2026-09-03-causality-audit.json`(신규 raw),
  `wiki/outputs/output-encoder-causality-audit.md`(신규), `wiki/tasks/task-qwen-aut-causal-adaptation.md`(신규),
  `task-audit-encoder-causality-lookahead` → done, `decision-asr-backbone` 수정(감사 결과 절 추가, summary 변경),
  `question-encoder-lookahead-and-causality` → stable(답변), `source-qwen3-asr`·`source-nemotron`·
  `streaming-causality-and-latency-budget` 에 실측 반영, `TODO.md`·`todo.md`·`index.md`·`status.md`·`log.md` 갱신.
  부수: `scripts/aihub-download.sh`, `experiments/verify_aihub_sample.py`, `.env` AI Hub 항목 (사용자 질문 대응).
- Reason: Phase 0 p0. 특징 단위 절단 실험(fp32, rel tol 1e-3)으로 encoder 별 실효 lookahead 를 측정했다.
  CPC/VAP 0 ms(대조군). Nemotron `[56,0]` ≤80 ms, `[56,1]` ≤160, `[56,3]` ≤320, `[56,6]` ≤480, `[56,13]` ≤880 —
  문서의 chunk 크기가 최대 lookahead 임을 확인. **Qwen3 AuT 는 transformers/sdpa 경로에서
  `_prepare_attention_mask` 가 호출되지 않아 전체 발화 양방향**(n_window_infer 800 vs 100 출력 비트 동일),
  의도된 1 s 블록 모드도 lookahead 0–800 ms(평균 420) + 블록 간 좌측 context 부재. 320 ms 관문 발동 →
  Paper 1 은 Nemotron 단일 backbone(80/160 ms chunk), Qwen 은 적응 연구 결과에 종속으로 결정 수정.
  첫 실행에서 (1,128,T) 텐서의 mel 축을 잘라 Nemotron 이 0 으로 나온 버그를 잡았고,
  vap 패키지가 켜는 전역 deterministic 모드도 해제했다.
- Next: [[task-verify-aihub-stereo-and-access]](사용자 신청 대기), [[task-build-vap-target-pipeline]],
  [[task-reproduce-vap-turnbench-baseline]]. Nemotron ko-KR CER 을 80/160 ms chunk 에서 측정
  ([[task-latency-quality-curve]]). Qwen 적응 연구는 p2.
- By: tskim
