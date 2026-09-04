## [2026-09-04] decision | U0.5 관문 재정의·통과, U1 interleaved streaming ASR 학습 준비 착수

- Changed: `decision-asr-backbone`(관문 재정의), `task-uslm-u05-adapter-bridge` → done, 신설 `output-uslm-u05-adapter-bridge`(결과 보고서),
  `task-uslm-u1-interleaved-asr` → doing(설계·구현 기록), `README.md`, `TODO.md`, `wiki/status.md`, `wiki/index.md`.
  코드: `vapasr/uslm/interleave_data.py`(창 데이터셋), `vapasr/uslm/model.py::InterleavedASR`(joint chunk token·특수 토큰·스트리밍 디코더),
  `experiments/u1_train_interleaved.py`(학습 + 스트리밍 평가), `scripts/wiki-regen.py`(log/todo 재생성).
- Reason: 사용자 결정 — U0.5 관문을 "동일 인코더 RNN-T `[56,0]` 보다 우수 + 오프라인 Qwen 대비 ≤ +50 % 상대" 로 재정의. 원안(오프라인 ×1.15)은
  비인과 시스템 기준으로 인과 시스템을 재는 비교였다. 결과(18.2–18.8 / 17.5 / 14.5–15.1 %) 는 새 관문을 충족하므로 backbone 유지, U1 착수.
  U1 설계: 창 시작 = 양 화자 침묵, δ ∈ {2,3,4,6} 무작위 + `<DELAY_d>` 조건화, M=4, 특수 토큰은 임베딩 여유 행 + grad mask, U0.5 ckpt 초기화.
- Next: 스모크(`u1-smoke`) 통과 → v0 run(12k step) → 관문 판정(RNN-T `[56,0]` 대비 ≤ +10 %, 실질 목표 오프라인 U0.5 수준 유지) + 지연 분포·evidence 위반률 보고.
  정렬: otoSpeech 완료, vs02 진행(107/186) → 완료 후 val 창 확대. 인코더 상위 블록 unfreeze ablation.
- By: tskim
