## [2026-10-19] task | Stage 4 준비 결정과 S1·S2 — 대화 번역 원문 단위·시험 생성(관문 96–99 %), 네 흐름 시퀀스 구현·dry-run 통과, 학습 옵션

- Changed:
  - [[decision-stage4-prep-20261019]](신규, 리뷰 필요), [[output-stage4-prep-plan-20261019]](신규)
  - `experiments/stage4_dialogue_units.py`(신규) — 정제 대화 발화 → `ast_mono_generate` 단위(행 = 원 발화, 되살린 발화는 `--revive-dir`)
  - `vapasr/data/cst_dialogue.py` — 네 흐름(두 화자 전사 + 번역) `window_streams4`·`build_cst4_sequence`·`read_back4`·`eot_after_tails`·`clamp_to_spans`, 데이터셋 번역 모드(`trans`·`trans_input`·`ast_frac`)
  - `tests/test_cst_dialogue.py` — Stage 4 테스트 6 개(28 개 통과)
  - `experiments/cst_stage4_dryrun.py`(신규), `experiments/stage3_train.py` — `--dlg-trans`·`--dlg-trans-input`·`--dlg-ast-frac`
  - raw [[2026-10-19-stage4-dialogue-translation]]: README, `pilot_chain.sh`, `row_probe.py`, `dry_pilot.sh`, 시험 타깃 요약·dry-run 두 번의 집계
- Reason: 사용자 결정(실제 같은 언어 대화를 지금 번역, 교차 언어는 이어 붙인 합성 대화, TTS 병행, `<EOT>` 는 말이 끝날 때, δ 는 Stage 3 와 같게, 한–영 평가셋 보류) → "그리고 Stage 4 준비를 이어서 수행해."
- 결과:
  - 원문 단위: 일반 548 만·되살림 181 만 단어 × 방향(약 39 GPU 시간 추정)
  - 시험 생성(코퍼스당 3 대화): 수리 뒤 관문 en→de 98.9 %, en→ko 98.1 %, ko→en 95.7 %, de→en 98.6 %
  - 행 점검: 정제 조각으로 나뉜 행이 episode 여럿에 걸치는 경우가 1–15 % 다. 번역은 확정 시각이 든 episode 하나에만 준다.
  - dry-run 1 차에서 두 가지를 고쳤다. 번역이 원문 전사보다 앞서는 경우, `<EOT>` 가 마지막 전사 단어보다 앞서는 경우(`<EOT>` 의 25–41 %)다. 2 차는 7 코퍼스 386 창 문제 0 이다.
- Next:
  - 사용자: S1 전체 생성 제출(`stage4-dlg-units-v1`·`-revive-v1`, 생성 → 수리), 끝나면 타깃 묶기
  - S3 교차 언어 합성 대화, S4 번역 평가, S5 섞기·스모크
- By: tskim
