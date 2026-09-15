## [2026-09-15] task | Phase 2 Q0 데이터 준비 — D0 모듈·코퍼스 로더

- Changed: `vapasr/data/dialogue.py`·`lane_alloc.py`·`eot_soft.py`·`dialogue_interleave.py`·`dialogue_tokens.py`·`dialogue_corpora.py`·`dialogue_mix.py`, `experiments/p2_build_dialogues.py`·`p2_align.py`, `tests/test_lane_protocol.py`(19)·`test_dialogue_mix.py`(4); [[task-phase2-data-prep]] 진행 갱신.
- Reason: 사용자 지시 "확보된 DB 로만 Phase 2 데이터 준비". 기존 streams/interleave/targets 가 2화자 고정이라 dialogue 계층을 신설했다. lazy-free/never-free allocator, EOT soft target, lane 태그·ONSET·EOT 직렬화·flatten, 활동 타깃, 6 코퍼스 로더, mono 혼합기.
- Next: 서버 접속 복구 후 로더 실물 검증(ICSI Words·mrt 매핑), 코퍼스별 dialogues.jsonl 생성, p2_align SLURM 제출(사용자), Dataset 레코드·crop 창·QC.
- By: tskim
