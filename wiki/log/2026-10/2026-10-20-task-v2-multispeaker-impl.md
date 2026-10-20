## [2026-10-20] task | V2 첫 학습 준비 — 최대 4 화자·창 20–90 s 재추출·회의 번역 결정, 슬롯·시퀀스·창·평가·합성·학습 코드

- Changed:
  - [[decision-v2-run-scope-20261020]](새), [[output-v2-run-plan-20261020]](새)
  - `vapasr/hf/modeling_vapasr.py`(`enable_slots`·`SLOT_TOKENS`, `add_spk_head` 확장), `vapasr/hf/configuration_vapasr.py`(`slot_tokens`)
  - `vapasr/data/cst_dialogue.py`(N 슬롯·선언 슬롯·`shrink_speakers`·`speaker_firsts`), `vapasr/data/dialogue_dataset.py`(`window_draws`·`_windows_tiled`·`keep_items`), `vapasr/hf/packing.py`(draw 별 batch)
  - `vapasr/data/cst_eval.py`(N 슬롯 cp·오류 분해), `experiments/cst_stage3_eval.py`·`cst_stage4_eval.py`·`cst_attr_errors.py`(N 슬롯·창 인자·prefix 별 배치)
  - `vapasr/data/sim_dialogue.py`·`experiments/sim_build_dialogues.py`(N 화자·`--speakers`·`--p-back`), `experiments/stage3_train.py`·`slurm/submit-stage3-train.sh`(V2 인자·`dlg_mt`)
  - `tests/test_v2_slots.py`(새, 12 개), raw `2026-10-20-v2-multispeaker/sim_build_multi.sh`
- Reason: 사용자 "A 와 B 를 묶자. 그리고 최대 화자 수를 4명으로 늘리고 싶어." → 질문 답: 회의 번역 타깃도 생성, 창 20–90 s, V2-a 와 한 run 으로
- 결과:
  - 2 화자 동작은 그대로다: 기존 대화·합성·평가 시험 통과, 합성 2 인 세션은 HEAD 와 바이트까지 같다(5 seed). 로컬 실패 10 개는 변경 전과 같은 환경 문제다.
  - mxc 개발 사본(`VAPKT-dev` — 돌고 있는 평가 큐의 코드는 건드리지 않는다): 단위 시험 94 통과
  - 회의 원문 단위: 262,380 단위·2.04 M 단어(AMI 91 h·ICSI 56 h·NOTSOFAR 30 h 말소리)
  - 4 인 합성 모의: `p_back` 0.4 면 90 s 세션의 28 % 에서 한 화자가 발화 1 개 이하 → 0.2(16 %), 빌드에서 화자마다 발화 2 개 미만이면 다시 뽑는다
- Next: 회의 번역 생성 제출(사용자), 합성 셋 빌드, 오라클·dry-run·스모크, 학습 전 점검, 제출 명령
- By: tskim
