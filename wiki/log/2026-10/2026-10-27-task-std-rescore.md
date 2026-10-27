## [2026-10-27] task | AMI 표준 정규화 재채점 도구(Whisper 정규화 + meeteval cpWER)

- Changed:
  - `vapasr/data/cst_eval.py`(set_en_norm·whisper_en), `experiments/cst_stage3_eval.py`(--en-norm; E8 헤드 바꿔 끼울 때 shift 대조 포함)
  - `experiments/meeteval_cpwer.py`(새), `raw/sources/experiments/2026-10-27-std-rescore/rescore_std.sh`(새)
  - `tests/test_cst_eval.py`(1 — whisper_normalizer 가 없으면 건너뜀)
  - [[output-ami-external-comparison-20261027]](재채점 도구·E6 결과)
- Reason:
  - 사용자 "표준 정규화로 다시 채점하는 것도 만들어줘". 외부 AMI 수치(Parakeet·VibeVoice)와 같은 채점으로 맞추려는 것이다.
  - 서버 env 에 이미 있는 whisper_normalizer 0.1.15·meeteval 0.4.3 을 쓴다.
  - E6 final 결과:
    - Whisper 정규화로 창 cpWER 가 0.5–0.8 내렸다(SDM eval δ10 45.35 → 44.73 meeteval, IHM-mix eval 37.68 → 37.09).
    - 우리 cp 는 meeteval cpWER 와 같다.
  - 서버 시험 test_cst_eval 10 개가 통과했다.
- Next: 회의 통째 결과(96172·96461)가 끝나면 `rescore_std.sh amifull …` 로 다시 잰다.
- By: tskim
