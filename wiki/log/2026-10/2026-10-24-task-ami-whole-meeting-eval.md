## [2026-10-24] task | AMI 회의 통째 평가(--window full)

- Changed: `vapasr/hf/long_session.py`(새 — `encode_long`·`LongThinker`·`long_decode`, 문맥 창), `vapasr/hf/realtime.py`(`realtime_decode` 의 `on_chunk`), `vapasr/data/cst_eval.py`(긴 열 비트 병렬 편집거리), `experiments/cst_stage3_eval.py`(`--window full`·`--context-s`·`verify_long`), `experiments/cst_attr_errors.py`(full 설정 읽기), `tests/test_long_session.py`, `raw/sources/experiments/2026-10-21-v3-attribution/`(README 28 절, `v3_eval_task.sh amifull`, `amifull_timeline.py`), `slurm/submit-evals.sh`(v2-amifull·mt4-amifull·q17-mt4-amifull), [[status]]
- Reason:
  - 사용자 "AMI 평가에 한번에 돌릴 수 있도록 창 길이를 늘리고 싶은데, 자르지 않고 말이야".
  - 회의(14–50 분)를 창으로 자르지 않고 한 스트림으로 디코드해 세션 단위 cpWER 를 잰다.
  - 파일 전체 인코딩은 O(T²) 메모리이고, fp32 SDPA 는 긴 위치에서 느리다. 그래서 겹침 조각 인코딩과 블록 행렬곱 어텐션을 만들었다. 둘 다 기존 경로와 대조했다.
  - V2 final 은 문맥 전부로는 3.5–4 분 뒤 말을 그쳤다. 그래서 문맥 창 옵션을 넣었다.
  - 문맥 창으로는 받아쓰기가 창 평가와 같다(agn 20–22.5). 그러나 세션 cp 는 84–89 로, 창 평가 39.9 보다 훨씬 높다. 회의 수준 화자 귀속을 배우지 않았다.
- Next:
  - 사용자가 정한다: V2 로 34 회의 통째 평가를 낼지(`submit-evals.sh v2-amifull`, 약 4 h).
  - 회의 수준 귀속을 가르치는 학습 설계를 할지(앞 창 전사를 문맥으로 둔 이어 붙인 창, 회의 수준 슬롯).
  - 필요하면 문맥 창 디코드를 CUDA graph 로 빠르게 한다.
- By: tskim
