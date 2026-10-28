## [2026-10-28] task | E7 회의 통째 귀속 대책 — 사후 진단·디코드 때 태그 이끌기·E7b

- Changed:
  - `vapasr/data/cst_eval.py`(relabel_by_activity), `vapasr/hf/realtime.py`(realtime_decode retag), `vapasr/hf/long_session.py`(make_retag·long_decode retag)
  - `experiments/cst_stage3_eval.py`(--act-tag-guide·--relabel-act·--relabel-oracle)
  - `v3_eval_task.sh`(amifull GUIDE=1), `slurm/submit-evals.sh`(q17-e7-amifull-guide·q17-e7b)
  - `slurm/submit-q17-f1.sh`(--init), `slurm/submit-q17-e7b.sh`(새)
  - 시험: test_cst_eval·test_long_session 각 1
  - 위키: [[output-diarization-integration-20261025]] 7 절 대책 셋
- Reason:
  - 사용자가 1·2·3 을 모두 진행하자고 했다(사후 진단, 디코드 때 태그 이끌기, E7b).
  - 96172 회의 통째 cpWER 의 귀속이 46 pt 다. LLM 이 헤드의 회의 단위 슬롯을 따르지 않는다.
- Next:
  - 사후 진단: E7a AOSC 확률 저장 중(호스트 GPU 6·7, `results/diar-session-e7a-probs`). 끝나면 96172 원출력을 E7a·정답 활동으로 다시 귀속해 잰다.
  - 사용자 제출: `bash slurm/submit-evals.sh q17-e7-amifull-guide`(1 노드), `bash slurm/submit-q17-e7b.sh`(2 노드 + 평가 1 노드). E7b stage3 dry-run 이 확인 중이다.
- By: tskim
