## [2026-10-30] task | E10 첫 평가 무효 — <BEGIN> 디코드 차단 수정, 다시 재기 준비

- Changed:
  - 코드(커밋 c1c551b): `vapasr/hf/modeling_vapasr.py`(add_slot_blocks 이미-있음 경로·__init__ 에서 <ONSET> 차단 풀기), `tests/test_slot_blocks.py` 시험 1 개
  - 평가: `v3_eval_task.sh` EV_TAG(산출 이름 꼬리), `slurm/submit-evals.sh` q17-e10-fix
  - 위키: [[output-diarization-integration-20261025]] 10 절 E10 결과 0
- Reason:
  - E10 평가 97923 의 AMI 창 cpWER 이 약 96 이었다. E10 config 에 <BEGIN>(<ONSET>)이 막혀 새 턴을 열지 못했다.
  - 슬롯 블록 init(E9 final)에서 add_semantic_tokens 가 다시 막고 add_slot_blocks 가 일찍 돌아가 풀지 못했다.
  - 사용자가 97923 을 취소했다.
- Next:
  - 확인(AMI SDM 창 16 개): 수정 후 E10 cp 52.33 · agn 28.41 · <BEGIN> 403 번. E9 같은 창 50.57 · 31.16 · 371 번.
  - 사용자 제출: `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-evals.sh q17-e10-fix)`
  - E11(98306, 같은 버그로 저장되지만 읽을 때 풀린다) 평가 98307 은 그대로 둔다.
- By: tskim
