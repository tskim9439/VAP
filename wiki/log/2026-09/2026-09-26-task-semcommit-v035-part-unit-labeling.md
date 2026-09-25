## [2026-09-26] task | semcommit v0.3.5·파트 단위 전량 라벨링 준비

- Changed: `vapasr/data/semcommit_llm.py`(v0.3.5: `-ㅂ니까` 의문 어미 제외, `rule_masks` conn_end_punct), `experiments/semcommit_{gold,teacher,train}.py`,
  `experiments/semcommit_{split_qc_pass,part_finalize,collect_done}.py`(신규), `experiments/semcommit_approve_speechlm_words.py`(파트 모드),
  `slurm/semcommit-part-label-{apex.sbatch,worker.sh}`·`slurm/submit-semcommit-label-v035-apex-n2.sh`(신규), 테스트.
  mxc: `semcommit-work/gate/speechlm-all19-g1-punctcoord-v035-20260926/`(accepted), `semcommit-work/qc-pass-split-v035-20260926/`(6,824 파트, 8,807,815 행).
- Reason: 파일럿(job 75518, 9,760 스트림, 약 40 분)에서 `-니까` 규칙이 `-ㅂ니까` 의문 어미를 잡고, ASR 마침표 때문에 연결어미로 끝난 발화 끝이 A 가 됨(스트림 끝 A 의 약 9 %) →
  v0.3.5 로 고치고 CPU 관문 재확인(accepted, 한국어 A 0.926 / 0.926, 영어 그대로). 사용자 요청: 단계별이 아니라 중간까지 만든 DB 로도 학습 가능하게 —
  파트 하나를 단어 → 승인 → A/B/C → 채점 → 풀 분리 → DONE.json 까지 끝내고 다음 파트로, 파트 순서는 DB 번갈아, `semcommit_collect_done.py` 가 끝난 파트만 모아 학습 목록(@파일)을 만든다.
- Next: 사용자 제출(`bash slurm/submit-semcommit-label-v035-apex-n2.sh`, 24 시간 단위로 다시 제출해 이어감) → 첫 파트들 QC → 중간 스냅샷으로 학습.
- By: tskim
