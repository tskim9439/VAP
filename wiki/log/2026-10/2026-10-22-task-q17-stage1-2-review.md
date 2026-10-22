## [2026-10-22] task | Qwen3-ASR 1.7B 체인 Stage 1·2 검토

- Changed: `vapasr/hf/stage.py`(`weights_complete`·`last_complete_checkpoint`), `experiments/semcommit_train.py`·`experiments/stage3_train.py`(공용 재개 탐색), `slurm/submit-q17-s1m3-d10-s0.sh`(새 Stage 1), `slurm/submit-q17-s1m3-d10.sh`(worker 4), `slurm/submit-q17-s2-ast.sh`(init), `slurm/submit-q17-chain.sh`(`--s1`, 시간), `tests/test_resume_checkpoint.py`, `raw/sources/experiments/2026-10-22-q17-chain/README.md`, [[status]]
- Reason: 사용자 "Stage 1과 2 도 적합하게 수정되어 있는지 검토해줘". 실제 학습 인자(`results.json`·run 기록)와 10-08 job 82730 로그를 0.6B 와 대조했다.
  - 1.7B Stage 1 은 worker 8 로 노드 메모리 부족(oom_kill)에 세 번 죽었다(0.6B 82738 과 같은 원인).
  - 1.7B 조각 저장 체크포인트를 재개 탐색이 못 찾았다(`model.safetensors` 만 봄) → requeue 마다 init 부터 다시 했을 것이다.
  - Stage 1 레시피가 0.6B 와 달랐다(r3 뒤 영·한·독 1 epoch — 독일어·δ4–10 은 1/3). 기본을 0.6B 와 같은 `q17-s1m3-d10-s0` 로 바꿨다.
  - Stage 2 는 0.6B 와 같다. 번역 타깃 묶음(토큰 id)은 0.6B·1.7B tokenizer 어휘 지문이 같아 그대로 쓴다.
  - 1.7B·0.6B step 시간이 같아(데이터 읽기 병목) 체인 어림을 38 h → 약 30 h(+ 시작·재시작)로 고쳤다. 처음 어림은 `results.json` 학습 시간이 마지막 이어 하기 구간만 센다는 것을 놓쳤다.
- Next: 이어 하기 판 지문 dry-run 결과 기록 → 사용자 체인 제출(기본 mirror) → 1 단계 시작 로그(worker 4·데이터셋·step 시간) 확인 → 6 단계 뒤 평가·0.6B R1x 와 비교.
- By: tskim
