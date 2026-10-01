---
author_id: tskim
created: 2026-10-01
source_type: source-drop
---

# Stage 0/1 파일럿 평가 원자료 (mxc, 2026-10-01)

- 모델: `/soundai/Model/VAPASR/semcommit-{q17,q06}-s1/final`
  - job 77755(1.7B)·77757(0.6B), 2 노드 × 8 GPU, 1 epoch 3,980 step
  - init 은 각 Stage 0(77754·77756, adapter 워밍업 2,000 step)
  - Stage 0·1 모두 저장·재로드 parity 통과(키 차이 0, 인코더 638 키 일치)
- 평가: `single-turn-vpaa-kspon-v1`(VoxPopuli-AA 628 + Kspon eval_clean 3,000 + eval_other 2,687), δ 2/4/6 + final, fp32 greedy, tail 1 s, mxc 호스트 GPU 3–6.
- 기준선:
  - E2·v035-d8 의 Kspon 은 single-turn-v1 결과에서 가져왔다. Kspon 행은 바이트 단위로 같다.
  - VoxPopuli-AA 는 `single-turn-vpaa-*`(2026-10-01, GPU 1·2)
- 파일:
  - `s1_compare.txt`: 오류율 표 + 짝 bootstrap(2,000 회; VoxPopuli 는 세션 군집 292, Kspon 은 발화 단위)
  - `s1_compare.py`: 위 표를 만든 스크립트(mxc results/ 에서 실행)
  - `vpaa_bootstrap_E2_vs_v035.py`: VoxPopuli-AA 에서 E2 대 v035-d8
  - `summaries-noGroups.jsonl`: 두 모델 summary.json(groups 제외, AA-WER·config 포함)
- 발화별 predictions 는 mxc `VAPKT-data/results/single-turn-vpaa-kspon-{q17,q06}-s1/` 에 있다.
