## [2026-10-21] task | V3 R2x·R4x 제출과 평가 큐

- Changed: `raw/sources/experiments/2026-10-21-v3-attribution/README.md`(13 절), `raw/sources/experiments/2026-10-19-stage4-dialogue-translation/stage4_eval_queue.sh`(단계마다 GPU 선택·OOM 한 번 이어 하기), [[status]]
- Reason: 사용자 "A 안으로 진행해줘" — R2x `v3-p1-xlw`(`--decision-weight-sets xl`, job 94525)와 R4x `v3-p1-xlw-arrival`(R2x + `--xl-slot-order arrival`, job 94526)를 사용자가 제출했다. 점검: r4 스모크 PARITY OK·평가 경로, r4x dry-run 불일치 0.
- Next: 학습 끝(PARITY) 확인, 평가 큐 8 개 완료 뒤 dev 선택·가드(같은 언어 태그 수·빈 교대·강제 진행 포함)·TAXI 보고, R4x 온셋 분류·slot-map rule/swap.
- By: tskim
