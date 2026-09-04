## [2026-09-03] task | VAP baseline TurnBench dev 재현 완료 — 공식과 완전 일치

- Changed: `task-reproduce-vap-turnbench-baseline` → done, `wiki/outputs/output-vap-turnbench-baseline-reproduction.md`(신규),
  `raw/sources/experiments/2026-09-03-vap-turnbench-repro/`(점수 3종·예측 2종), `experiments/reproduce_vap_turnbench.sh`,
  `source-turnbench`(dev 수치), `turn-taking-evaluation-protocol`(기준선 표 split 명시), `task-stage1-encoder-probing`
  (head 학습 데이터 통일), `todo.md`·`TODO.md`·`status.md`·`index.md`·`log.md`.
- Reason: 사용자 요청. HF gated 데이터셋 3개 접근 확보 후 dev(38 대화)에서 (1) 동봉 예측 재채점, (2) 사전학습 원본
  직접 예측, (3) oto fine-tune 체크포인트 직접 예측을 수행했다. (3) 은 sweep 임계값(0.91615/0.85913)과 점수
  (EOT 0.841/0.045/463 ms, INT 0.957/0.100/896 ms)가 동봉 예측과 **완전히 일치** — 재현 성공. (2) 는 0.793/0.094/613 으로
  otoSpeech fine-tune 효과가 큼을 확인. 리더보드 수치(0.845/0.055/368)는 test split 이므로 논문에서 split 명시 필요.
  INT 는 FP(373) 가 TP(332) 수준으로 오경보가 과제. EOT p10 latency 가 음수(−34 ms) — projection 의 선점 사례.
  운영: GPU 배정 변경으로 GPU 3 → 1 로 이전 후 재실행. 심볼릭 링크 깊이 오류 1회.
- Next: [[task-build-vap-target-pipeline]](p0, 마지막 Phase 0), otoSpeech 290 GB·TS_01.실내_5 수신 완료 대기,
  [[task-add-missing-baselines]] 는 turnbench 동봉 baseline(rms_vad, dualturn, wavlm causal) 재사용.
- By: tskim
