## [2026-10-19] task | Stage 4 파일럿 st4-p1 제출(apex job 91917) — p11 ck3000 + 대화 번역 0.5 + 교차 언어 합성 0.8, 평가 큐 셋

- Changed: [[output-stage4-prep-plan-20261019]] S6
- Reason: 사용자 제출(Stage 4 준비 완료 뒤)
- 설정:
  - init Stage 3 최종 p11 ck3000, 2 노드, lr 4e-5, 4000 step, 맞장구 1.0,3, 화자 헤드 0.5, 되살리기 v2
  - 대화 번역 `--dlg-ast-frac 0.5`(되살림 번역은 입력만), 교차 언어 합성 `--sim-trans-frac 0.8`
  - 가중치 dlg_ko:2, dlg_en·de:0.7, sim_ko:0.5, sim_en·de:0.8, sim_en-de:1.5, sim_en-ko:1.0, rep_*:1.2
  - 제외 노드 hpc-50·51·101·102(+ 기본 hpc-52·4)
- 결과: 13:48 hpc-78·79 에서 시작. 평가 큐(`stage4_eval_queue.sh`) ck2000·ck3000·final 을 호스트에 걸었다.
- Next: 체크포인트마다 같은 언어 대화 번역·영↔독 합성·TAXI·Stage 3 회귀·단일 화자 평가, p11(출발점)과 비교
- By: tskim
