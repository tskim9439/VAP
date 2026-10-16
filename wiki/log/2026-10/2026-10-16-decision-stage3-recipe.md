## [2026-10-16] task | Stage 3 레시피 결정 — 맞장구는 턴 끝 아님, 언어 비율 KO 2·EN 0.7·DE 0.7, replay 0.5, soft EOT; 확인 run p6·p7

- Changed:
  - [[output-stage3-prep-plan-20261014]]: "결정 (2026-10-16)" 절, 열린 질문 두 개 해결 표시
  - 평가 큐(mxc `results/stage3-regress/pilot_eval_queue.sh`): `BC=1.0,3` 이면 맞장구 규칙 참조
  - p2·p4·p5 를 맞장구 규칙 참조로 재채점(디코드 없음)
- Reason: 파일럿 비교([[2026-10-16-stage3-pilots]])를 보고 사용자가 권고안에 동의했다("그래 동감해 그렇게 진행하자").
- Next:
  - 사용자 제출: p6(조합), p7(조합 + replay 0.75 + lr 4e-5)
  - 끝나면 호스트 GPU 0·1 큐가 평가한다.
  - 결과로 본 학습 레시피·step 수를 정한다.
- By: tskim
