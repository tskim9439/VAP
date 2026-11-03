## [2026-11-03] query | E13 결과(표준 재채점·번역)와 게이트 진단, 추론 게이트 판 준비

- Changed:
  - 위키: [[output-e13-gate-trans-20261102]] 7 절(결과·게이트 진단·게이트 판), [[output-ami-results-summary-20261030]](E13·E11 지연 완화 줄), [[output-diarizer-replacement-20261031]] 11 절 결과(화자마다 세션 99349 · 지연 완화 99459)
  - 코드(커밋 6aae1b7·a700fc4): 게이트 진단 `raw/sources/experiments/2026-11-03-gate/gate_diag.py`, 평가 `--gate-lags`·`--gate-thresh`(decode_blocks_batch gate_acts · realtime/long_block_decode gate_act · n3d_act keep_n3d_probs), 묶음 `q17-e13-gate`, `auto_gate.sh`·`gate_report.py`, E13 재채점 `auto_e13.sh`·`e13_report.py`, `tests/test_gate_override.py`
  - 그림 페이지 https://claude.ai/artifact/31pYMe3VVW2WNV1cpyG2SM (E13 구조 — 사용자 "E13 이 뭐였지? 그림과 함께 설명해줘")
- Reason:
  - E13 AMI 회의 통째 표준 24.39 / 31.33(eval) — 재귀속 없이 E11 + 재귀속(23.07 / 29.62)과 비슷, 귀속 몫 거의 0, Whisper agn +2.45 / +1.14. 번역 BLEU 평균 41.3(E6 41.4).
  - 사용자 "N3D 가 놓친 말을 받아쓰도록 게이트를 고칠 방법을 찾아줘": 태그가 닫혀 있던 참조 단어 IHM-mix eval 3.6 %, E13 은 그중 약 17 % 만(대부분 다른 슬롯), E11 은 열린 슬롯이 없던 구간 단어의 87 %.
    게이트 창 넓히기는 효과가 작다(200 청크에도 덮음 +1.3 pt). 학습 게이트는 정답 활동이라 모델은 활동 없는 열린 태그에서도 받아쓰도록 배웠다 → 추론 게이트 판(gt0.3·gt0.1·gl0)을 재학습 없이 잰다.
- Next:
  - 사용자 제출: 게이트 판 셋(판마다 노드 하나) → `auto_gate.sh` 로 표준 재채점·게이트 진단·비교 표.
  - 판 결과로 학습 쪽 수정(E14 — 추론과 같은 게이트로 학습, 닫힌 슬롯을 여는 출력 등)이 필요한지 정한다. G2b(99503) 뒤 E13 + G2b 학생.
- By: tskim
