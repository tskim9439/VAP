## [2026-11-02] task | SDM 진단 — 학생 손해의 절반은 학습·추론 청크 차이(32.65 → 27.71), 나머지는 회의 중 슬롯 합쳐짐, G2b 준비

- Changed:
  - 위키: [[output-diarizer-replacement-20261031]] 13 절
  - 코드(커밋 1cd9ad0·d7244b6): `raw/sources/experiments/2026-10-31-diar-identity/g2b_diag.sh`·`g2b_smoke2.sh`, `experiments/n3d_graft_train.py --stream-chunk N1,N2`
- Reason:
  - 사용자 "우리 목표에서는 G3 형태로 가야 하는데, SDM 에서 성능이 안 좋네".
  - 저장 확률 분석: 학생 놓친 프레임 최대 확률 0.3–0.5 가 57 %, 회의 중간부터 화자 합쳐짐·슬롯 바뀜. 청크 16 진단: 학생 SDM eval 32.65 → 27.71, 교사 24.23 → 23.54.
  - 5 분 창 스모크(청크 16): 걸음 36–59 s, 40.5 GB, 재로드 대조 통과.
- Next:
  - 청크 4·16 번갈아 스모크 시간을 보고 G2b 제출 명령(회의 셋 4–6 분 창, 원거리 가중, 청크 번갈아)을 드린다.
- By: tskim
