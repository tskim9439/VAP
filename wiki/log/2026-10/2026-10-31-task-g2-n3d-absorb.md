## [2026-10-31] task | G2(학생 N3D 위층 함께 학습) 구현·서버 스모크, N3D 흡수(A1) 맥 구현 — 제출 명령 준비

- Changed:
  - 코드(커밋 369f812, 서버 반영·md5 대조): `vapasr/hf/n3d_graft.py`(LoraLinear·n3d_trainable·load_n3d_trainable, GraftExpert 가 n3d_state 읽기), `experiments/n3d_graft_train.py`(--n3d-train lora|full 등, rank 평균·일치 검사, 저장·재개·재로드 대조), `tests/test_n3d_graft.py`, `slurm/submit-evals.sh`(q17-graft 회의 통째 --save-probs), `slurm/submit-n3d-graft.sh`(G2 쓰기), `raw/sources/experiments/2026-10-31-diar-identity/g2_*.sh`
  - 코드(커밋 88adc86, 맥만 — 서버 반영은 장치별 판이 끝난 뒤): `vapasr/hf/modeling_vapasr.py`(absorb_n3d·_shared_mel·_load_embedded_n3d, save·from_pretrained), `vapasr/hf/n3d.py`(build_n3d 분리, stream_probs·N3DExpert 의 pre, 가지 늘 eval), `vapasr/hf/stage.py`(n3d_config.yaml), `experiments/n3d_absorb.py`, `tests/test_n3d.py`
  - 위키: [[output-diarizer-replacement-20261031]] 8 절(구현·스모크·제출 명령)
- Reason:
  - 사용자 "진행해봐"(G2), "N3D 흡수하는 방향도 좋아".
  - G2 서버 스모크(U3 final 에서, AMI·ICSI 60–90 s 창, 배치 2):
    - 단일 GPU 4 step: LoRA 6.03M, 12–19 s/step, 20.2 GB, 재로드 max_diff 0.
    - 2 rank 3 step: 15 s/step, 21.4 GB, rank 일치 검사 통과, 재로드 max_diff 0.
  - 흡수: N3D 체크포인트의 filterbank·window 가 bf16 이라 우리 전처리기와 mel 값이 달랐다(log-mel 최대 1.96). 그래서 기본은 N3D 자신의 mel 모듈을 가지 안에 둔다(출력 = E11).
  - 맥 시험: 이식 9 개·N3D 10 개 통과. 전체 841 통과, 실패 9 는 환경·실행 순서 탓이다(따로 돌리면 통과하거나 무관한 모듈).
- Next:
  - 사용자: G2 제출(위키 8.1 절 명령 — low_p_hpc 1 노드, 학습 뒤 q17-graft·q17-u3-asr 평가가 이어서 난다).
  - 장치별 판(98956–98963) 학습·평가가 끝나면 흡수 코드를 서버에 올리고 E11 → 흡수판 변환·평가(E11 과 같아야 한다).
  - G2 결과에 identity_drift.py 를 돌려 30 s 혼동과 긴 거리 몫을 본다.
- By: tskim
