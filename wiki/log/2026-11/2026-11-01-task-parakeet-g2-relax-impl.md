## [2026-11-01] task | Parakeet + G2(저장 확률 되돌려 주기)·E11 지연 완화 평가 구현

- Changed:
  - 코드(커밋 fb9517f): `experiments/baseline_ami_ext.py --diar-probs`, `experiments/cst_stage3_eval.py --encoder-rc`, `vapasr/hf/long_session.py`(기본 조각을 청크 배수로), `tests/test_long_session.py`, `raw/sources/experiments/2026-10-30-ami-ext/ami_ext_task.sh`(mtp-g2·mtp-n3d-replay), `raw/sources/experiments/2026-10-21-v3-attribution/v3_eval_task.sh`(ENC_RC·N3D_RC), `slurm/submit-evals.sh`(ext-parakeet-g2·q17-e11-relax)
  - 위키: [[output-diarizer-replacement-20261031]] 11 절
- Reason:
  - 사용자 "둘 다 구현해놓자" — 오프라인 쪽 평가(지연 완화)와 Parakeet + N3D 와 같은 꼴의 G2 평가.
  - 맥 시험: long_session 10·N3D 10·이식 9 통과. 맥 배선 점검으로 저장 확률 경로를 확인했다(화자 분리기가 불리지 않음).
  - 인코더 설정이 지원하는 우측 문맥은 0·3·6·13 이다(모델 카드의 160 ms(1)는 설정 목록에 없다).
- Next:
  - 공용 GPU 가 비면 서버 스모크(지연 완화 창 8·회의 1 --verify, Parakeet + 저장 N3D 확률 회의 2 — 직접 잰 값과 대조).
  - G3 평가(99301)가 끝나면 `v3_eval_task.sh` 를 서버에 올린다(실행 중 bash 스크립트를 바꾸지 않는다).
  - 사용자 제출: `submit-evals.sh ext-parakeet-g2`, `submit-evals.sh q17-e11-relax`.
- By: tskim

## [2026-11-01] task | 정정 — 우리 모델 + N3D, 화자마다 세션(Diarization Masked Input) 구현

- Changed:
  - 코드: `experiments/baseline_ami_ext.py --system vapasr-spk`, `vapasr/hf/long_session.py`(encode_long act — 바깥에서 준 슬롯 활동), `ami_ext_task.sh`(vspk-g2·vspk), `slurm/submit-evals.sh`(q17-spk), `tests/test_baseline_ami_ext.py`
  - 위키: [[output-diarizer-replacement-20261031]] 11 절
- Reason:
  - 사용자 "1번 반대야. G2 + N3D 조합으로 추론하는거, 그런데 화자마다 세션을 부여하고". 앞의 Parakeet + G2 는 역할을 거꾸로 읽은 것이다(코드는 남긴다, 제출할 필요는 없다).
  - 우리 E11 은 대상 화자 조건을 학습하지 않아, 학습 없는 Masked Input 으로 화자 세션을 만든다.
- Next:
  - 서버 스모크(SDM 회의 1 개)를 본 뒤 사용자 제출 `submit-evals.sh q17-spk`.
- By: tskim
