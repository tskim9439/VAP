## [2026-10-20] task | V2 첫 학습 v2-p1 제출(apex job 93883), 평가 큐(기존 + V2 추가) 4 대상

- Changed:
  - `experiments/cst_stage3_eval.py`·`cst_stage4_eval.py`·`cst_stage4_taxi.py`: 출력 해석은 모델 슬롯 전부(V2 4)
    - prefix 선언은 창 화자 수 그대로다. 2 인 창에서 C·D 로 낸 말은 대화 평가에서 삽입, TAXI 에서 삭제로 센다(예전처럼 태그를 텍스트로 읽지 않는다).
  - `experiments/cst_attr_errors.py`: 체크포인트 slot_tokens 로 같은 해석
  - raw `stage4_eval_queue.sh`: 빈 메모리 40 GB 이상 GPU 를 기다린다(OOM 방지)
  - mxc 본 checkout·개발 사본 반영(md5 대조)
- Reason: 사용자 제출(st4-m2a 설정 + V2: split v4·회의 dlg_mt·최대 4 화자·창 20–90 s draw 6·탭 L00·합성 v3·회의 번역 타깃, 3,000 step) → `/soundai/Model/VAPASR/stage3-v2-p1`
- 결과: 17:52 hpc-35·36 에서 시작. 평가 큐 = 기존 6 단계(`TRANS_LEAD_SPACE=1`) + V2 추가 12 단계 × ck1500·ck2000·ck2500·final
- Next: 학습 진행 확인 → 평가 → V1(m2a ck2000) 대조, 회의·AMI(IHM-mix·SDM)·3–4 인 합성 결과
- By: tskim
