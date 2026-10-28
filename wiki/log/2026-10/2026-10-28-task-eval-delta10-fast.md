## [2026-10-28] task | 빠른 평가 — AMI IHM·SDM δ10 만

- Changed:
  - `v3_eval_task.sh`: AMI 작업(ami·amiact·amitgt·amie3·amifull)에 DELTAS 를 넣었다. 기본 4,10 이고, 다르면 산출 이름에 -d10 등이 붙는다. AMI 밖 작업은 4,10 고정이다.
  - `slurm/submit-evals.sh`
    - ami8d10(창 AMI δ10 8 몫), q17-e8-ami10, q17-e6-amifull-d10
    - q17-e7-amifull-guide·q17-e7b 를 δ10 으로
  - `slurm/submit-q17-e8.sh`: 평가 묶음 기본을 q17-e8-ami10 으로
- Reason:
  - 사용자 "평가하는 데, 너무 오래 걸린다. 일단 AMI IHM 이랑 SDM 두개에 대해 delta 10 으로만 평가해서 비교해서 의사결정을 빠르게 하자".
  - E8 의 예전 평가 묶음(q17-e8, v3 묶음 45 과제)은 약 4 h 다. 창 AMI δ10 8 몫이면 한 노드에서 약 20 분이다.
- Next:
  - 사용자 제출(10:35)
    - E8 예전 평가 96582 를 취소했다.
    - E8 빠른 평가 q17-e8-ami10 은 job 96941(96581 성공 뒤)이다.
    - 태그 이끌기 q17-e7-amifull-guide 는 job 96942 다.
  - E7b 는 stage3 dry-run 이 확인되면 낸다.
    - 첫 dry-run 은 `--freeze-encoder` 와 `--train-encoder` 가 배타 옵션이라 exit 2 였다.
    - submit-stage3-train 을 고쳐 다시 돌리고 있다.
- By: tskim
