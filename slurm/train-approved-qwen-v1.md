# 승인 Qwen-verbatim 데이터 학습 실행안

데이터 뷰:

```text
/soundai/users/tskim/VAPKT-data/data/selection/approved-qwen-verbatim-mono-v1.1-20260921
```

- `approved-en`: 342,352 streams
- `approved-ko`: 183,846 streams
- 정렬 승인: 1,767,472 (`6490bd2f...`), 최종 학습 가능: 1,766,695
- 동일 시각/초당 토큰 과밀 정렬 777건은 명시적으로 제외
- 타깃: Qwen3-ASR 원출력, TN 없음
- `speaker_training_eligible=false`, `turn_training_eligible=false`
- 기존 dev/test/eval과 exact audio path/locator 중복 0

기존 `BS_KO=48`은 짧은 Kspon 발화용 값이다. 새 KO 스트림 p50이 22.51초이므로
사용하지 않는다. 먼저 언어 모두 batch 8로 20-step 메모리 스모크를 한다.
batch 8, 8노드(64 rank), `proportional` 기준은 약 1,027 step/epoch다.

```bash
VIEW=/soundai/users/tskim/VAPKT-data/data/selection/approved-qwen-verbatim-mono-v1.1-20260921
sbatch --partition=apex --nodes=1 \
  --export=VAPASR_STRICT_SCHEMA=1,RUN=approved-qwen-v1-smoke,TRAIN=approved-en:approved-ko,EPOCHS=1,BS_EN=8,BS_KO=8,EVAL_EVERY=1000000,SAVE_EVERY=20,LANGUAGE_SCHEDULE=proportional,TRAIN_MAN_ROOT=$VIEW/manifests,TRAIN_ALIGN_ROOT=$VIEW/align,EXTRA_ARGS='--max-steps 20' \
  slurm/s3_train_hf.sbatch
```

스모크에서 GPU peak와 step 시간을 확인한 뒤 `BS_EN`/`BS_KO`를 따로 올린다.
`LANGUAGE_SCHEDULE=proportional`은 각 스트림을 epoch마다 정확히 한 번만 사용한다.
기존 `balanced` 모드는 작은 언어 셋을 반복하므로 이 데이터의 epoch 실험에는 쓰지 않는다.

본 학습은 로그인 셸이나 `.env`를 변경하지 않는 전용 wrapper로 제출한다. wrapper는
호출자 환경 전체(`ALL`)를 전달하지 않고 필요한 값만 해당 Slurm job에 전달한다.

```bash
./slurm/submit-approved-qwen-v1-e10-n2.sh
```

encoder는 기본으로 해동되며 `LR_ENCODER=1e-5`가 적용된다. encoder 동결 ablation만
위 `--export`에 `TRAIN_ENCODER=0`을 추가한다.
