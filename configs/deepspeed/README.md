# DeepSpeed 설정

`--deepspeed configs/deepspeed/zero2.json` (vapasr/speedup.py) 로 `s3_train_hf.py`·`p2_train_hf.py`·`semcommit_train.py` 에 넘긴다. torchrun 으로 실행한다.

- ZeRO-2 만 쓴다(optimizer state·gradient 분할). ZeRO-3 은 파라미터를 쪼개 인코더(NeMo)·`build()` 의 임베딩 gather 와 맞지 않는다.
- `bf16.enabled=false` + `torch_autocast` bf16: DeepSpeed 의 bf16 모드는 모델 전체를 bf16 으로 캐스팅해 fp32 로 돌아야 하는 동결 인코더를 망가뜨린다.
  master weight 는 fp32 그대로, 계산만 autocast. 이때 `TrainingArguments(bf16=False)` 로 둔다(training_args_kwargs 가 자동 처리).
- 0.6B thinker 는 DDP 로도 메모리가 충분해 이득은 optimizer state(약 7 GB/GPU) 절감 정도다. 8 GPU 이상에서 배치·packing 예산을 키울 때 쓴다.
- `deepspeed` 패키지가 vapasr env 에 없다 — 로컬에서 wheel 을 받아 mxc 로 올려 설치한다(서버에서 직접 받지 않음).
