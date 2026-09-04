# env/ — 컨테이너 학습 환경

| 파일 | 역할 |
|------|------|
| `requirements.txt` | 손으로 정리한 최소 의존성 (사람이 읽는 용도) |
| `requirements-lock.txt` | `pip freeze` 결과. **서버에서 생성**되며 정확한 재현용 |

## 설치

컨테이너(`tskim_env`) 안에서:

```bash
bash scripts/setup-container-env.sh          # 전체: conda torch nemo hf verify
bash scripts/setup-container-env.sh hf verify  # 일부 단계만
```

conda env 이름·Python 버전·캐시 경로는 저장소 루트 `.env` 에서 읽는다.

## 사용

```bash
source scripts/activate-env.sh   # .env 로드 + conda activate vapasr
python scripts/smoke-test-models.py   # 두 backbone + aligner 로드·추론 확인
```

로컬에서는 `./scripts/sync-rack4.sh shell` / `exec` 가 자동으로 activate 한다.

## 주의

- 모델 가중치 캐시는 `/data3/tskim/cache/{huggingface,nemo,torch}` 다. `/root` 에 쌓지 않는다.
- conda env 는 컨테이너 overlay(`/opt/conda/envs/vapasr`)에 있다. **컨테이너를 재생성하면 사라진다** —
  그때는 setup 스크립트를 다시 돌린다 (캐시된 가중치는 `/data3` 에 남는다).
- 공용 서버. GPU 점유 전 `./scripts/sync-rack4.sh status` 로 확인하고 `CUDA_VISIBLE_DEVICES` 를 지정한다.
