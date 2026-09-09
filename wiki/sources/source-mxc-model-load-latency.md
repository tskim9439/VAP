---
type: source
status: active
created: 2026-09-09
updated: 2026-09-09
summary: mxc 모델 로드 16 분의 원인(NFS 위 conda env 의 import 지연)과 해결(로컬 디스크 컨테이너 sa_tskim_fd, 노드별 env 스테이징, 인코더 캐시) — 960 s → 17 s
observed: 2026-09-09
---

# mxc 모델 로드 지연 진단·해결 (2026-09-09)

추론 데모가 `cuda True` 뒤에 멈춘 듯 보이고 학습 job 이 시작→학습까지 16–25 분 걸리던 문제.

## 진단(컨테이너 sa_tskim, `load_prof.py`)

| 단계 | cold | warm |
|---|---|---|
| import torch | 66 s | 12 s |
| import transformers | 197 s | 24 s |
| import vapasr.hf | 180 s | 52 s |
| thinker from_pretrained | 10 s | 3 s |
| NeMo restore_from(.nemo) | 506 s | 157 s |
| 합계 | 960 s | 248 s |

원인: conda env(8.7 GB)가 NFS(`/soundai`, vers=3 Azure) 위라 파이썬 import 가 파일마다 왕복. 가중치·`.nemo` 만 tmpfs 로 옮겨도 30–60 s 밖에 못 줄였다. 컨테이너 `/dev/shm` 은 noexec, 루트(overlay 2 TB)는 만석이라 실행 가능한 로컬 공간이 없었다.

## 해결

1. **새 컨테이너 `sa_tskim_fd`**: 호스트 `/tmp/sa_tskim`(md0 28 TB, xfs) → `/scratch`. `scripts/stage-env-local.sh` 로 env 복사(파일 단위 병렬 cp 18 분; 이후 `pack` 으로 tar 9.1 GB 를 NFS 에 두어 다른 곳은 1–2 분).
2. **컴퓨트 노드**: `/tmp` 가 같은 로컬 md0 → sbatch 3 종(학습·정렬·평가)이 시작 시 `srun` 으로 노드마다 tar 를 `/tmp/sa_tskim/vapasr-env` 에 풀고 그 python 을 쓴다(마커로 재복사 생략). `LOCAL_ENV=` 로 끔.
3. **인코더 캐시**: `.nemo` 복원은 로컬에서도 62 s(2.4 GB tar 해제 + 디코더·joint 구성). 인코더·전처리기 config+state 만 `VAPASR_STAGE_DIR` 에 캐시 → 5.8 s, 출력 차이 최대 1.9e-5.

| 구성 | import | 모델 로드 | 합계 |
|---|---|---|---|
| sa_tskim, NFS env, cold | 443 s | 516 s | 960 s |
| sa_tskim_fd, 로컬 env + 스테이징 + 인코더 캐시 | 4 s | 13 s | **17 s** |

부수 발견: 컨테이너(root)가 만든 manifest 디렉토리에 SLURM 사용자가 쓰지 못해 prep 의 카드 단계가 죽었다(67077) → 소유자 변경, `ds_cards.py` 경고로 격하. D2 첫 제출(67126)은 `aihub-bc-train` 이름이 언어 판정에서 `aihub` 로 잘려 KeyError — `lang_of` 를 최장 접두어 매칭으로.

관련: [[source-hf-trainer-migration]], [[output-dataset-schema-v1]]
