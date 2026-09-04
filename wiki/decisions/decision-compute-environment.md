---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-12-03
created: 2026-09-03
updated: 2026-09-03
summary: 학습은 rack4의 tskim_env 컨테이너에서 수행, 체크포인트는 /data4, 데이터는 /data3, 설정은 .env 단일 관리
sources:
  - [[output-streaming-vap-research-plan]]
---

# 결정: 실험 환경과 동기화 방식

## 맥락

[[output-streaming-vap-research-plan]] 의 실험은 GPU 서버에서 수행하고, 계획·위키는
로컬에서 유지한다. 두 곳이 어긋나면 실험 재현이 불가능해지므로 동기화 방식과
경로 규약을 고정해야 한다.

## 환경 (2026-09-03 실측)

| 항목 | 값 |
|------|-----|
| 서버 | `rack4` (hostname `rack4-a100`), SSH alias 사용 |
| GPU | **NVIDIA A100-PCIE-40GB × 4** (공용 서버 — 점유 전 확인 필요) |
| 컨테이너 | `tskim_env` (image `tskim_docker:v1`), running |
| 컨테이너 사용자 | **root** |
| 프로젝트 | `/home/tskim/VAP` |
| 체크포인트 | `/data4/tskim/VAPASR/{experiments,exports}` |
| 데이터 | `/data3/tskim/{corpora,features,manifests,logs}` |
| Python | 3.13.11 (`/opt/conda`) — **torch 미설치** |

## 결정

### 1. `.env` 단일 관리

모든 호스트·컨테이너·경로 설정은 저장소 루트 `.env` 한 곳에 둔다.
스크립트와 학습 코드 모두 여기서 읽는다. **`.env` 는 커밋한다**
(비밀정보가 없고, 팀원이 클론하면 바로 같은 규약을 쓰게 하기 위함).
머신별 차이는 `.env.local` 로 덮어쓴다 (gitignored).

### 2. 동기화는 rsync, 정본은 로컬

`scripts/sync-rack4.sh push` 로 로컬 → 원격 미러링한다 (`--delete`).
아직 Git 원격이 없어 `git clone` 방식을 쓸 수 없다. private 원격이 생기면
서버에서 clone + `scripts/pull-safe.sh` 방식으로 전환하는 편이 낫다.

**로컬이 정본이다.** 원격에서 생긴 산출물은 프로젝트 폴더가 아니라
`CKPT_ROOT` / `DATA_ROOT` 에 둔다. 그래야 `--delete` 가 안전하다.

### 3. 컨테이너 접근은 `docker exec`, SSH 직결 아님

`~/.ssh/config` 에 `rack4_tskim_env` (172.17.0.6) 항목이 있으나
**컨테이너 실제 IP 는 172.17.0.18 이라 접속되지 않는다.**
도커 IP 는 재시작마다 바뀌므로 IP 고정 방식은 쓰지 않는다.
`ssh rack4 docker exec tskim_env ...` 경로만 쓴다 (`sync-rack4.sh exec` / `shell`).

## 중요한 발견 — `/home` 은 bind mount 다

컨테이너는 `/data1`–`/data5` 와 `/home` 을 호스트에서 그대로 bind mount 한다.
`/home/tskim` 의 inode 가 호스트와 컨테이너에서 **동일함을 확인했다**
(`66308:50995245`).

따라서 **호스트로 rsync 하면 컨테이너가 즉시 같은 파일을 본다.**
컨테이너로 따로 복사(`docker cp`)할 필요가 없다. 동기화 경로가 하나로 단순해진다.

## 결과 / 파급 — 주의할 점

1. **`/data4` 가 97% 찼다.** 15T 중 약 **575G 여유** (2026-09-03).
   체크포인트를 여기 두라는 지시를 따르되, 대용량 중간 산출물
   (frozen encoder 특징 캐시 등)은 여유 있는 `/data3` (2.5T)로 보낸다.
   → [[task-compute-budget-and-feature-cache]] 에 반영됨.
   체크포인트 보존 정책(오래된 것 정리)을 조기에 정해야 한다.
2. **`/data3/tskim`, `/data4/tskim` 은 root 소유였다.** 컨테이너(root)에서
   하위 폴더를 만들고 `1019:1019` 로 chown 해 호스트 계정도 쓸 수 있게 했다.
   새 폴더를 만들 때 같은 처리가 필요할 수 있다.
3. **컨테이너에 torch 가 없고 rsync 도 없다.** 학습 시작 전 환경 구축이 필요하다.
   rsync 는 호스트에 있으므로 동기화에는 지장이 없다.
4. **공용 서버다.** 컨테이너가 80개 돌고 있고 A100 4장 중 일부는 이미 점유 상태였다.
   GPU 점유 전 `sync-rack4.sh status` 로 확인한다.

## 재검토

2026-12-03 — 또는 Git 원격이 생겨 clone 방식으로 전환할 때, 혹은 `/data4` 여유가
200G 아래로 떨어질 때.
