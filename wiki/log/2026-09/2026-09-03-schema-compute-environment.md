## [2026-09-03] schema | rack4 실험 환경 구성과 .env 단일 관리

- Changed: `.env`(신규, 커밋), `.rsyncignore`(신규), `.gitignore`(`.env.local` 추가),
  `scripts/sync-rack4.sh`(신규), `wiki/decisions/decision-compute-environment.md`(신규),
  `wiki/status.md` 갱신. 원격에 `/home/tskim/VAP`, `/data4/tskim/VAPASR/{experiments,exports}`,
  `/data3/tskim/{corpora,features,manifests,logs}` 생성.
- Reason: 사용자가 학습 서버(rack4), 컨테이너(tskim_env), 프로젝트 경로(/home/tskim),
  체크포인트(/data4/tskim/VAPASR), 데이터(/data3/tskim) 규약을 지정하고 하나의 파일로
  관리할 것을 요청했다. 설정 파일을 쓰기 전에 실제 접속·경로를 검증했고 네 가지를 발견했다:
  (1) `/home` 이 컨테이너에 bind mount 되어 호스트와 inode 가 동일 —
  호스트로 rsync 하면 컨테이너가 즉시 같은 파일을 보므로 별도 복사가 불필요하다.
  (2) `~/.ssh/config` 의 `rack4_tskim_env` 가 172.17.0.6 을 가리키나 실제 IP 는
  172.17.0.18 이라 접속 불가 — 도커 IP 는 재시작마다 바뀌므로 `docker exec` 경로로 고정했다.
  (3) `/data3/tskim`·`/data4/tskim` 이 root 소유라 호스트 계정으로 쓸 수 없어
  컨테이너(root)에서 하위 폴더를 만들고 1019:1019 로 chown 했다.
  (4) **`/data4` 가 97% 사용 중(575G 여유)** 이고 컨테이너에 torch 가 없다.
  macOS rsync 2.6.9 가 `--info=` 를 지원하지 않아 버전 감지 폴백을 넣었고,
  bash 3.2 의 빈 배열 확장 문제도 수정했다. 첫 push 로 112개 파일 동기화를 확인했다.
- Next: `/data4` 체크포인트 보존 정책 수립, 컨테이너 torch 등 학습 환경 구축,
  `AGENTS.md` Directory Contract 에 `.env`·`sync-rack4.sh` 반영하는 유지보수 PR,
  Git private 원격 생성 후 rsync → clone 방식 전환 검토.
- By: tskim
