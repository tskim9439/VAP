## [2026-10-18] task | 노드 로컬 env 복사본 무결성 점검 — p11 job 91438 이 /tmp 자동 정리로 pyexpat 이 빠진 복사본을 써서 실패

- Changed: `scripts/stage-env-local.sh` — "이미 복사됨" 판정에 파일 수 대조(`.staged-count`)를 더했다. 다르거나 기록이 없으면 tar 를 다시 푼다. `pack` 은 새 마커를 tar 에서 뺀다.
- Reason: job 91438(`low_p_hpc`, hpc-2·3, `--exclude hpc-102`)이 시작 4 분 만에 rc=1 로 실패하고 requeue 됐다.
- 결과:
  - hpc-3 의 rank 8–15 가 `ModuleNotFoundError: No module named 'pyexpat'` 로 죽었다. 그 노드는 예전 job 의 `/tmp/sa_tskim/vapasr-env` 를 마커만 보고 "이미 복사됨" 으로 썼다.
  - 원본(NFS) env 에는 `pyexpat` 이 있다. 오래 안 쓴 파일을 `/tmp` 자동 정리(systemd-tmpfiles)가 지운 것으로 본다(노드 안은 직접 보지 못했다).
  - 가짜 env 로 mxc 호스트에서 4 경우(처음·다시·파일 빠짐·기록 없음)를 시험했다. 서버에서는 지우지 않고 시험 폴더 안에서 옮겨 흉내 냈다.
  - 기존 복사본은 기록이 없어 노드마다 한 번 다시 푼다(24–35 s).
- Next: 91438 의 다음 시도를 감시한다. `scripts/` 변경이라 병합 때 다른 팀원 리뷰가 필요하다(AGENTS.md).
- By: tskim
