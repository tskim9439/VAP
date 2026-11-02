## [2026-11-02] task | AI Hub 031 폴더 이름 변경 — replay 오디오 경로를 로더에서 찾기

- Changed:
  - 코드(커밋 a87d4af): `vapasr/data/archive.py`(MOVED·resolve_path, read_member 가 부름), `vapasr/data/streams.py`(load_utt_audio 가 부름), `tests/test_moved_paths.py`, 서버 점검 `raw/sources/experiments/2026-11-02-e13/moved031_check.{py,sh}`
  - 위키: [[output-e13-gate-trans-20261102]] 6 절, [[output-n3d-branch-20261029]] 6.8 절 E12 비교 주의, [[status]] 미해결 이슈
- Reason:
  - E13(99484) 학습 로그의 오디오 조립 실패가 모두 replay ahbc(AI Hub 031)였다. 공용 DB 가 폴더를 `031.broadcasting_koen` 으로 바꾸고 안쪽 이름을 NFD 로 다시 넣었다. 10-30 18:49 뒤 시작한 E12 다섯 판과 E13 이 ahbc 를 읽지 못했다.
  - 공용 폴더는 고치지 않고 읽는 쪽에서 찾는다. 서버 점검에서 압축 14 개 중 13 개를 찾아 학습 로더 경로로 읽었다. 없는 1 개(ahbc 줄의 3.3 %)는 새 폴더·trash 어디에도 없다.
- Next:
  - E13 은 선점 뒤 다시 시작할 때 반영된다. 반영 뒤 로그의 ahbc 실패가 그 압축 하나로 줄었는지 본다.
  - 031 다시 내려받기가 끝났는지 데이터 관리자에게 확인한다(사용자).
  - E13 이 쓰는 다른 데이터에 031 경로가 있는지 서버 grep(`tmp/e13-031/grep031.log`)이 끝나면 확인한다.
- By: tskim
