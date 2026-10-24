## [2026-10-25] task | 1.7B 회의 + 화자 슬롯 4 시험 준비

- Changed: `raw/sources/experiments/2026-10-21-v3-attribution/v3_check.sh`(INIT·TAG 덮어쓰기 — 기본은 예전 그대로), `raw/sources/experiments/2026-10-22-q17-chain/README.md`(시험 절), [[status]]
- Reason:
  - 사용자 "1.7B 회의 + 4 슬롯 시험을 내고 싶어".
  - 5 단계(대화 Stage 3)는 R1x 와 같으므로, 6 단계만 회의 + 4 슬롯으로 바꿔 R1x 6 단계(94814)와 나란히 낸다(`submit-q17-chain.sh --from 6 --dlg mt4 --eval`).
  - 1.7B 에서 처음 쓰는 경로(슬롯 4 늘리기, 화자 헤드 2→4, 회의 데이터)를 호스트에서 점검했다. dry-run·스모크 20 step·저장 재로드 대조 모두 통과했다.
- Next:
  - 사용자 제출 뒤 job 번호를 받아 학습(약 1–2 h)과 평가(`q17-mt4-v3`)를 지켜본다.
  - R1x(94814·94841)와 비교한다: dev 귀속, TAXI, 같은 언어 가드, AMI(4 슬롯 선언), 번역 BLEU.
- By: tskim
