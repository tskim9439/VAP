## [2026-10-21] task | V3 mxc 반영·학습 전 점검 통과, tel 셋 빌드, dev 준비 — R1 v3-p1·R2 v3-p1-nodata 제출 준비

- Changed: raw [[2026-10-21-v3-attribution]] "mxc 반영과 학습 전 점검" 절, `v3_check.sh`·`v3_dev_build.sh`·`v3_dev_decode.sh`, mxc 본 checkout(V3 25 파일, md5 대조), `VAPKT-data/data/sim-dialogue-xl-tel-v1`·`sim-dialogue-xl-tel-eval-v1`·`v3-dev/`(새)
- Reason: 사용자 "그래 V3 점검 이어서 진행해"
- 결과: mxc 시험 168 통과. tel 셋 10,000 세션·250 h(겹침 0·같은 화자 지속 0.049·레벨 차 중앙 4.7 dB·교대 간격 0.05–0.8 s), dev 160 세션. R1·R2 dry-run 불일치 0, 스모크 PARITY OK. 제출 스크립트 dry-run 으로 플래그 확인.
- Next: 사용자 제출(R1·R2) → 평가 큐(기존 6 단계 + 전화 dev·TAXI 턴 결정 지표). V1 dev 기준선·C3 격자 진행 중.
- By: tskim
