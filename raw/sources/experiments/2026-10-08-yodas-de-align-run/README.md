---
author_id: tskim
created: 2026-10-08
source_type: source-drop
---

# YODAS-Granary 독일어 GPU 단계 전량 실행과 30–40 s 처리 시험 (mxc, 2026-10-08)

## 본 처리 (1–30 s, job 82542)
- 사용자 제출: `bash slurm/submit-yodas-de-align.sh --nodes 8 --partition hpc`
- 실행: 8 노드(GPU 64 장), 13:08:41 – 14:11:17(약 63 분), 선점·재시작 없음, shard 오류 0
- `align-summary-main.json`: shard 2,966 개 전부 완료
  - 채택(ALIGNED) 1,340,112 발화 / 2,045.42 h — words train 1,335,209 행, dev 4,903 행
  - 원 전사와 불일치(WER > 3 %) 2,029,526 발화 / 4,224.53 h
  - 정렬 실패 34,406 발화 / 68.7 h — 토큰 시각 역전 19,065, 클립 끝 넘침 80–320 ms 15,072, 320 ms 초과 269
  - 빈 전사 18, 토큰 상한 7
- 출력: mxc `VAPKT-data/data/yodas-de-v1/results/<shard>/`, 약 242 GB(`du -sb`)
- 표본 검증(무작위 shard 12 개, `../2026-10-08-yodas-de-align-smoke/validate_outputs.py` 의 표본판): 5,445 행, 80 ms 보정 989 행
  - 위반 0 건: tar 색인, 오디오 길이, 토큰 시각, tokenizer 재분할, decode
  - `SemCommitDataset` 로더로 5,445 항목 생성, bad 0

## 30–40 s 처리 시험 (long pass)
- 사전 점검에서 `dur_long` 만 걸린 행: 138,258 발화 / 1,394.5 h(train 137,773 / 1,389.6 h, dev 485 / 4.9 h), 모두 30–40 s
  - 다른 사유가 함께 걸린 30 s 초과 행 5,658 발화 / 57.4 h 는 계속 제외
- 전사 점검 표본에서 30 s 초과 67 발화의 채택 비율: 시간 기준 13 %(1–30 s 는 26 %). 표본이 작아 불확실
- `align-summary-smoke-long.json`: `--select-flags dur_long --max-dur 40`, shard 4 개, 후보 77 발화
  - 채택 14 발화 / 0.14 h(시간 기준 18 %), 정렬 실패 0
  - 같은 검증에서 위반 0, 로더 14 항목(예: 34.8 s, K 442)
- 시험 출력: mxc `VAPKT-data/results/yodas-de-align-smoke-long-20261008/`
