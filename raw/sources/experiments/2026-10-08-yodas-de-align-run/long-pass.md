---
author_id: tskim
created: 2026-10-08
source_type: source-drop
---

# 30–40 s 처리(long pass) 전량 실행 (job 82624)

- 사용자 제출: `bash slurm/submit-yodas-de-align.sh --long --nodes 8 --partition hpc`
- 실행: 8 노드(GPU 64 장), 14:42:56 – 15:05:38(약 23 분), 선점·재시작 없음, shard 오류 0
- `align-summary-dur_long.json`: shard 2,899 개 전부 완료
  - 채택 23,086 발화 / 231.12 h — words train 22,990 행, dev 96 행
  - 원 전사와 불일치(WER > 3 %) 113,151 발화 / 1,142.98 h
  - 정렬 실패 2,017 발화 / 20.31 h — 토큰 시각 역전 1,934, 클립 끝 넘침 80–320 ms 77, 320 ms 초과 6
  - 토큰 상한 4
- 정렬 실패 비율(WER 통과 행 기준) 8.0 % — 본 처리(2.5 %)보다 높고, 대부분 토큰 시각 역전
- 표본 검증(무작위 long shard 40 개, 채택 행이 있는 24 개): 371 행, 80 ms 보정 64 행
  - 위반 0 건: tar 색인, 오디오 길이, 토큰 시각, tokenizer 재분할, decode
  - `SemCommitDataset` 로더 371 항목, bad 0(예: 39.5 s, K 502)
