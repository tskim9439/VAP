## [2026-10-20] task | V2 학습 전 점검 통과·split v4(회의 공식 dev·eval held-out)·AMI 공식 평가 데이터(IHM-mix·SDM)

- Changed:
  - raw `2026-10-20-v2-multispeaker/`: `make_split_v4.py`, `ami_bench_build.py`, `ami_report.py`, `targets-meet-v1-summary.json`, README, `v2_eval_extra.sh`(12 단계 AMI·CODE 사본), `v2_check.sh`(v4)
  - `slurm/submit-stage3-train.sh`(`--split`), mxc 본 checkout 에 V2 코드 22 파일 반영(md5 대조, 서버 판이 모두 예전 커밋 판과 같았다)
  - [[output-v2-run-plan-20261020]], [[output-vibevoice-stream-comparison-20261020]](TAXI 세션 통째), raw [[2026-10-19-stage4-dialogue-translation]](m2 4–6 단계)
- Reason:
  - 사용자 "그래 그 비율로 진행해."
  - "AMI-IHM 과 AMI-SDM 으로도 평가 진행하려는 데, 학습 셋과 겹치지는 않지?" → "mxc 에 바로 받자."(이번 건만 예외)
- 결과:
  - 겹침 점검: AMI 공식 eval 16 은 held-out 이었다. 학습 쪽에 있던 AMI 공식 dev 18 과 **NOTSOFAR 원본 eval_set 129** 를 split v4 로 held-out 에 옮겼다. V1 은 회의를 학습하지 않았다.
  - AMI SDM: 공식 사이트는 mxc 에서 막혀(https SSL 오류·http 470) `diarizers-community/ami`(@8cdaae2, CC-BY-4.0)로 받았다. 2.03 GB, sha256 불일치 0.
    - `ami-bench-v1`(34 회의 × IHM-mix·SDM): 길이 차 1 s 이내, 시각 맞춤 −40–0 ms
  - 회의 번역 타깃: 영→독 98.7 %·영→한 96.5 % 행 유지(GPU 12.2 h)
  - V2 dry-run 12 셋 불일치 0, 스모크 20 step PARITY OK(슬롯 4·헤드 4 출력·탭 L00)
  - V1 m2a ck2000 TAXI 세션 통째 cpWER 26.4(40 s 로 나눔 28.1) → VibeVoice 19.2 와 같은 조건 차이 7.2 pt
- Next: V2 첫 학습 제출(사용자, `--run v2-p1`), V1 기준선 추가 평가(긴 창·회의·3–4 인 합성·AMI) 완료, 이어 돌리기 끝나면 m2 final 결과 마무리
- By: tskim
