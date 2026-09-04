## [2026-09-03] task | AI Hub 실물 검증 완료 — 진짜 분리 stereo

- Changed: `task-verify-aihub-stereo-and-access` → done, `raw/sources/experiments/2026-09-03-aihub-71631-vs02-verify.json`(신규 raw),
  `experiments/aihub_extract_and_verify.sh`(신규), `experiments/verify_aihub_sample.py`(무작위 표본·JSON 인덱스·`_f` 파서),
  `source-conversation-corpora`(실물 검증 표), `todo.md`·`TODO.md`·`status.md`·`index.md`. 서버: VS_02.실외 186 wav 해제.
  운영: 오늘 GPU 배정 1번 → `.env.local` `GPU_DEFAULT=1`, `activate-env.sh` 가 `CUDA_VISIBLE_DEVICES` 기본값으로 사용.
  장시간 작업용 `sync-rack4.sh bg/jobs` 추가.
- Reason: VS_02.실외(6.4 GB) 수신 후 무작위 100 wav 검증. **16 kHz / 2 ch / PCM_16** (라벨의 48000 은 원본 표기),
  채널 누설 중앙값 **−64 dB**, 상관 6e-5 — 진짜 분리. 에너지 VAD overlap 비율 중앙값 7.4 % 로 실제 대화 역학 존재.
  라벨 StartTime vs VAD 온셋 오차 중앙값 **30 ms**(p90 370) — 라벨 통계의 '겹침 50 %' 는 발화 끝이 넉넉한 탓.
  VAP target 은 채널 VAD 로, 라벨은 화자·텍스트·대략적 온셋으로 쓴다. 첫 검증 실행에서 `_f` 미정의로 JSON 항목이
  null 이 나온 편집 실수를 고쳐 재실행했다.
- Next: 이용약관 원문에서 어노테이션 파생물 공개 가능 여부 확인(사용자). 2차 TS_01.실내_5(≈95 h) 수신 →
  [[task-build-vap-target-pipeline]]. VAP baseline 재현 진행 중(GPU 1).
- By: tskim
