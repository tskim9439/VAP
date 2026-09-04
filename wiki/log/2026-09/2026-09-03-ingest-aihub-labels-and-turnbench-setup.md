## [2026-09-03] ingest | AI Hub 라벨 전체 통계 + TurnBench 재현 환경 준비

- Changed: `raw/sources/experiments/2026-09-03-aihub-71631-label-stats.json`(신규 raw), `experiments/aihub_label_stats.py`(신규),
  `experiments/verify_aihub_sample.py`(실제 스키마·천단위 구분자 반영), `scripts/aihub-download.sh`(사용 확인),
  `source-conversation-corpora`(라벨 통계 절), `source-turnbench`(코드 저장소 절), `task-verify-aihub-stereo-and-access`,
  `task-build-vap-target-pipeline`(채널 VAD 우선), `task-add-missing-baselines`(turnbench baseline 재사용),
  `task-reproduce-vap-turnbench-baseline` → doing. 서버: `/data3/tskim/corpora/aihub/71631/` 라벨 4개 해제,
  VS_02.실외 다운로드 중; `/data3/tskim/third_party/turnbench` 클론·설치.
- Reason: 사용자가 AI Hub API 키를 확보해(.env → .env.local 로 즉시 이동, 로컬·서버 모두) 서버 직접 다운로드가
  가능해졌다. 라벨 11,023 JSON 통계: **2,765 h**(Training 2,370 / Validation 396), 파일 중앙값 15 min, 3.3 M 발화,
  1.72 M 화자 교대, 10 ms 해상도. 그러나 교대의 50 % 가 '겹침'·겹침/gap 중앙값 ~1 s 로 **라벨 시간은 전사용 발화
  구간이지 VAD 경계가 아님** → VAP target 은 채널별 에너지 VAD 로 만들기로. 라벨상 48 kHz/2 ch 는 스펙(16 kHz)과
  달라 실물 확인 대기. 사용자 요청으로 VAP baseline 재현 착수: TurnBench HF 데이터셋 3개 식별(모두 gated),
  `SesameAILabs/turnbench` 에 scorer·sweep·baseline 20종(dualturn, wavlm causal, rms_vad …)·VAP `predictions-dev.json`
  동봉 확인. 리더보드 VAP 는 **oto fine-tune 체크포인트**(θ 0.9161/0.8591).
- Next: 사용자가 HF 약관 동의 + `HF_TOKEN` 을 `.env.local` 에 → dev/test/otoSpeech 다운로드 → 동봉 predictions-dev 재채점
  → `--pretrained` / `oto` 체크포인트로 직접 예측 재현. VS_02 도착 시 `verify_aihub_sample.py` 로 실물 검증.
- By: tskim
