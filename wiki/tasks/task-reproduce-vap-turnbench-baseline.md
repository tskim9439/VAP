---
type: task
status: done
owner: tskim
due: 2026-10-01
priority: p0
created: 2026-09-03
updated: 2026-09-03
summary: 원 VAP를 TurnBench 프로토콜로 재현해 0.845/0.055/368ms 기준선 확보
sources:
  - [[output-streaming-vap-research-plan]]
---

# VAP baseline 재현

## 배경

[[source-turnbench]] 의 VAP 수치(EOT recall 0.845 @ FPR 0.055, p50 368 ms;
INT 0.945 @ 0.107, 994 ms)를 **직접 재현하지 못하면 이후 모든 개선 주장이 공중에 뜬다.**

## 완료 조건

- [ ] **[사용자]** HF 계정으로 3개 데이터셋 약관 동의 (gated: auto — 즉시 승인): `otoearth/otoSpeech-full-duplex-turn-104h`, `mundo-ai/turn-benchmark-dev`, `mundo-ai/turn-benchmark-test`
- [ ] **[사용자]** HF 토큰(read) 발급 → rack4 `/home/tskim/VAP/.env.local` 에 `HF_TOKEN=...`
- [x] dev(4.2 GB) 수신 → HF 캐시. test(13 GB)·otoSpeech(**290 GB**) 는 `/data3/tskim/corpora/turnbench/` 로 다운로드 중 (`sync-rack4.sh jobs`)
- [x] `SesameAILabs/turnbench` 클론·설치, scorer 동작 확인. 리더보드 VAP = **oto fine-tune 체크포인트** (θ_eot 0.9161 / θ_int 0.8591). `predictions-dev.json` 동봉 → 데이터 확보 즉시 재채점
- [x] 원 VAP 구현체 확보 — `/data3/tskim/third_party/VoiceActivityProjection` (환경 태스크에서 로드 확인)
- [x] [[turn-taking-evaluation-protocol]] — 공식 scorer 를 그대로 사용 (자체 구현 불필요)
- [x] **동봉 `predictions-dev.json`(oto ckpt) 재채점 — dev**: EOT recall **0.841** / fp **0.045** / lat p10/50/90 −34/**463**/1657 ms (tp 1601 fn 303 fp 48 tn 1015);
      INT recall **0.957** / fp 0.100 / 257/896/2002 ms. 리더보드(test) 0.845/0.055/368 과 같은 자릿수 — 스코어러·데이터 파이프라인 확인
- [x] `--pretrained`(사전학습 원본, fine-tune 없음) 직접 예측 — dev: EOT recall **0.793** / fp 0.094 / lat −89/**613**/2000 ms; INT 0.865 / 0.099 / 928 ms (sweep θ_eot 0.905, θ_int 0.91). oto 대비 recall −0.05, latency +150 ms → **otoSpeech fine-tune 효과가 크다**
- [x] `oto` 체크포인트 직접 예측 → **동봉 파일과 임계값·점수 완전 일치** (θ 0.91615/0.85913, EOT 0.841/0.045/463) → [[output-vap-turnbench-baseline-reproduction]]
- [ ] test 셋 예측 생성 (라벨 없음; 리더보드 제출용) — 추후
- [x] 재현 성공 — 원인 문서화 불필요. 리더보드 수치는 test split 임을 명시
- [x] 평가 코드 = `turnbench` 패키지(`vapasr` env 설치) + `predictions.json` 규약. backbone 독립

## 진행 기록

- 2026-09-03: 생성.
- 2026-09-03: **완료.** oto 직접 예측이 동봉과 동일. 사전학습 원본은 0.793/0.094/613. 결과 raw: `2026-09-03-vap-turnbench-repro/`.
- 2026-09-03: HF 토큰 확보 → dev 수신, 동봉 예측 재채점 성공 (위 수치). `experiments/reproduce_vap_turnbench.sh` 로 절차 고정.
  장시간 작업은 `sync-rack4.sh bg` (nohup/setsid) 로 세션과 분리. 심볼릭 링크 깊이 오류로 1회 실패 후 재실행 중.
- 2026-09-03: 환경 태스크에서 원 VAP 저장소·동봉 체크포인트 로드 확인
  (`/data3/tskim/third_party/VoiceActivityProjection`, `scripts/smoke-test-models.py vap`).
  `strict=False` 로드 시 missing 8 키 — 재현 전 어떤 키인지 확인할 것.
