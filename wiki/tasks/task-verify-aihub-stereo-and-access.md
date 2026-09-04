---
type: task
status: done
owner: tskim
due: 2026-09-10
priority: p0
created: 2026-09-03
updated: 2026-09-03
summary: AI Hub 감정 태깅 자유대화 신청·승인 및 실물 stereo·타임스탬프 품질 확인
sources:
  - [[output-streaming-vap-research-plan]]
---

# AI Hub 데이터 접근과 실물 검증

## 배경

[[source-conversation-corpora]] 스펙 페이지상 stereo(화자별 mono → stereo 병합),
16 kHz/16-bit, 발화 타임스탬프 보유로 확인했다. 그러나 **스펙 문서와 실물은 다를 수 있다.**
이 데이터가 [[voice-activity-projection]] 학습의 유일한 한국어 자원이므로
여기서 막히면 한국어 전체가 막힌다.

## 완료 조건

- [ ] **[사용자]** AI Hub 계정 생성 + 내국인 본인인증 (aihub.or.kr)
- [ ] **[사용자]** 데이터 활용 신청 — 성인 **71631**, 청소년 **71632** (활용 목적: 학술 연구 / turn-taking 모델 학습). 승인 대기
- [x] 1차: 라벨 4개 + VS_02.실외.zip — 서버에서 aihubshell 로 직접 수신 (API 키를 .env.local 에 보관)
- [x] 2차: **TS_01.실내_5.zip** 수신·해제 완료 — **757 wav** (`/data3/tskim/corpora/aihub/adult-ts01-5-wav`). manifest 빌드 완료 → `aihub-ts01-5`: **196.6 h**(zip 추정의 2×), 757 파일, 플래그 46, 채널 뒤바뀜 26
      (API 키 발급 없음, PC 다운로드만 가능 — 2026-09-03 확인. 맥 여유 82 GB 이므로 한 번에 50 GB 이하로)
- [ ] `./scripts/aihub-upload.sh adult ~/Downloads/<파일> --move` 로 rack4 전송 (≈15 MB/s, 재개 가능, 전송 후 로컬 삭제)
- [ ] `ssh rack4 'cd /home/tskim/VAP && scripts/aihub-upload.sh extract adult'` 로 분할 병합·압축 해제
- [x] `experiments/verify_aihub_sample.py` — VS_02.실외 186 wav 중 무작위 100개 검증 (결과 `raw/sources/experiments/2026-09-03-aihub-71631-vs02-verify.json`)
- [x] **두 채널의 상관·누설 측정** — 누설 중앙값 **−64 dB** (p90 −35 dB), 채널 상관 **6e-5** → 진짜 분리 stereo
- [x] 발화 타임스탬프 — 소수 2자리(10 ms). 실물: **16 kHz / 2 ch / PCM_16** (라벨의 48000 은 원본 녹음 표기)
- [x] 라벨 StartTime vs 에너지 VAD 온셋 오차 — **중앙값 30 ms**, 평균 187, p90 370 ms (n=18,334). 시작은 정확, 꼬리 길다. 발화 끝(EndTime)은 넉넉해 VAD 경계로 부적합 → 파이프라인은 채널 VAD
- [ ] 이용약관 원문에서 **어노테이션 파생물 공개 가능 여부** 확인 → [[question-korean-corpus-licensing]]

## AI Hub 파일 목록 (성인 71631, 2026-09-03 사용자 제공)

| 파일 | 크기 | key | 추정 시간(stereo) | 용도 |
|------|-----:|-----|------:|------|
| TS_01.실내_1~4.zip | 47–51 GB ×4 | 507715–507718 | ≈ 205–220 h 씩 | 본 학습 (후순위) |
| **TS_01.실내_5.zip** | 21.77 GB | 507719 | **≈ 95 h** | **Stage 1 학습 서브셋** |
| TS_02.실외.zip | 37.14 GB | 507720 | ≈ 160 h | 후순위 (실외 잡음) |
| **TL_01.실내.zip / TL_02.실외.zip** | 96.6 MB / 17.3 MB | 507721 / 507722 | — | **라벨 전체 — 즉시** |
| VS_01.실내.zip | 38.33 GB | 507723 | ≈ 165 h | 평가셋 (후순위) |
| **VS_02.실외.zip** | 6.37 GB | 507724 | ≈ 28 h | **실물 검증용 샘플 — 즉시** |
| **VL_01.실내.zip / VL_02.실외.zip** | 16.6 MB / 2.7 MB | 507725 / 507726 | — | **라벨 전체 — 즉시** |

추정: 16 kHz·16 bit·stereo = 230 MB/h. 전체 ≈ 1,300 h stereo 대화 (AI Hub 의 3,000 h 는 화자별 mono 합산 추정).
전송 15 MB/s 기준 TS_01.실내_5 ≈ 24 분, VS_02.실외 ≈ 7 분.

## 진행 기록

- 2026-09-03: 생성. 스펙 페이지 기준 stereo 확인, 실물 미확인.
- 2026-09-03: **실물 검증 완료 → 태스크 done.** VS_02.실외(6.4 GB, 186 wav) 해제. 16 kHz/2ch/PCM_16, 누설 −64 dB, 상관 ~0,
  에너지 VAD overlap 비율 중앙값 7.4 %(실제 대화 역학), 라벨 온셋 오차 중앙값 30 ms. 남은 항목은 이용약관 원문 확인(사용자)과 2차 다운로드.
- 2026-09-03: **API 키 확보 → 서버 직접 다운로드 성공.** 라벨 4개(133 MB) 수신·해제, 11,023 JSON 통계 완료
  (2,765 h, 파일 중앙값 15 min, 48 kHz/2ch 표기, 타임스탬프 10 ms 해상도·천단위 구분자). VS_02.실외(6.37 GB) 다운로드 중.
  라벨 시간이 넉넉한 발화 구간이라 VAD 경계로는 부적합 — 채널 VAD 필요.
- 2026-09-03: 사용자 확인 — 마이페이지에 API 발급 없음, 'PC 에서만 다운로드 가능'. aihubshell 경로 폐기,
  맥 브라우저 다운로드 → `scripts/aihub-upload.sh` 전송(--move 순환) → 서버 extract 로 절차 변경. 맥→rack4 실측 15 MB/s.
- 2026-09-03: 성인 71631 / 청소년 71632 ID 확정. 서버 직접 다운로드용 `scripts/aihub-download.sh`(aihubshell 래퍼)와
  실물 검증 `experiments/verify_aihub_sample.py` 준비. 계정·본인인증·신청·API 키는 사용자 직접 수행 필요.
