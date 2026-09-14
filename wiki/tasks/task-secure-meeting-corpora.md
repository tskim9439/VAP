---
type: task
status: doing
owner: tskim
due: 2026-09-19
priority: p0
created: 2026-09-11
updated: 2026-09-14
summary: 영어 회의·파티 코퍼스 5 종(AMI·ICSI·NOTSOFAR-1·DiPCo·CHiME-6) 확보 — 로컬 다운로드 → mxc 업로드 순환, 라이선스·용량·절차·진행 상태
sources:
  - '[[output-phase2-training-db]]'
  - '[[decision-multi-speaker-scope]]'
---

# 영어 회의 코퍼스 확보 (AMI · ICSI · NOTSOFAR-1 · DiPCo · CHiME-6)

## 배경
2026-09-11 사용자 결정: 다화자 자연 데이터로 다섯 코퍼스를 확보한다. 모두 화자별 근접(헤드셋/착용) 채널이 있어 A 등급([[output-phase2-training-db]] §6). 규칙: **서버에서 직접 받지 않고 로컬(맥)에 내려받아 올린다**. 로컬 여유 54 GB 라 "받기 → 올리기 → 지우기" 순환. 업로드는 `scripts/sync-mxc.sh bigpush`(azcopy SAS 미설정 → rsync 폴백, **실측 4 MB/s** → 190 GB ≈ 13 h). 서버 목적지 `/soundai/DB/raw/<이름>/`.

## 코퍼스별 사실 (2026-09-11 확인)
| 코퍼스 | 배포·라이선스 | 받을 것 | 용량 | 속도(맥, 실측) | 사용자 조치 |
|---|---|---|---|---|---|
| DiPCo | Zenodo 8122551, CDLA-Permissive 1.0 | `DipCo.tgz`(10 세션, 참가자 4 명 근접 마이크 + 어레이 5 대, 전사) | 12.4 GB | ≈0.5 MB/s → ≈7 h | 없음 |
| AMI | groups.inf.ed.ac.uk 미러, CC BY 4.0 | 회의별 `Headset-0..3.wav`(+3 회의는 Headset-4) + `Mix-Headset.wav`, 수동 어노테이션 `ami_public_manual_1.6.2.zip`(NXT, 단어 시각) | **54 GB 실측**(171 회의, wav 858 개) | ≈1–3 MB/s | 없음 |
| ICSI | 같은 사이트, CC BY 4.0 | 회의별 개별 헤드셋 SPH(≈356 MB/회의, 75 회의) + `ICSI_core_NXT.zip`·`ICSI_plus_NXT.zip`·`ICSI_original_transcripts.zip` | ≈27 GB | 직접 URL 확인: CGI(`estimate.cgi`)가 `https://groups.inf.ed.ac.uk/ami//ICSIsignals/SPH/<회의>/chan<N>.sph` 목록의 wget 스크립트를 생성 → curl 이어받기로 실행(`~/Downloads/vapkt-corpora/icsi/download.sh` 준비) | 없음 |
| NOTSOFAR-1 | Hugging Face `microsoft/NOTSOFAR`, CC BY 4.0 | 최신 버전만: `240825.1_train`(2,703 파일 29.5 GB), `240825.1_dev1`(1,390 파일 15.3 GB), `240825.1_eval_full_with_GT`(4,509 파일 49.1 GB). 근접 마이크 포함(논문). 구버전(240130–240501)은 제외 | **93.9 GB** (저장소 전체 301 GB) | HF snapshot_download(토큰은 `.env.local` HF_TOKEN) | 없음 |
| CHiME-6 | OpenSLR 150, CC BY-SA 4.0(2024 재배포, 상업 포함 무료) | `CHiME6_train/dev/eval.tar.gz` + `transcriptions` | 97 + 11 + 12 GB = 120 GB | OpenSLR ≈0.4–0.7 MB/s → ≈2 일 | 없음 |

## 순서와 상태 (2026-09-14 09:30)
1. [x] DiPCo — `/soundai/DB/raw/dipco/DipCo.tgz` 13.4 GB, 검증 완료(2026-09-13 01:29)
2. [x] AMI — `/soundai/DB/raw/ami/` 171 회의 wav 858 + 어노테이션, 54 GB, 검증 완료(03:34)
3. [x] ICSI — `/soundai/DB/raw/icsi/` 75 회의 SPH 922 + NXT zip 3, 34.2 GB, 검증 완료(08:30)
4. [ ] NOTSOFAR-1 — iCloud 복구로 `.env.local` 읽힘 → `scripts/notsofar-relay.sh` 로 09:29 시작(dev1 → train → eval_full 순, T5 스테이징)
5. [~] CHiME-6 — 소형 5 파일(dev 11.25 GB·eval 12.52 GB 포함) 업로드 검증 완료(15:14); train 97 GB 조각 릴레이 10 GB × 10 중 5 개 서버 완료, 6 번째 로컬 수신 완료·업로드는 mxc SSH 포트 복구 대기(`chime6-train` 서브커맨드, 40 회 재시도)
6. [ ] 완료 후 데이터 목록 §6.2 갱신(서버 위치·시간)

## 릴레이 자동화
`scripts/corpus-relay.sh run`(맥에서 nohup 백그라운드, 로그 `~/Downloads/vapkt-corpora/relay.log`): DiPCo·AMI 다운로드 완료 대기 → rsync 업로드 → 바이트 합계 검증 → **로컬만** 삭제 → ICSI 다운로드·업로드 → NOTSOFAR-1 서브셋 3 개(HF) → CHiME-6 소형 파일 → CHiME-6 train(97 GB, 로컬 디스크 초과)은 10 GB 범위 조각으로 받기→올리기→지우기를 반복한 뒤 서버에서 `cat` 으로 합침(`_parts_*` 보존). 서버 목적지 `/soundai/DB/raw/{dipco,ami,icsi,chime6,notsofar}`(사용자 지정 2026-09-11).

## 진행 기록
- 2026-09-14: iCloud 복구(사용자, 다른 네트워크). CHiME-6 eval 수신 시간 초과 → 아카이브 완전성(Content-Length) 검증 뒤 업로드하도록 `stage_chime6` 개정; part-0005 검증이 SSH 포트 폐쇄로 실패 → `chime6-train` 재실행. NOTSOFAR-1 별도 스크립트로 시작.
- 2026-09-13: 스테이징을 내장 디스크에서 **T5 SSD(`/Volumes/Samsung_T5/vapkt-corpora`)** 로 이전(사용자 지시). rsync `-e` 호스트명 버그·exFAT `._` 사이드카 검증 제외 수정.
- 2026-09-13: DiPCo·AMI 다운로드 완료 확인. 릴레이가 이틀간 멈춰 있었음 — 원인: DiPCo 재개 시 `curl -sS` 가 진행 줄 뒤에 개행 없이 `exit=0` 을 붙여 `^exit=` 대기 조건이 안 맞음. 마커를 별도 줄로 고쳐 재개. 로컬 여유 23 GB(AMI 54 + DiPCo 12.5 GB 보관 중)
- 2026-09-11: 릴레이 오케스트레이터 시작(DiPCo·AMI 다운로드 중). NOTSOFAR-1 은 HF 토큰 대기.
- 2026-09-11: 생성. 라이선스·용량·속도 확인, DiPCo 시작. azcopy SAS URL 이 `.env.local` 에 없어 rsync 업로드 예정 — SAS 를 받으면 azcopy 로 전환.
