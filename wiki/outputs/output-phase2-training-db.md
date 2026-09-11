---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 학습 DB 확정안(한국어/영어) — 같은 대화가 여러 형태(AI Hub 71631 = NIA24 134-1 = Phase 1 발화 crop)로 있는 중복을 제거한 뒤 역할별(자연 대화 A/B 등급, replay, 합성 원료, 평가 전용)로 정리. KO 자연 대화 확정 557 h(+청소년·실내 부분 조건부), EN 확정 ≈90 h(+Switchboard·AMI 조건부)
sources:
  - '[[output-phase2-db-survey]]'
  - '[[output-phase2-data-inventory]]'
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[decision-multi-speaker-scope]]'
  - '[[output-phase1-report]]'
---

# Phase 2 학습 DB 확정안 (한국어 / 영어, 중복 제거)

## 질문
조사한 DB([[output-phase2-db-survey]]) 가운데 Phase 2 에서 실제로 학습에 쓸 것을 언어별·역할별로 정한다. 같은 대화가 여러 사본·형태로 존재하므로 **한 대화는 한 형태로만** 쓴다.

## 요약
- **한국어 자연 대화(A 등급, 확정) 557 h**: AI Hub 감정 자유대화(성인) 실내 stereo 원본 757 대화 196.6 h + 실외 발화 조각 복원 1,492 대화 ≈360 h. 조건부로 실내 조각 부분 대화(결손 마스크)와 청소년 전체(검증 후)를 더한다. 다화자는 NIA23 회의 코퍼스(B 등급).
- **영어 자연 대화(A 등급, 확정) ≈90 h**: otoSpeech train. 조건부로 Switchboard 복원 230 h, AMI ~100 h, CANDOR.
- **중복 제거 규칙**: 성인 자유대화는 71631 stereo 원본·NIA24 134-1 조각·Phase 1 `aihub71631` 발화 crop 이 **모두 같은 녹음**이다 → 원본이 있는 대화는 원본만, 없는 대화는 조각만, Phase 1 crop 은 replay 에서 제외. 청소년(134-2 = 71632)은 성인과 다른 대화이므로 중복이 아니다. Switchboard 를 대화로 복원하면 Phase 1 replay 의 Switchboard 발화도 제외한다.

## 1. 동일 자원 대응표 (중복 제거의 근거)
| 같은 녹음 | 형태 1 | 형태 2 | 형태 3 | Phase 2 에서 쓰는 형태 |
|---|---|---|---|---|
| AI Hub 134-1 감정 자유대화 **성인** (데이터셋 번호 71631) | `/soundai/DB/raw/aihub/71631_audio`: stereo 원본 실내 757·실외 186 | `asr_db/korean_16kHz/NIA24/134-1_Emotion_Conv_Adult`: 화자 채널 기원 발화 조각(실내 8,270·실외 1,492 대화) | Phase 1 manifest `aihub71631-train/dev`(같은 원본의 발화 crop 159 h + 44 h) | 원본이 있는 763 대화(실내 563 + 실외 186 등)는 **원본**, 나머지는 **조각 복원**. Phase 1 crop 은 replay 에서 **제외** |
| AI Hub 134-2 감정 자유대화 **청소년** (71632) | 라벨만 `/soundai/DB/raw/aihub/71632` | `NIA24/134-2_Emotion_Conv_Youth` 조각(실내 7,821·실외 1,649 대화) | — | 조각 복원(검증 후). 성인과 **다른 대화**라 중복 아님 |
| AI Hub 464 주요 영역별 회의 음성 | 라벨만 `/soundai/DB/raw/aihub/464` | `NIA23/002_Meeting` 조각 + JSON | — | NIA23 사본 |
| Switchboard | Phase 1 replay(발화 단위 290 h) | 서버 사본 파일명 시각으로 2 채널 복원 가능 | — | 복원에 성공하면 **대화**로만 쓰고 replay 에서 제외 |
| NIKL 일상대화 | Phase 1 replay(발화 단위 1,451 h 표본) | 발화 시각으로 준자연 대화 복원 가능(겹침 없음) | — | Phase 2 는 **replay 로만**(준자연 복원은 후속 옵션) |

## 2. 한국어

| 역할 | 자원 | 형태·등급 | 규모 | 상태 | split |
|---|---|---|---|---|---|
| 자연 대화 (dyadic) | 71631 stereo 원본 실내 TS_01.실내_5 | stereo, A | 757 대화 **196.6 h** | 확정 | train (E2 노출 기록) |
| 자연 대화 (dyadic) | 134-1 조각 복원 실외 TS_02 | 2 채널 복원, A | 1,492 대화 **≈360 h** | 확정 | train / 일부 untouched test 후보 |
| 자연 대화 (dyadic) | 134-1 조각 복원 실내(원본 없는 대화) | 2 채널 복원, 발화 45 % 결손 → 마스크, A− | ≈7,700 대화(유효 발화 ≈40 %) | 조건부(Q0 마스크 규칙) | train, 전사·activity 만 |
| 자연 대화 (dyadic) | 134-2 청소년 조각 복원 | 2 채널 복원, A | 9,470 대화, 최대 수천 h | 조건부(완전성·채널 기원 검증) | train / dev / untouched test |
| 자연 대화 (다화자) | NIA23 002_Meeting | 혼합 mono 복원 + 화자·시각 라벨, B | 7,983 회의, 수백 h(집계 예정) | 조건부(Q0 집계) | train / dev |
| dev (dyadic) | 71631 VS_02.실외 stereo 원본 | A | 186 대화 51.7 h | 확정 | dev (E2 dev 재사용 명시) |
| untouched test | 134-1 실외 조각 복원 중 E2 미노출 대화 20–30 개 **또는** 134-2 실외 | A | 5–10 h | Q0 선정 | test |
| replay (단일 화자) | KsponSpeech | 발화 | 1,189 h | 확정 | Phase 1 split |
| replay | NIKL 일상대화 표본 | 발화 | 1,451 h | 확정 | Phase 1 split |
| replay | AI Hub 031/033 방송 | 발화 | 578 h | 확정 | Phase 1 split |
| ~~replay~~ | ~~Phase 1 aihub71631 crop~~ | — | ~~159 h~~ | **제외**(자연 대화와 동일 녹음) | — |
| 합성 원료 | KsponSpeech·NIKL train split | 발화 | — | 확정 | train 만 |

**한국어 합계**: 자연 대화 확정 **557 h**(A) + 조건부 실내 부분·청소년·회의; replay 3,218 h; dev 51.7 h.

## 3. 영어

| 역할 | 자원 | 형태·등급 | 규모 | 상태 | split |
|---|---|---|---|---|---|
| 자연 대화 (dyadic) | otoSpeech | 화자별 wav, A | 420 대화 104.9 h → train **≈90 h** | 확정(학습 사용 결정) | actor id 로 train / dev / untouched test |
| 자연 대화 (dyadic) | Switchboard 복원 | 파일명 시각으로 2 채널, A | ≈230 h | 조건부(Q0 시각 규약 검증) | train; 성공 시 replay 에서 제외 |
| 자연 대화 (dyadic) | CANDOR | 화자별 채널, A | 최대 850 h | 확보 시 | train / dev |
| 자연 대화 (다화자) | AMI | IHM 채널 + 원거리, A/B | ~100 h | 조건부(원본 회의 파일·시각 확보) | train / dev |
| 자연 대화 (다화자) | NOTSOFAR-1, DiPCo, CHiME-6, ICSI | 원거리 혼합 + 화자 라벨, B | ~24 / 5 / 50 / 72 h | 확보·라이선스 확인 후 | train / dev |
| 평가 전용 | TurnBench dev | dual-channel + gold | 38 대화 7.3 h | 확정 | test(EN turn) |
| ~~대화~~ | ~~CallHome~~ | 시각 없음, C | — | **제외** | — |
| replay (단일 화자) | LibriSpeech 960 | 발화 스트림 | 1,033 h | 확정 | Phase 1 split |
| replay | VoxPopuli | 발화 | 521 h | 확정 | Phase 1 split |
| replay | Granary-YODAS en129 | 발화 | 334 h | 확정 | Phase 1 split |
| replay | Switchboard 발화 | 발화 | 290 h | 복원 실패 시에만 replay 유지 | Phase 1 split |
| 합성 원료 | LibriSpeech train | 발화 | — | 확정 | train 만 |

**영어 합계**: 자연 대화 확정 **≈90 h**(A) + 조건부 Switchboard 230·AMI 100·CANDOR; replay 1,888 h(+Switchboard 290 h 조건부); 평가 7.3 h.

## 4. 중복 제거 규칙 (manifest 생성 시 강제)
1. 대화 ID(파일 stem)를 키로 **모든 자원에서 한 번만** 등장하게 한다. 71631 원본 stem 은 134-1 조각 목록·Phase 1 `aihub71631` manifest 에서 제거한다.
2. 화자 ID 로 split 을 나눈다(otoSpeech actor id, 71631/71632 Speaker ID). 같은 화자가 train 과 test 에 걸치지 않게 한다.
3. Phase 1 replay 의 Switchboard 는 대화 복원이 성공하면 replay 에서 뺀다. NIKL 은 Phase 2 동안 replay 로만 쓴다(준자연 복원과 동시 사용 금지).
4. 합성 원료는 train split 의 발화만 쓰고, 합성 대화의 화자가 dev/test 화자와 겹치지 않게 한다.
5. E2 노출(71631 실내 757·VS_02·Phase 1 replay 전부)을 레코드에 기록하고, untouched test 는 E2 미노출 대화에서만 뽑는다.

## 5. 우선 작업 (Q0)
1. 134-1 실외 1,492 대화 2 채널 복원 + 채널별 정렬 → 첫 Phase 2 manifest.
2. 71631 원본과 134-1 조각의 stem 대조표 작성(중복 제거 1 번 규칙).
3. 134-2 청소년 완전성·채널 기원 검증(§3 의 상호상관 방법).
4. Switchboard 파일명 시각 전수 검증 → 성공 시 복원.
5. 002_Meeting 시각 합산 집계와 혼합 복원 파일럿.
6. otoSpeech actor id 기반 split 고정, 라이선스 원문 확인.

## 불확실성
- 134-1 실외 ≈360 h 는 라벨 통계(Training 2,370 h / 9,799 대화) 비례 추정이며 실제는 발화 시각 합산으로 확정한다.
- 청소년 134-2 는 성인과 같은 방식으로 잘렸을 것으로 추정하나 검증 전이다.
- 회의 코퍼스의 겹침 표기 유무·총 시간 미확인.

## 근거
- [[output-phase2-db-survey]] — 조사 결과·검증 실험
- [[output-phase2-data-inventory]] — 이전 목록·시간 합계
- [[output-phase1-report]] §3 — Phase 1 replay 8 코퍼스
- [[decision-multi-speaker-scope]] — 다화자 포함
- [[output-phase2-streaming-asr-diarization-plan]] §5 — 역할·split 원칙
