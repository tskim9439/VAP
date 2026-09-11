---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2(정본 계획 §5.1) 사용 데이터 목록 — mxc 실측(2026-09-11) 기준. 자연 대화 KO 71631 stereo 248 h(E2 노출 196.6 h + dev 51.7 h)·EN otoSpeech 104.9 h(미노출, 화자 ID 있음), 평가 전용 TurnBench dev, 합성 원료 LibriSpeech·Kspon·NIKL, replay Phase 1 5.6 k h, 확장 후보 Switchboard(파일명 시각으로 복원 가능)·CallHome(시각 없음)·CANDOR(미확보)·영어 회의 코퍼스 AMI/ICSI/CHiME-6/NOTSOFAR-1/DiPCo(3 명 이상, 2 명 활동 창 선택으로 활용)
sources:
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[source-conversation-corpora]]'
  - '[[output-vap-target-pipeline]]'
  - '[[output-phase1-report]]'
  - '[[source-turnbench]]'
---

# Phase 2 사용 데이터 목록 (mxc 실측 2026-09-11)

정본 계획 [[output-phase2-streaming-asr-diarization-plan]] §5.1 의 표를 서버 실측으로 채운 것이다. 원본 측정 로그: `raw/sources/experiments/2026-09-11-phase2-data-inventory/mxc-inventory.txt`. 시간은 파일 크기 환산(16 kHz·16-bit PCM)이며 무음 포함 대화 경과시간이다. 정본 §5.1 의 원칙대로 두 채널 시간을 더하지 않았다(otoSpeech 화자별 wav 합계 209.9 h → 대화 104.9 h).

## 1. 이번 단계에 쓰는 데이터

| # | 자원 | 위치(mxc) | 실측 | 형식 | 역할(정본 단계) | E2 노출 | split 계획 |
|---|---|---|---|---|---|---|---|
| 1 | **AI Hub 71631 자유대화 TS_01.실내_5** | `/soundai/DB/raw/aihub/71631_audio/Training/01.원천데이터/TS_01.실내_5` | 757 대화, 45.3 GB, **196.6 h** | stereo 16 kHz(화자=채널), JSON 발화 전사+시각(TL_01.실내 8,306 개 중 757 개에 오디오) | KO 자연 대화·겹침 전사(Q1–Q3), turn 라벨 | **노출**(E2 `aihub71631-train` 의 발화 crop) | Phase 2 **train** 만. 정본 §5.2 대로 "pretraining 노출" 로 기록 |
| 2 | **AI Hub 71631 VS_02.실외** | `…/Validation/01.원천데이터/VS_02.실외` | 186 대화, 11.9 GB, **51.7 h** | 위와 동일(VL_02.실외 186 JSON) | KO **dev / 보고**(Q0 고정 pack) | E2 dev(`aihub71631-dev`) | dev. 정본대로 untouched test 라 부르지 않음 |
| 3 | **otoSpeech 16 k** | `/soundai/DB/raw/otoSpeech16k/<1..420>/` | 420 대화, 24.2 GB, **104.9 h** | 폴더당 `speaker_{1,2}_audio.wav`(mono 16 k) + `speaker_{1,2}_annotation_a.srt`(turn 유형 태그 포함, 예 `[Normal Turn]`) + `metadata.json`(actor id·성별·대화 유형) | EN 자연 대화(Q1–Q3), turn 라벨 | **미노출** | actor id 로 **화자 분리 split**: train / dev / untouched test (대화 단위, Q0 에서 고정). 라이선스 non-commercial·voice-cloning 금지 — 학습 사용 결정(2026-09-11), 원문 재확인은 Q0 |
| 4 | **TurnBench dev** | `/soundai/DB/raw/turnbench/dev/data` (parquet 3 shard) | 38 대화, 7.3 h | dual-channel + gold EOT/INT | **EN 평가 전용**(Q3, 공식 scorer) | 미노출 | 평가만. test(116 대화)는 오디오만 있어 제출용 |
| 5 | **Phase 1 단일 화자 데이터** | Phase 1 manifest 8 종 | EN 2.2 k h + KO 3.4 k h(≈5.6 k h) | E2 와 같은 TN·split | 단일 화자 ASR **replay**(모든 Q), turn 손실 마스크 | 노출(당연) | Phase 1 train split 그대로, dev/test 는 가드레일 평가에 |
| 6 | **합성 겹침 원료** | LibriSpeech train, KsponSpeech train, NIKL(발화 단위) | 1,033 / 1,189 / ≈3,800 h | 단일 화자 발화 | 서로 다른 화자 혼합(overlap·음량차 제어, Q2) — 정본 §5.1 조건 7 종 | 노출(train split) | **train split 원료만** 사용(정본 규칙) |


## 1b. 학습에 쓰는 총 시간 (대화 경과시간 기준, 평가 전용 제외)

| 구분 | 자원 | 시간 | 비고 |
|---|---|---|---|
| **자연 2 화자 대화(확정)** | 71631 TS_01.실내_5 | 196.6 h | KO, E2 노출 |
| | otoSpeech train | ≈ 90–95 h | 104.9 h 중 dev·untouched test(대화 단위 ≈ 10 %)를 뺀 값. 정확한 split 은 Q0 |
| | **소계** | **≈ 290 h** | KO 2 : EN 1 |
| 자연 2 화자 대화(조건부) | Switchboard 복원 | + ≈ 230 h | 파일명 시각 규약 검증 통과 시(train split) |
| | CANDOR | + 최대 850 h | 확보 시 |
| 합성 2 화자(생성) | LibriSpeech·Kspon·NIKL train 원료 | 설계값 | 정본 §7.1: 대화 배치 안 합성 비율은 "최대 약 절반부터". 자연 290 h 와 같은 규모(≈ 300 h)를 출발점으로 두면 대화 합계 ≈ 600 h |
| 단일 화자 replay(풀) | Phase 1 8 코퍼스 | 5,555 h(EN 2,178 + KO 3,377) | 전량을 쓰지 않고 배치의 30 %(오디오 초 기준)로 섞음. 71631 159 h 는 위 196.6 h 와 같은 오디오의 발화 crop |

- 한 epoch 의 노출량(정본 §7.1 의 대화 70 % : replay 30 %): 자연 290 h + 합성 ≈ 300 h = 대화 ≈ 590 h → replay ≈ 250 h → **≈ 840 h / epoch**. Phase 1 (5,555 h/epoch) 의 15 % 수준이라 Q1/Q2 run 은 E2 처리량(1.02 s/step, 32 GPU) 기준 epoch 당 1 시간 이내다.
- Switchboard 가 들어오면 자연 대화 ≈ 520 h, epoch ≈ 1.2 k h.
- **평가 전용(학습 제외)**: 71631 VS_02 51.7 h, otoSpeech dev/test ≈ 10–15 h, TurnBench dev 7.3 h, Phase 1 dev/test.

## 2. 1 차 관문 이후 확장 후보 (정본 §5.1 "Switchboard/CANDOR 등")

| 자원 | 실측 | 판정 |
|---|---|---|
| Switchboard(서버 사본) | `…/EN/TRAIN/OPEN/switchboard/{TRAIN,VALIDATION,TEST}` 발화 단위 wav, 파일명 `sw03322A_18271375_191563125.wav` = 대화·채널(A/B)·시작 182.71375 s·끝 191.563125 s(duration 8.87 s 와 일치) | **대화 시간축 복원 가능**(화자별 발화를 원 시각에 배치 → 2 채널, 겹침 보존, 발화 사이 무음은 원음 없음). Q0 에서 파일명 규약을 전수 검증하면 EN 자연 대화 230 h 추가 후보 |
| CallHome EN | `…/callhome/eng/chunked/<call>/0638_0000_A.wav`, 8 kHz, 176 통화 19.9 h, 파일명에 **시각 없음**(순번·화자만) | 시간축 복원 불가 → Phase 2 오디오로는 제외. 텍스트·화자 순서만 활용 가능 |
| CANDOR | 서버에 없음 | 확보 시 사용(사용자 결정 2026-09-11). 신청·로컬 경유 업로드 필요 → [[task-secure-english-corpora]] |
| AI Hub 71631 미반입분 | 라벨만 있음: TL_01.실내 8,306(오디오 757), TL_02.실외 1,493, VL_01.실내 1,038 | **untouched KO test** 를 만들려면 미반입 대화에서 20–30 개를 새로 반입해야 한다(PC 브라우저 경유). 사용자 결정 대기 |
| NIKL 준자연 대화 | 발화 단위 PCM + 시각·화자 ID, 겹침 발화는 quarantine | 발화를 시각대로 배치하면 겹침 없는 대화가 된다. 정본 범위 밖이므로 Q1 결과 후 검토 |


### 2b. 영어 회의 코퍼스 (사용자 추가 후보, 2026-09-11)

**2026-09-11 사용자 결정으로 "최대 2 화자" 제약을 제거**했다([[decision-multi-speaker-scope]]). 따라서 이 코퍼스들은 창 선택 없이 **다화자 자연 데이터**로 쓴다: 원거리 마이크(SDM/어레이) 또는 헤드셋 전 채널 합산을 mono 입력으로, 헤드셋 채널별 VAD·전사를 화자별 라벨로. 실제 겹침·원거리 잡음·다자 turn 역학이 있는 자연 데이터다. 시간·라이선스는 사용자 제공값과 기억에 의존하므로 Q0 에서 원문을 확인한다.

| 코퍼스 | 규모(대략) | 화자 수 | 채널 | 라이선스·확보 | 서버 | 활용 |
|---|---|---|---|---|---|---|
| AMI | ~100 h | 회의당 4 | IHM(헤드셋) + SDM/MDM(원거리) | CC BY 4.0, 공개 | **있음** `english_16kHz/ami`(발화 단위 wav ihm/sdm + CSV·JSONL `spk_id, meeting_id, microphone`; 시각은 원본 어노테이션 필요) | 서버 사본은 발화 단위라 원본 회의 파일·단어 시각을 추가 확보해야 시간축 복원. 1 순위 |
| ICSI | ~72 h | 3–10 | 헤드셋 + 테이블 마이크 | LDC(LDC2004S02) 배포, 연구용 계약 | 없음 | 화자 수 최대 10 — 슬롯 상한 설계에 영향 |
| CHiME-6 | ~50 h | 4 | 참가자 binaural + 6 개 Kinect 어레이 | CHiME 계약(연구용 무료) | 없음 | 원거리 mono 로 잡음·잔향 강건성. 전사 라벨은 있으나 정렬 품질 낮음(대화 파티) |
| NOTSOFAR-1 | ~24 h | 4–8 | 다중 어레이 + 헤드셋 | CC BY 4.0(확인 필요), 공개 | 없음 | 회의 도메인 원거리 |
| DiPCo | ~5 h | 4 | 헤드셋 + 어레이 | CDLA-Permissive(확인 필요), 공개 | 없음 | 소량, 평가·검증용 |

- 우선순위: **AMI**(서버에 있고 라이선스 자유) → NOTSOFAR-1·DiPCo(공개 다운로드, 로컬 경유 업로드) → CHiME-6(계약) → ICSI(LDC 비용).
- 기대 규모: 전량 사용 시 약 250 h(확보 가능한 AMI·NOTSOFAR-1·DiPCo 만이면 약 130 h). 자연 dyadic 대화 290 h 와 합쳐 자연 다화자 데이터 ≈ 420–540 h.
- 주의: 회의는 dyadic 대화와 turn 역학이 다르다(다자 발언권 경쟁). 도메인 태그 또는 별도 통계로 구분해 보고한다. 한국어 다화자 후보(NIA23 002_Meeting, AI Hub 464 회의 음성)는 서버 조사 결과에 따라 추가한다.

## 3. 누출 감사 요약 (정본 §5.2)
- 71631 TS_01.실내_5 757 대화는 E2 학습에 노출됐고 VS_02 는 E2 dev 였다. 따라서 **서버에 있는 71631 로는 untouched KO test 를 만들 수 없다**. KO 최종 보고는 VS_02(dev) 로 하되 "E2 dev 재사용" 을 명시하거나, 미반입 대화를 새로 들여온다.
- otoSpeech 는 E2 에 노출되지 않았고 actor id 가 있어 화자 분리 split 이 가능하다. 단, TurnBench 와 화자가 겹치지 않는다는 원 저자 설명은 있으나 서버 사본에서 재확인한다.
- 합성 원료(LibriSpeech·Kspon·NIKL)는 E2 train split 에서만 뽑는다.

## 4. Q0 에서 확인할 것
1. otoSpeech: SRT 의 turn 유형 태그 목록과 의미, actor id 기준 화자 수, 라이선스 원문.
2. 71631: 화자↔채널 뒤바뀜(기록상 26/757)·한 채널 무음(30 파일) 목록을 Phase 2 manifest 에 플래그로 반영.
3. Switchboard 파일명 시각 규약 전수 검증(시작·끝·duration 일치율).
4. TurnBench dev 의 parquet 스키마와 scorer 재클론.
5. 사용자 결정: 71631 추가 반입(untouched KO test), CANDOR 신청.

## 근거
- 실측 로그 `raw/sources/experiments/2026-09-11-phase2-data-inventory/mxc-inventory.txt`
- [[output-phase2-streaming-asr-diarization-plan]] §5.1–5.2 — 역할·split 원칙
- [[source-conversation-corpora]] — 71631 실물 검증, 라이선스
- [[output-vap-target-pipeline]] — otoSpeech 420 대화 104.9 h, 71631 플래그 파일
- [[output-phase1-report]] §3 — Phase 1 데이터 8 종
- `raw/sources/mxc-soundai-DB-survey.md`(2026-09-06) — Switchboard·CallHome 서버 사본 구조
