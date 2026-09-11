---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2(정본 계획 §5.1) 사용 데이터 목록 — mxc 실측(2026-09-11) 기준. 자연 대화 KO 71631 stereo 248 h(E2 노출 196.6 h + dev 51.7 h)·EN otoSpeech 104.9 h(미노출, 화자 ID 있음), 평가 전용 TurnBench dev, 합성 원료 LibriSpeech·Kspon·NIKL, replay Phase 1 5.6 k h, 추가 조사(§5): NIA24 134-1/134-2 = AI Hub 감정 자유대화 성인·청소년 전체의 화자 채널별 발화 조각(상호상관으로 검증, 2 채널 복원 가능, 실외 1,492 대화 완전 ≈360 h), 확장 후보 Switchboard(파일명 시각으로 복원 가능)·CallHome(시각 없음)·CANDOR(미확보)·영어 회의 코퍼스 AMI/ICSI/CHiME-6/NOTSOFAR-1/DiPCo(3 명 이상, 2 명 활동 창 선택으로 활용)
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


## 5. 확보된 DB 추가 조사 (2026-09-11, 사용자 지정 경로 5 곳) — 정식 보고: [[output-phase2-db-survey]]

조사 경로: `/soundai/DB/raw/aihub`, `…/databricks_build_managed/…/raw`, `…/asr_db/english_16kHz`, `…/asr_db/korean_16kHz/NIA23`, `…/asr_db/korean_16kHz/NIA24`. 읽기 전용, phishing 관련 폴더 제외. 원본 로그는 `raw/sources/experiments/2026-09-11-phase2-data-inventory/`.

### 5.1 최대 발견: NIA24 134-1/134-2 = AI Hub 감정 자유대화 성인·청소년 전체의 발화 단위 사본
| 항목 | 134-1 성인 | 134-2 청소년 |
|---|---|---|
| 위치 | `asr_db/korean_16kHz/NIA24/134-1_Emotion_Conv_Adult/Training/{TS_01.실내,TS_02.실외}` | `…/134-2_Emotion_Conv_Youth/Training/{TS_01.실내,TS_02.실외}` |
| 형식 | 발화 단위 mono 16 kHz wav, 파일명 `<대화 stem>_<TextNo>.wav` | 동일 |
| 전사 목록 | `Transcription/Training/134_1-16k-trans_kor`: 실내 2,072,791 + 실외 371,810 발화, 대화 9,777 개(라벨 9,799 와 일치) | 목록 없음(`wav_list` 폴더) |
| 디스크 | 실내 1,081,267 파일 / 8,270 대화(**완전 38, 부분 8,232**), 실외 380,760 / **1,492 대화 전부 완전** | 실내 1,755,051 / 7,821 대화, 실외 417,170 / 1,649 대화(완전성 미확인) |
| 검증 오디오 | 없음(VS_02 186 대화는 `71631_audio` 의 stereo 원본으로 보유) | 없음 |
| 라벨 | `VAPKT-data/data/labels/aihub71631`(TL/VL 11,023 JSON) — `Conversation[{TextNo, SpeakerNo, StartTime, EndTime, Text, 감정}]` | `/soundai/DB/raw/aihub/71632`(라벨만) |

**검증(상호상관, 2026-09-11)**: stereo 원본을 가진 실내 757 대화 중 563 개에 사본 조각이 있어, 조각을 원본의 채널 0·1 과 상호상관했다. 모든 표본에서 **한 채널과 r = 1.000, 다른 채널과 r ≈ 0**(겹침 구간 포함), 조각 길이 = 라벨 발화 길이. 즉 조각은 **그 화자의 채널에서 라벨 시각대로 잘린 것**이고 `_N` = `TextNo` 다.
- 따라서 조각 하나가 "독립 채널의 한 구간" 이다. **강제 정렬은 조각 단위로 하면 되고**(Phase 1 의 `path#chN` + `src_offset_s` 와 같은 방식, 조각 안 토큰 시각 + StartTime = 절대 시각), 조각을 화자별 채널에 StartTime 대로 배치하면 **겹침이 보존된 2 채널 대화**가 복원된다. 발화 사이 무음의 원음만 없다(무음으로 채움).
- 실외 1,492 대화(완전)는 즉시 쓸 수 있다: 라벨 통계 기준 ≈ 360 h. 실내는 대화당 발화의 약 절반만 있어(결손 패턴 §5.1b) 구멍이 난 대화가 된다 — 결손 발화 구간을 활동·전사 손실에서 마스크하면 전사·activity 학습에는 쓸 수 있고 turn 라벨은 구멍 주변에서 제외한다.
- 청소년 134-2 는 라벨(71632)과 대조해 완전성을 같은 방법으로 확인해야 한다(Q0).

### 5.2 그 밖의 확보 DB
| 자원 | 위치 | 실측 | 판정 |
|---|---|---|---|
| NIA23 002_Meeting(회의) | `NIA23/002_Meeting/Training/{Jsons,wav}/{Broadcast,Internet,Radio,Recording}` | JSON 7,983 회의(예: 화자 5 명, 역할 사회자/토론자), `utterance[{start,end,speaker_id,speaker_role,form,original_form}]` 이 **연속 구간**(0–7.56, 7.56–15.98 …)으로 전체를 덮음; wav 는 발화 단위 mono 16 k 로 혼합 녹음에서 그 시각대로 잘림(Broadcast 만 558,881 파일). NIKL 과 같은 라벨 형식 | **다화자 KO 자연 데이터, B 등급**: 조각을 시각대로 이으면 회의 전체 혼합 오디오가 복원되고 화자·시각 라벨이 있다. 겹침은 조각 안에 섞여 있어 정렬은 혼합 파형 위에서(품질 플래그). 규모는 발화 시각 합산으로 Q0 집계(Broadcast 만 수백 h 추정) |
| NIA24 129_JobInterview | `NIA24/129_JobInterview` | wav 56,958(mono 16 k, 86 s 단위), JSON 5 단계 안에 없음 | 2 자 인터뷰 후보. 라벨 위치 확인 중 |
| NIA24 128_ExpertInterview | 분야 15 종 Training/Validation + Transcription | 발화 단위 mono 16 k(≈17 s), JSON 없음 | 2 자 인터뷰 후보. 시간축 여부 확인 중 |
| NIA23 186_WelfareCallCenter | `NIA23/186_WelfareCallCenter/Training/{Jsons,Wavs}/{Hospital,Mentality,Mobility}` | JSON 30,661; 발화 단위 mono 16 k(`HOS…A015.wav`, 화자 유형 상담사/고객, `sptime_start/end` 는 파일 내 상대 시각) | **통화 단위 시간축 없음** → 대화 복원 불가. 단일 화자 replay·텍스트 순서로만 |
| raw/132 연령대별 특징적 발화 | `raw/132…/Training/TS_대화_<연령>_<주제>` 297 폴더 | 예: 20대_일상 64,725 파일, mono 44.1 kHz ≈ 12.5 s; JSON 은 4 단계 안에 없음(`132_400K_info` 확인 중) | 발화 단위 mono. 시간축 정보가 없으면 복원 불가 |
| korean_8kHz | `callcenter.data/{AI_Assist_1200h, KT_114, lina_cs, lotte-capital, nh…}`, `voicebot` | 사내 콜센터 통화, 8 kHz mono, 발화 단위(`…_1.pcm_0.wav`) | 2 자 통화이나 발화 단위·8 kHz·고객 개인정보. 통화 단위 시간축 여부와 사용 허가 확인 전에는 후보에서 제외 |
| english_16kHz | ami, commonvoice, fleurs, mls, peoples_speech, voxpopuli, yodas | AMI 만 다화자 | AMI 는 §2b |
| english_8kHz | callhome, commonvoice, mls | CallHome 만 대화 | 시각 없음(§2) |
| /soundai/DB/raw/aihub 번호 폴더 | 100·107·108·109·130·132·463·464·470·87·94·95·98 등 | **원천 오디오 없음**(라벨만), 98 은 zip | 2026-09-06 조사와 동일 |
| raw/SD-QA, VoiceAssistant, gigaspeech, commonvoice | — | 단일 화자·QA | 해당 없음 |


### 5.3 강제 정렬 관점의 등급 (사용자 지적 2026-09-11 반영)
정렬은 "그 화자의 소리만 있는 오디오 + 그 화자의 텍스트" 에서만 믿을 수 있다. 자원을 세 등급으로 나눈다.

| 등급 | 조건 | 정렬 방법 | 해당 자원 |
|---|---|---|---|
| **A** | 화자별 독립 채널 또는 **화자 채널에서 잘린 발화 조각** | 채널(또는 조각) 단위 강제 정렬 → 토큰 시각 + 시작 시각 = 절대 시각. 겹침 보존. Phase 1 방식 그대로 | 71631 stereo 원본(실내 757·실외 186), **NIA24 134-1/134-2 조각(상호상관으로 채널 기원 확인)**, otoSpeech, TurnBench, Switchboard(복원 시), AMI IHM |
| B | 혼합 mono 녹음 + 발화 단위 화자·시각 라벨 | 혼합 파형에 화자 텍스트로 정렬하되 겹침 구간은 품질 플래그 낮음. 활동 라벨은 발화 시각에서 | CHiME-6·NOTSOFAR-1·DiPCo 원거리, 002_Meeting(혼합이면), 인터뷰(혼합이면) |
| C | 발화 단위 화자별 오디오인데 **대화 시간축 없음** | 대화 복원 불가. 단일 화자 replay·텍스트 순서 학습에만 | 186 복지 콜센터, CallHome, 사내 8 kHz 콜센터, raw/132(메타 없으면) |

정본 §5.2 의 "혼합 파형에 단일 화자 aligner 를 그대로 적용하지 않는다" 는 원칙은 A 등급에서 자동으로 지켜지고, B 등급은 품질 플래그로 분리 집계한다.

### 5.4 자연 대화 확보 우선순위 (조사 결과)
1. **NIA24 134-1 실외 1,492 대화(완전) ≈ 360 h** — A 등급, 즉시 복원 가능. 현재 248 h 에 더해 KO 자연 대화 ≈ 600 h.
2. **NIA24 134-1 실내 부분 대화 ≈ 7,500 개(발화의 ~45 % 만 존재, 화자·길이·감정에 무관한 무작위 결손)** — 결손 발화 구간을 마스크하면 전사·activity 학습에 사용. 마스크 비율이 절반이라 turn 라벨은 구멍 없는 구간에서만.
3. **NIA24 134-2 청소년(실내 7,821·실외 1,649 대화)** — 라벨(`/soundai/DB/raw/aihub/71632`)과 대조해 완전성 확인 후 1·2 와 같은 방식. 완전하면 최대 수천 h.
4. **NIA23 002_Meeting(KO 다화자 회의, B 등급)** — 화자·시각 라벨 완비, 혼합 mono 전체 복원 가능, 규모 큼(Broadcast 55.9 만 발화). 다화자 전사·diarization 학습의 주 KO 자원 후보. Switchboard 복원(EN 230 h), AMI(§2b).
5. CANDOR(확보 시), ICSI·CHiME-6·NOTSOFAR-1·DiPCo(§2b).
- 제외: AI Hub 번호 폴더(라벨만), 186 복지 콜센터·CallHome·사내 8 kHz 콜센터(시간축 없음/개인정보), raw/132(메타 확인 전).

## 근거
- 실측 로그 `raw/sources/experiments/2026-09-11-phase2-data-inventory/mxc-inventory.txt`
- [[output-phase2-streaming-asr-diarization-plan]] §5.1–5.2 — 역할·split 원칙
- [[source-conversation-corpora]] — 71631 실물 검증, 라이선스
- [[output-vap-target-pipeline]] — otoSpeech 420 대화 104.9 h, 71631 플래그 파일
- [[output-phase1-report]] §3 — Phase 1 데이터 8 종
- `raw/sources/mxc-soundai-DB-survey.md`(2026-09-06) — Switchboard·CallHome 서버 사본 구조
