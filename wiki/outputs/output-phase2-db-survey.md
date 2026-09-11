---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 자연 대화 데이터 확보를 위한 mxc 확보 DB 조사 보고(2026-09-11, 사용자 지정 5 경로 + 공개 DB 후보) — NIA24 134-1/134-2 = AI Hub 감정 자유대화 성인·청소년 전체의 화자 채널 기원 발화 조각(상호상관 검증), 실외 1,492 대화 완전 ≈360 h, NIA23 회의 코퍼스는 다화자 B 등급, 정렬 등급 A/B/C 와 우선순위, 제외 목록
sources:
  - '[[output-phase2-data-inventory]]'
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[decision-multi-speaker-scope]]'
  - '[[source-conversation-corpora]]'
  - '[[output-vap-target-pipeline]]'
---

# Phase 2 확보 DB 조사 보고 (2026-09-11)

## 질문
Phase 2 의 자연 다화자 대화 데이터(현재 확정 ≈290 h)가 부족하다. mxc 에 이미 있는 DB 와 확보 가능한 공개 DB 가운데 (1) 화자별 채널 또는 화자·시각 라벨이 있어 대화를 복원할 수 있고, (2) 강제 정렬을 믿을 수 있는 자원은 무엇인가. 사용자가 지정한 다섯 경로를 읽기 전용으로 조사했다.

## 요약
- **가장 큰 발견**: `asr_db/korean_16kHz/NIA24/134-1_Emotion_Conv_Adult`·`134-2_Emotion_Conv_Youth` 는 AI Hub 감정 자유대화 **성인·청소년 전체**가 발화 단위 mono 조각으로 들어 있는 사본이다. 상호상관 검증으로 조각이 **그 화자의 채널에서 라벨 시각대로 잘린 것**임을 확인했다(한 채널 r = 1.000, 다른 채널 ≈ 0, 겹침 구간 포함). 따라서 조각 단위 강제 정렬과 겹침이 보존된 2 채널 복원이 가능하다.
- 성인 **실외 1,492 대화는 완전**(≈360 h)해 즉시 쓸 수 있고, 실내 ≈7,500 대화는 발화의 약 45 % 만 있어(무작위 결손) 마스크가 필요하다. 청소년(실내 7,821·실외 1,649 대화)은 완전성 미검증.
- 2 화자 제약 제거([[decision-multi-speaker-scope]])로 **NIA23 002_Meeting(한국어 회의, 화자 5 명 규모, 화자·시각 라벨 완비, 혼합 mono 전체 복원 가능)** 이 다화자 학습의 주 한국어 후보가 됐다.
- `/soundai/DB/raw/aihub` 번호 폴더는 라벨만 있고 원천 오디오가 없다. 콜센터(복지 186·사내 8 kHz)·CallHome·raw/132·인터뷰는 발화 단위 오디오만 있고 통화/대화 시간축이 없어 대화 복원이 안 된다.
- 조사 결과로 한국어 자연 대화는 248 h → **≈600 h(A 등급) + 회의 수백 h(B 등급)** 로 늘어난다. 영어는 otoSpeech 104.9 h 에 Switchboard 복원(230 h)·AMI 가 후보다.

## 1. 조사 방법
- 경로: `/soundai/DB/raw/aihub`, `…/databricks_build_managed/…/raw`, `…/asr_db/english_16kHz`, `…/asr_db/korean_16kHz/NIA23`, `…/asr_db/korean_16kHz/NIA24`. 추가로 `asr_db/korean_8kHz`, `english_8kHz`, `RequestDB`, `nia24_raw` 루트를 목록만 확인.
- 읽기 전용(`ls`, `find -maxdepth ≤ 6`, 파일 헤더 36 바이트, JSON 앞부분). phishing 관련 폴더는 열지 않음. 서버 부하(load 105, 접속자 132)로 `find` 는 timeout 을 두었고 큰 디렉토리(100 만 항목)는 목록 파일로 대신 셌다.
- 검증 실험: 조각 ↔ stereo 원본 상호상관(`xcorr2.py`), 결손 패턴(`missing.py`). 로그·스크립트: `raw/sources/experiments/2026-09-11-phase2-data-inventory/`.

## 2. 경로별 결과

### 2.1 `/soundai/DB/raw/aihub`
| 폴더 | 내용 | 오디오 | 판정 |
|---|---|---|---|
| 71631, 71631_audio | 감정 자유대화(성인) 라벨 11,023 JSON + stereo 원본 실내 757(196.6 h)·실외 186(51.7 h) | 있음 | 기존 보유(A 등급) |
| 71632 | 감정 자유대화(청소년) 라벨 | 없음 | 134-2 조각의 라벨로 사용 |
| 100·107·108·109·130·132·463·464·470·87·94·95·112·115·131·484·485·540·568·571·71260·71349·71405·71417·71481·71487·71502·71556·71557·71592·71627 | 상담·자유대화·회의·콜센터·강의·명령어 등 라벨 | **없음**(maxdepth 4 안에 wav/pcm/zip 없음) | 제외(2026-09-06 조사와 동일) |
| 98 | 민원(콜센터) 질의응답 | zip(원천 존재 기록) | 라벨 부재로 보류 |
| 71610 | 텍스트 QA(JSON) | — | 해당 없음 |

### 2.2 `…/raw`
| 폴더 | 내용 | 판정 |
|---|---|---|
| 132.연령대별_특징적_발화 | `Training/TS_대화_<연령>_<주제>` 297 폴더, 발화 단위 mono 44.1 kHz(예: 20대_일상 64,725 파일 ≈12.5 s), JSON 은 4 단계 안에 없음(`132_400K_info` 미확인) | 시간축 정보 없으면 대화 복원 불가. 보류 |
| SD-QA, VoiceAssistant-400K, gigaspeech, commonvoice_21.0 | 단일 화자·QA | 해당 없음 |

### 2.3 `…/asr_db/english_16kHz`
| 폴더 | 내용 | 판정 |
|---|---|---|
| ami | AMI 회의, 발화 단위 wav(ihm/sdm) + CSV·JSONL(spk_id, meeting_id, microphone) | 다화자 후보. 서버 사본은 발화 단위라 원본 회의 파일·단어 시각 확보 시 복원(§4) |
| commonvoice, fleurs, mls_english, peoples_speech, voxpopuli, yodas-granary | 단일 화자 | 해당 없음 |
| (참고) `english_8kHz/callhome` | 2 자 통화, 발화 단위, 파일명에 시각 없음 | 복원 불가 |

### 2.4 `…/asr_db/korean_16kHz/NIA23`
| 폴더 | 내용 | 판정 |
|---|---|---|
| **002_Meeting** | 회의 7,983 개(방송·인터넷·라디오·녹음). JSON: `metadata.speaker_num`(예 5), `speaker[]`(역할), `utterance[{start,end,speaker_id,speaker_role,form,original_form}]` 이 연속 구간으로 전체를 덮음. wav 는 혼합 녹음을 발화 시각대로 자른 mono 16 k 조각(Broadcast 558,881 파일) | **다화자 KO, B 등급**. 조각을 시각대로 이으면 회의 전체 혼합 오디오 복원. 규모는 Q0 집계 |
| 186_WelfareCallCenter | JSON 30,661; 발화 단위 mono(상담사/고객, `sptime_*` 는 파일 내 상대 시각) | 통화 시간축 없음 → C 등급, 제외 |
| 008·011·158 | 잡음·아동·낭송 | 해당 없음 |

### 2.5 `…/asr_db/korean_16kHz/NIA24`
| 폴더 | 내용 | 판정 |
|---|---|---|
| **134-1_Emotion_Conv_Adult** | 발화 조각 `<stem>_<TextNo>.wav`(mono 16 k). 전사 목록 실내 2,072,791 + 실외 371,810 발화, 대화 9,777(라벨 9,799 와 일치). 디스크: 실내 1,081,267 파일/8,270 대화(완전 38, 부분 8,232, 결손 15), 실외 380,760/1,492 대화 **전부 완전**. 검증 오디오 없음 | **A 등급**(§3 검증). 실외 즉시 사용, 실내는 결손 마스크 |
| **134-2_Emotion_Conv_Youth** | 실내 1,755,051 파일/7,821 대화, 실외 417,170/1,649 대화. 전사 목록 없음(`wav_list`). 라벨은 `aihub/71632` | A 등급 추정. 완전성 미검증 |
| 128_ExpertInterview | 분야 15 종, 발화 단위 mono(≈17 s), JSON 없음 | 시간축 미확인, 보류 |
| 129_JobInterview | wav 56,958(≈86 s), JSON 없음 | 시간축 미확인, 보류 |
| 133_SpeakingStyle, 130_Address, 135/136 소음 | 단일 화자·합성용 | 해당 없음 |

### 2.6 그 밖의 루트(목록만)
- `korean_8kHz/callcenter.data/{AI_Assist_1200h, KT_114, lina_cs, lotte-capital, nh, …}`, `voicebot`: 사내 콜센터 통화 8 kHz 발화 단위. 통화 시간축·사용 허가 미확인, 고객 개인정보 → 후보에서 제외.
- `RequestDB`, `nia24_raw`(128·133·135·136·139-1 방언·144·254·293 원본), `NIA_FIRST`: 대화 자원 없음(139-1 방언은 미확인).
- NIKL 트리 실측 832 GB(발화 단위 mono, 시각·화자 ID 있음, 겹침 발화 quarantine).

## 3. 검증 실험: NIA24 조각의 채널 기원
- 대상: stereo 원본을 가진 실내 757 대화 중 사본 조각이 있는 563 대화에서 2 대화·12 발화.
- 방법: 조각 파형을 라벨 StartTime 위치의 원본 채널 0·채널 1·합산 신호와 상관.
- 결과: 모든 발화에서 한 채널과 **r = 1.000**, 다른 채널과 |r| ≤ 0.055. 겹치는 두 발화(7.35–9.15 s / 7.50–9.90 s)도 각자 자기 채널과만 일치. 조각 길이 = 라벨 발화 길이(예 0.95/1.80/2.40/4.30 s). `_N` = `TextNo`.
- 결론: 조각은 화자 채널에서 라벨 시각대로 잘렸다. **조각 단위 강제 정렬이 곧 독립 채널 정렬**이고(Phase 1 의 `path#chN`+`src_offset_s` 방식), 조각을 화자별 채널에 배치하면 겹침이 보존된 2 채널 대화가 복원된다. 발화 사이 무음의 원음만 없다.
- 결손 패턴(실내 12 대화 2,843 발화): 존재율 화자 1 45 %·화자 2 45 %, 길이 중앙값 존재 1.54 s·결손 1.70 s, 감정 라벨 분포 동일 → 무작위 결손.

## 4. 정렬 등급과 활용
| 등급 | 조건 | 정렬 | 자원 |
|---|---|---|---|
| A | 화자별 채널 또는 채널 기원 조각 | 채널/조각 단위, 겹침 보존 | 71631 원본, NIA24 134-1/134-2, otoSpeech, TurnBench, Switchboard(복원 시), AMI IHM |
| B | 혼합 mono + 화자·시각 라벨 | 혼합 파형 위 정렬, 겹침 구간 품질 플래그 | NIA23 002_Meeting, CHiME-6·NOTSOFAR-1·DiPCo, 인터뷰(혼합이면) |
| C | 발화 단위 오디오, 시간축 없음 | 복원 불가, 단일 화자 replay 만 | 186 콜센터, CallHome, 사내 8 kHz, raw/132 |

## 5. 우선순위와 예상 규모
| 순위 | 자원 | 규모 | 조치 |
|---|---|---|---|
| 1 | NIA24 134-1 실외 1,492 대화 | ≈360 h(라벨 통계 비례) | 2 채널 복원 → manifest(A) |
| 2 | NIA24 134-1 실내 부분 ≈7,500 대화 | 발화 45 % | 결손 구간 마스크, 전사·activity 용 |
| 3 | NIA24 134-2 청소년 9,470 대화 | 최대 수천 h | 71632 라벨과 완전성 검증(§3 방법) |
| 4 | NIA23 002_Meeting | 수백 h(추정) | 시각 합산 집계, 혼합 복원, B 등급 정렬 |
| 5 | Switchboard 복원, AMI | 230 h, ~100 h | 파일명 시각 검증, 원본 회의 파일 확보 |
| 6 | CANDOR, ICSI, CHiME-6, NOTSOFAR-1, DiPCo | 확보 시 | [[task-secure-english-corpora]] |

## 6. 미확인 사항
- 134-2 청소년 조각의 완전성·채널 기원(성인과 같은 방식으로 검증).
- 002_Meeting 의 총 시간, 겹침 표기 유무(`environment` 등), Internet/Radio/Recording 의 녹음 조건.
- raw/132 의 `132_400K_info`, 128/129 인터뷰의 라벨 위치, 139-1 방언의 대화 구조.
- Switchboard 파일명 시각 규약 전수 검증, AMI 원본 회의 오디오 확보.
- 라이선스: otoSpeech 원문, NOTSOFAR-1·DiPCo.

## 근거
- 실측 로그·스크립트: `raw/sources/experiments/2026-09-11-phase2-data-inventory/`(survey1–7, meeting, xcorr2.py, missing.py)
- [[output-phase2-data-inventory]] — 이번 단계 사용 데이터 목록(본 보고의 결과를 §5 에 반영)
- [[decision-multi-speaker-scope]] — 2 화자 제약 제거
- [[source-conversation-corpora]], [[output-vap-target-pipeline]] — 71631 실물 검증·라벨 통계
- `raw/sources/mxc-soundai-DB-survey.md`(2026-09-06) — 이전 조사
