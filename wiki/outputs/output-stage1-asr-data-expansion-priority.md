---
type: output
status: active
created: 2026-09-07
updated: 2026-09-07
summary: 1,930h ASR 이후 추가할 EN·KO DB를 준비도·전사 품질·도메인·중복·라이선스로 우선순위화
sources:
  - [[source-mxc-soundai-dataset-survey]]
  - [[source-conversation-corpora]]
  - [[output-asr-tn-v1-spec]]
  - [[output-stage1-mono-pilot]]
---

# Stage 1 ASR 데이터 확장 추천 리스트

## 질문

LibriSpeech 960 h + KsponSpeech 965 h의 30 epoch 학습이 진행되는 동안, 다음 단일 화자
streaming ASR 학습을 위해 어떤 보유 DB를 먼저 준비해야 하는가.

여기서 Stage 1은 **전사 정확도와 token timing을 먼저 검증하는 mono ASR 트랙**을 뜻한다.
turn-taking label은 보존하되 이번 학습 목표에는 넣지 않는다.

## 결론

한 DB만 먼저 고르면 **NIKL 일상대화 음성**이다. 규모가 크고 KO 자연대화라서
KsponSpeech의 짧은 발화·전사 도메인 편향을 가장 직접적으로 보완한다. 다만 전량을 바로
넣지 말고 **overlap 제외 500 h**로 첫 효과를 측정한다.

언어 균형을 맞춘 첫 확장 묶음은 다음이 적절하다.

- **KO:** NIKL 400–500 h
- **EN:** Switchboard 230 h + otoSpeech 105 h + AMI IHM 약 81 h

약 900 h를 추가하되 EN/KO update를 1:1로 sampling한다. 그 다음 묶음에서 MNSC 500 h와
AI Hub 71631을 추가한다. MNSC 4,035 h와 방송 031/033을 처음부터 전량 넣는 것은 중복과
라벨 문제를 가려 버리므로 권하지 않는다.

## 우선순위

| 순위 | DB | 언어·첫 투입량 | 추천 이유 | 투입 전 필수 관문 | 판정 |
|---:|---|---:|---|---|---|
| 1 | **NIKL 일상대화 2020–2025** | KO 500 h | 대규모 자연대화, speaker/time/overlap 정보, 압축 해제 완료 | 실제 duration, PCM 포맷, overlap 제거, speaker/session split, `form`/`original_form` TN audit, 이용조건 | **지금 준비** |
| 2 | **Switchboard** | EN 230 h | 자발적 2자 전화 대화라 LibriSpeech와 상보적, label 준비됨 | native 8 kHz 표시, LDC 권한, train split만, 비언어 marker와 fragment TN | **지금 준비** |
| 3 | **otoSpeech** | EN 104.9 h | 최종 turn-taking 도메인에 가장 가까운 영어, 이미 구조 검증 | 화자별 mono, overlap 제외, non-commercial checkpoint 정책 | **지금 준비** |
| 4 | **AMI IHM** | EN 약 81 h | 회의·비원어민·짧은 backchannel을 보강, CC BY 4.0 | IHM만 사용, official Full-corpus-ASR split, cross-talk/overlap filter | **지금 준비** |
| 5 | **AI Hub 71631 성인 자유대화** | KO 현재 248 h부터 | target-domain 자유대화, 실제 분리 stereo 검증 완료 | 각 channel mono, VAD로 overlap 제외, label timestamp를 정밀 evidence로 간주하지 않음 | **지금 준비, 첫 묶음 뒤 투입** |
| 6 | **MNSC v1** | EN 500 h cap | 대규모 Singapore English와 accent 다양성 | local 4,035 h vs official 2,000 h 해소, audio hash, PART1/6·speaker split, TN audit | **감사 후 준비** |
| 7 | **NIKL 나머지** | KO +500 h 단위 | 첫 500 h가 유효하면 가장 싼 추가 scaling 축 | 연도·화자 분포 고정, 첫 subset과 speaker 중복 없음 | **효과 확인 후 증량** |
| 8 | **AI Hub 031/033 방송** | KO 각 최대 600 h | 방송·인터뷰·드라마·잡음 다양성 | zip 표본, 다화자/배경음 QC, 031↔033 동일 audio hash 제거, 전사 fidelity | **2차 준비** |
| 9 | **AI Hub 98 콜센터** | KO 최대 440 h+ | 상담 대화와 전화망 강건성 | channel/format/실제 시간/전사 매칭 표본검사 | **2차 준비** |
| 10 | **VoxPopuli EN** | EN 공식 543 h | 화자·악센트와 formal speech 다양성 | 현재 tar와 공식 label version 매칭, CC BY-NC 영향 확인 | **label 복구 후** |
| 11 | **CALLHOME** | EN 현재 19.9 h | 친밀한 자발 대화 | 8 kHz, LDC transcript 권한, 작은 규모 | **보조 adaptation** |

## 추천하지 않는 학습 데이터

| DB | 이유 | 사용처 |
|---|---|---|
| **Earnings-22** | 공식적으로 119 h ASR evaluation benchmark다. 서버의 `train` 형태를 근거로 학습하면 benchmark가 오염된다 | 고정 accent/long-form test |
| **TurnBench dev/test** | 최종 turn-taking 평가와 선택에 쓰는 held-out set | 평가 전용 |
| **YODAS-Granary en129** | WAV는 있지만 label provenance와 매칭 파일을 찾지 못했다 | 보류 |
| audio 없는 AI Hub 폴더 | label만으로 supervised ASR을 만들 수 없다 | 원천 재확보 전 제외 |
| 명령어·주소·숫자 패턴 DB | 발화 다양성이 낮고 모델이 제한 문법에 치우칠 위험 | 별도 robustness test 또는 후순위 |
| TTS·합성 음성 DB | 실제 acoustic/timing 분포와 다르고 합성기 artifact를 학습할 수 있음 | 초기 ASR scaling에서 제외 |

## 첫 확장 묶음

### Pack X1: 낮은 불확실성의 대화 적응

| 언어 | 구성 | nominal 시간 |
|---|---|---:|
| EN | Switchboard train + otoSpeech train + AMI Full-corpus-ASR train IHM | 약 416 h |
| KO | NIKL non-overlap, speaker/session 층화 subset | 400–500 h |

목적은 총시간 경쟁이 아니라 **낭독·짧은 발화에서 실제 대화로 옮겨도 WER와 timing이 함께
좋아지는지** 확인하는 것이다. 기존 1,930 h는 anchor/replay로 유지한다.

### Pack X2: 대규모 accent·대화 확장

| 언어 | 구성 | 첫 cap |
|---|---|---:|
| EN | dedup을 통과한 MNSC | 500 h |
| KO | AI Hub 71631 현재 보유분 + NIKL 신규 speaker subset | 약 500 h |

X1 대비 개선이 확인된 뒤 투입한다. MNSC와 NIKL은 한 번에 전량으로 늘리지 않고 500 h
단위 scaling point를 남긴다.

### Pack X3: acoustic robustness

AI Hub 031/033 방송, AI Hub 98 콜센터, VoxPopuli를 각각 독립적으로 붙인다. 이 단계는
clean WER만이 아니라 accent/noise/telephone/broadcast slice의 개선과 anchor regression을
함께 본다.

## 혼합 학습 규칙

1. **EN/KO update는 1:1을 유지한다.** raw hour 비율로 sampling하면 NIKL 또는 MNSC가
   한 언어를 압도한다.
2. **기존 1,930 h를 버리지 않는다.** 첫 실험은 언어 내부에서 기존 anchor 50%, 신규
   대화 DB 50%로 시작하고 corpus별 loss·WER를 본다.
3. **동일 audio의 microphone·번역쌍 복제는 한 번만 센다.** AMI IHM/SDM, MNSC,
   AI Hub 031/033은 audio fingerprint를 split 전에 계산한다.
4. **split은 speaker와 session 단위로 먼저 고정한다.** 발화 단위 random split은 같은
   화자의 목소리와 대화 문맥을 train/dev에 누출한다.
5. **Stage 1에서는 실제 overlap을 제외한다.** 화자별 channel이어도 상대 화자 누설이
   있는 구간은 QC로 제거하고, 원 overlap metadata는 Stage 3용으로 보존한다.
6. **native bandwidth를 보존·기록한다.** 8 kHz 전화 음성을 16 kHz로 resample해도
   `source_sample_rate=8000`을 manifest에 남긴다.
7. **새 DB는 `asr-tn-v1.1.0` parser로 추가한다.** 동결된 v1.0.0 LibriSpeech·KsponSpeech
   target은 바꾸지 않는다. 각 corpus에서 raw/lexical/display source, quarantine reason,
   tokenizer fingerprint를 저장한다.

## 모든 DB가 통과해야 할 준비 관문

- audio decode 성공률 100%, duration·sample rate·channel 통계
- raw transcript 보존, lexical normalization idempotence, tokenizer UNK 0
- 숫자·라틴·축약·비언어 marker 상위 100개 사람 검토
- forced-alignment 성공률과 confidence, 실패·극단 token-rate quarantine
- speaker/session 단위 train/dev split과 cross-corpus audio/text dedup
- 100개 waveform–transcript spot check
- 20–30 s stream 구성 시 같은 speaker/session만 연결
- license와 공개 checkpoint 사용 범위를 corpus metadata에 기록

## 현재 30 epoch run과의 관계

진행 중인 1,930 h 30 epoch run은 recipe를 바꾸지 않고 끝까지 **기준선**으로 남긴다. 새
DB는 준비만 병행하며, 다음 run에서 Pack X1을 추가한다. 비교할 때는 current best checkpoint
에서 이어 학습한 production 후보와 동일 base에서 다시 시작한 clean ablation을 구분한다.

최소 보고 세트는 기존 LibriSpeech/KsponSpeech dev에 corpus별 held-out을 추가한다. 신규
데이터가 clean WER를 조금 개선해도 기존 언어의 `viol80`, 삭제율, bias 0 방출률 또는
tick p99를 악화시키면 채택하지 않는다.

## 불확실성

- NIKL 사용 가능 시간은 2,600–3,800 h로 추정치가 갈리며, 첫 manifest에서 확정해야 한다.
- MNSC는 공식 2,000 h와 서버 4,035 h가 충돌한다. 중복 감사 전에는 4,035 h 데이터로
  계획하지 않는다.
- otoSpeech와 VoxPopuli의 non-commercial 조건이 공개 checkpoint까지 허용하는지는 법률
  검토가 필요하다.
- AI Hub 031/033/98은 서버에 zip이 있지만 실제 audio–label pairing을 아직 열어 보지 않았다.

## 근거

- [[source-mxc-soundai-dataset-survey]] — 서버 경로·시간·label 상태와 공식 자료 대조
- [[source-conversation-corpora]] — otoSpeech·AI Hub 71631의 채널·접근·라이선스
- [[output-asr-tn-v1-spec]] — 새 corpus parser는 minor TN version이라는 계약
- [[output-stage1-mono-pilot]] — 현재 1,930 h scaling과 WER/timing 관문
