---
type: source
status: active
created: 2026-09-07
updated: 2026-09-07
summary: mxc 보유 EN·KO 음성 DB를 실물 경로·시간·라벨·사용 가능성으로 조사하고 공식 정보와 대조한 서베이
raw_paths:
  - raw/sources/mxc-soundai-DB-survey.md
  - raw/sources/SLM원천데이터.md
observed: 2026-09-07
raw_authors:
  - unknown
---

# mxc `/soundai/DB` 음성 데이터셋 서베이

## 무엇인가

2026-09-06 mxc의 `/soundai/DB`와 팀 공용 EN 빌드 영역을 읽기 전용으로 조사한 기록과,
SLM-SPEECH ID 목록을 합친 source note다. 서버에 실제 audio와 transcript가 함께 있는지,
압축이 풀렸는지, 단일 화자 ASR 또는 이후 turn-taking에 사용할 수 있는지를 구분한다.

이 문서는 서버 실측을 우선하되, 공개 코퍼스의 공식 규모·원 포맷·사용 목적과 충돌하는
항목은 공식 자료로 교정하거나 미확인으로 남긴다.

## 서버에서 확인한 주요 후보

| 코퍼스 | 언어 | 서버 실측·표기 규모 | 현재 상태 | 핵심 가치 |
|---|---|---:|---|---|
| NIKL 일상대화 음성 2020–2025 | KO | 발화 515만, 대화 17,156개, 약 2,600–3,800 h 추정 | 압축 해제, PCM/WAV+JSON | 대규모 자연 대화, speaker/time/overlap metadata |
| AI Hub 71631 성인 자유대화 | KO | 팀 보유 248.3 h, 전체 라벨 실측 2,765 h | 일부 audio와 전체 label 보유 | 분리 stereo, target-domain 대화 |
| AI Hub 031/033 방송 통번역 | KO | 공식 각 600 h, 서버 원천+라벨 zip | 표본 미검증 | 방송·인터뷰·드라마 등 음향·발화 스타일 다양화 |
| AI Hub 98 민원 콜센터 | KO | 공식 audio 440 h 이상 | 서버 원천+라벨 zip, 내부 미검증 | 실제 상담 대화와 전화 음질 |
| otoSpeech | EN | 104.9 h | 팀 보유 | target-domain 화자별 대화 |
| Switchboard 파생 빌드 | EN | train CSV 합 230 h | 발화 WAV+정규화 CSV | 자발적 2자 전화 대화 |
| AMI 파생 빌드 | EN | IHM+SDM 합산 train 288 h | 발화 WAV+CSV/JSONL | 회의·비원어민·겹침, microphone metadata |
| MNSC v1 파생 빌드 | EN | train CSV 합 4,035 h | 발화 WAV+CSV | 싱가포르 영어·대규모 accent 확장 |
| VoxPopuli EN | EN | 압축 tar 60 GB | 로컬 label 미발견 | 공식 transcribed EN 543 h, 의회 연설 |
| YODAS-Granary en129 | EN | WAV 138,589개 | 로컬 label 미발견 | YouTube 구어·잡음 다양성 |
| CALLHOME 파생 빌드 | EN | chunked CSV 19.9 h | 발화 WAV+CSV | 친밀한 2자 전화 대화 |
| Earnings-22 파생 빌드 | EN | train 명칭 CSV 104.6 h | audio+CSV | accented long-form 평가용 |

## 공식 자료와 대조하면서 발견한 차이

| 항목 | 서버 조사 | 공식 근거 | 판정 |
|---|---|---|---|
| NIKL 시간 | PCM byte 환산 약 3,800 h, 발화 길이 합 약 2,600 h | 모두의 말뭉치는 연도별 PCM+전사 제공을 확인하지만 합산 시간은 이 조사에서 확인 못 함 | 두 추정치를 모두 보존하고 manifest에서 실제 decode duration 재계산 |
| MNSC 규모 | 파생 CSV 4,035 h | IMDA는 NSC v1을 2,000 h로 소개 | microphone/part/중복 또는 빌드 차이 감사 전 총량으로 인정하지 않음 |
| AMI 규모 | IHM+SDM 행 합 288 h | 공식 corpus 약 100 h, Full-corpus-ASR train 약 81 h | 같은 발화를 microphone별로 중복 집계한 값. Stage 1은 IHM 한 종류만 사용 |
| Switchboard 포맷·시간 | 파생 WAV 16 kHz, train 230 h | 원본은 약 260 h, 8 kHz 2-channel µ-law | 16 kHz는 resample이며 새 고주파 정보가 아니다. native bandwidth 표시 필수 |
| CALLHOME 시간 | 파생 chunk 19.9 h | 원 audio 약 56 h, transcript는 별도 상품 | 현재 매칭된 subset만 사용 가능, 공식 전체 시간과 혼동 금지 |
| Earnings-22 용도 | 서버 폴더에 train/valid/test 형태 | 공식 dataset card와 논문은 119 h **evaluation benchmark**로 정의 | 학습 금지, 고정 robustness test로 격리 |
| VoxPopuli | tar는 있으나 label 없음 | 공식 transcribed English 543 h | label provenance를 복구하기 전 학습 불가 |

## 사용 가능성 판정

### 바로 parser·QC를 만들 가치가 있는 것

- **NIKL**: 가장 큰 KO 확장 후보다. 단일화자 Stage 1에서는 overlap 표시 발화를
  제외하고, speaker·conversation 단위 split을 먼저 만든다.
- **Switchboard**: 작은 준비 비용으로 EN 자발 대화를 보강한다. native 8 kHz와 LDC
  사용 조건을 checkpoint metadata에 남긴다.
- **AMI IHM**: IHM만 택하고 공식 Full-corpus-ASR split을 사용하면 microphone 중복을
  피할 수 있다.
- **otoSpeech·AI Hub 71631 현재 보유분**: 이미 구조를 검증한 target-domain 자료다.
  분리 채널의 한 화자씩 mono로 사용하고 실제 overlap 구간은 Stage 1에서 제외한다.

### 감사를 통과한 뒤 사용할 것

- **MNSC**: official 2,000 h와 local 4,035 h 차이, speaker split, PART1/PART6 구성,
  중복 audio를 먼저 확인한다. 첫 학습은 500 h cap으로 제한한다.
- **AI Hub 031/033**: zip 표본을 풀어 포맷·전사 fidelity·화자 수를 보고, 두 DB와 언어쌍
  사이 동일 한국어 audio를 fingerprint로 제거한다.
- **AI Hub 98**: 상담사/고객 channel 구조, 원 audio 시간, transcript가 실제 음성과
  일치하는지 표본 검사한다.
- **VoxPopuli**: 공식 label과 현재 tar의 release/version 대응을 복구한 뒤 사용한다.

### 현재 학습에서 제외할 것

- **Earnings-22**와 **TurnBench dev/test**: 평가 전용으로 격리한다.
- **YODAS-Granary**: label provenance·license·audio-text 매칭이 없다.
- 원천 audio가 없는 AI Hub label-only 폴더, 명령어·TTS·합성 중심 DB.
- CALLHOME은 현재 매칭 19.9 h가 작고 native 8 kHz라 우선 준비 대상은 아니다.

## 라이선스와 공개성

- NIKL은 신청형 배포이므로 이용 범위와 파생 checkpoint 공개 조건을 별도로 확인한다.
- AI Hub는 영리·비영리 연구개발 활용을 허용하지만 원 데이터의 권리는 NIA와 수행기관에
  있으며 신청·보안·재배포 조건을 지켜야 한다.
- Switchboard와 CALLHOME은 LDC User Agreement 적용 대상이다.
- AMI audio·transcript는 CC BY 4.0으로 공개되어 있다.
- MNSC v1은 Singapore Open Data License로 소개된다. 다만 서버 파생본의 provenance와
  공식 release 일치 여부는 별도 확인한다.
- otoSpeech의 non-commercial 조건은 공개 checkpoint 계획과 함께 재검토한다.

## 불확실성 / 미확인

- NIKL의 2,600–3,800 h 차이는 침묵 포함 여부만으로 확정할 수 없다. decoder로 실제
  duration을 합산해야 한다.
- AI Hub 031과 033의 600 h가 서로 독립적인 한국어 audio인지 확인되지 않았다.
- MNSC local 4,035 h는 공식 2,000 h보다 두 배 이상이므로 그대로 학습 시간으로 쓰면 안 된다.
- 서버의 정규화 필드 `script_tn`이 [[output-asr-tn-v1-spec]]과 일치한다는 보장은 없다.

## 출처

- 원본 조사: `raw/sources/mxc-soundai-DB-survey.md`
- SLM 목록: `raw/sources/SLM원천데이터.md`
- [국립국어원 모두의 말뭉치](https://kli.korean.go.kr/corpus/main/requestMain.do?lang=ko) (관측 2026-09-07)
- [AI Hub 71631 성인 자유대화](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71631) (관측 2026-09-07)
- [AI Hub 031 방송 통번역](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71379) (관측 2026-09-07)
- [AI Hub 033 방송 통번역](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71384) (관측 2026-09-07)
- [AI Hub 98 민원 콜센터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=98) (관측 2026-09-07)
- [AI Hub 데이터 이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) (관측 2026-09-07)
- [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (관측 2026-09-07)
- [Switchboard-1 Release 2](https://catalog.ldc.upenn.edu/LDC97S62) (관측 2026-09-07)
- [CALLHOME American English Speech](https://catalog.ldc.upenn.edu/LDC97S42) (관측 2026-09-07)
- [IMDA National Speech Corpus v1 fact sheet](https://www.imda.gov.sg/-/media/Imda/Files/Industry-Development/Infrastructure/Technology/Digital-Services-Lab/2018-11-22-Fact-Sheet-for-AI-Open-Source-subFINAL-v3_2.pdf) (관측 2026-09-07)
- [VoxPopuli official repository](https://github.com/facebookresearch/voxpopuli) (관측 2026-09-07)
- [Earnings-22](https://arxiv.org/abs/2203.15591) (관측 2026-09-07)

## 이 볼트에 준 영향

- Stage 1 데이터 우선순위: [[output-stage1-asr-data-expansion-priority]]
- 기존 대화 코퍼스 현황: [[source-conversation-corpora]]
- 새 corpus parser 버전 정책: [[output-asr-tn-v1-spec]]
