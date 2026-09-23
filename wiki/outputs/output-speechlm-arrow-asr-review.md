---
type: output
status: active
created: 2026-09-22
updated: 2026-09-22
summary: 새 speechlm Arrow ASR 자료를 현재 2356시간 학습셋과 중복 제거해 도입하는 후보·제외 규칙·검증 순서를 제안한다.
sources:
  - [[source-speechlm-v1-encoder-alignment-2607]]
  - [[output-data-selection-automatic-framework]]
---

# speechlm Arrow ASR 데이터 검토

## 결론

사용할 가치가 크다. 특히 과거 `/soundai/DB/raw/aihub`에는 라벨만 있던 전화·상담·회의 자료의 실제 오디오가 tar에 들어 있어 한국어 도메인 확장에 유용하다. 그러나 이 저장소를 그대로 학습 DB로 간주하면 안 된다.

- 현재 승인셋과 동일한 코퍼스가 여러 개 포함돼 있다.
- Arrow에는 현재 모델이 요구하는 단어·토큰 타이밍이 없다.
- 한국어 AI Hub 대부분이 8 kHz 사본이고, 원전사는 Qwen-verbatim 표시 규약이 아니다.
- 한 카탈로그 폴더에 다른 코퍼스의 고아 tar가 섞인 사례가 있다.

따라서 **Arrow가 가리키는 행만 추출 → 현재 DB와 중복 제거 → 기존 Qwen/Whisper 선별 → Qwen forced alignment → 승인 뷰 생성** 순서로 별도 버전을 만들어야 한다.

## 현재 승인셋 기준

비교 대상은 `approved-qwen-verbatim-mono-v1.1-20260921`이다. 총 1,766,695 발화, 526,198 스트림, 2,356.36시간이며 다음 14개 source를 포함한다.

`librispeech-960`, `kspon-full`, `swbd-train`, `nikl-1000`, `voxpopuli-train`, `yodas-en129`, `aihub-bc-train`, `aihub71631`, `aihub134-1`, `aihub134-2`, `otoSpeech`, `ami`, `notsofar`, `icsi`.

여기서 `aihub-bc-train`은 AI Hub 031/033 방송 통번역 자료다. 새 카탈로그 `000003`의 NIA23 방송 콘텐츠 대화체와는 이름만 비슷하고 원 경로가 달라 신규 후보로 본다.

## 중복 처리표

| 새 ID | 현재 source | 판정 | 처리 |
|---|---|---|---|
| 000019 | `aihub71631`, `aihub134-1` | 동일 AI Hub 134-1의 실내/실외 계열 | 카탈로그 전체 제외 |
| 000020 | `aihub134-2` | 동일 AI Hub 134-2 | 카탈로그 전체 제외 |
| 000032 | `kspon-full` | 동일 KsponSpeech | 전체 제외 |
| 000080 | `librispeech-960` | 동일 LibriSpeech train | 전체 제외 |
| 000085 | `ami` | 동일 AMI | 전체 제외 |
| 000092 | `swbd-train` | 동일 Switchboard | 전체 제외 |
| 000091 | `yodas-en129` | YODAS 전체 중 en129만 중복 | `audio_path`의 `/en129/`만 제외하고 나머지는 후보 |
| 000003 | `aihub-bc-train`과 이름 유사 | 서로 다른 원 경로 | 신규 후보 |

NIKL, VoxPopuli, otoSpeech, NOTSOFAR, ICSI는 새 ASR 카탈로그에서 대응 원천을 찾지 못했다.

중복 제거는 다음 순서로 고정한다.

1. 위 corpus lineage denylist를 먼저 적용한다.
2. `add_info.task_meta.audio_path`를 정규화해 현재 source locator와 비교한다. YODAS `en129`는 이 단계에서 제거한다.
3. 남은 행의 오디오를 16 kHz mono PCM으로 결정적으로 decode한 뒤 SHA-256을 비교한다. 경로·컨테이너가 달라도 같은 파형이면 제외한다.
4. held-out dev/test/eval의 locator와 PCM hash도 같은 방식으로 차단한다.
5. 같은 전사와 비슷한 길이는 exact duplicate가 아니라 **near-duplicate 후보**로만 표시한다. 텍스트만 같다는 이유로 다른 화자의 발화를 제거하지 않는다.

## 우선순위

### P0: 먼저 100~200시간씩 선별할 후보

| ID | 자료 | 이유 | 주의 |
|---|---|---|---|
| 000004, 000005 | 일반·노인 자유대화 | 현재 한국어 화자·연령 다양성 보강 | 8 kHz, 원전사 QC 필요 |
| 000011 | 저음질 전화망 | 전화 대역·실사용 강건성 | 8 kHz가 목적에 맞지만 전체 혼합 비율 제한 |
| 000015, 000016 | 전문 인터뷰·채용면접 | 긴 문장과 질의응답 스타일 | 화자/세션 split 복원 확인 |
| 000027, 000029, 000031 | 복지콜센터·상담·고객응대 | 대화형 한국어와 실제 서비스 도메인 | clip만으로 턴 라벨을 만들지 않음 |
| 000030, 000034 | 한국인 대화·회의 | 구어체·회의 도메인 | 8 kHz, 세션 내 순서/시각 복원 여부 별도 확인 |
| 000063 | 연령대별 은어·속어 발화 | 최신 구어 표현 다양성 | `000036` TTS와 혼동 금지 |
| 000083 | People's Speech clean | 16 kHz 실제 영어, 도메인 다양성 | clean도 teacher 합의 관문 적용 |
| 000087 | Earnings22 | 비원어민 악센트·회의/발표 영어 | tar는 8 kHz 사본 |
| 000093 PART6 | MNSC 대화 파트 | 싱가포르 영어와 대화 다양성 | PART1 낭독과 분리, 기존 MNSC 경로/라벨 감사 이슈 재확인 |

P0는 카탈로그별 상한을 두고 균형 있게 뽑는다. 행 수가 큰 DB 하나가 배치를 지배하지 않도록 첫 뷰에서는 source당 승인 음성 100~200시간을 상한으로 둔다.

### P1: P0 뒤에 추가할 후보

- `000003` 방송 콘텐츠 대화체
- `000008` 외래어·한영 혼합
- `000012` 차량 내 대화/명령, `000013` 소음 환경, `000014` 아동 음성
- `000023`, `000033` 강의
- `000035` MLS, `000078` Common Voice English, `000093 PART1`
- `000091` YODAS 중 `en129` 제외분

이들은 유용하지만 현재 셋의 낭독 비중을 더 키우거나, 잡음·도메인·규모 때문에 P0보다 QC 비용이 크다. 특히 YODAS 3,455만 행과 MLS 1,080만 행은 전량부터 시작하지 않는다.

### 보류 또는 제외

- 중복: `000019`, `000020`, `000032`, `000080`, `000085`, `000092`; `000091/en129`
- 평가 보존: `000062`, `000079` FLEURS. 학습보다 미노출 평가 후보로 남긴다.
- 합성/TTS·번역 파생: `000018`, `000036`, `000037`, `000067`~`000073`
- 명령·주소·숫자 패턴 중심: `000009`, `000017`, `000021`, `000026`
- 저우선: `000084` People's Speech dirty, `000022` 극한소음, `000024` 뉴스, `000025` 낭독, `000028` 어린이 방송
- `000064` 한국어 Common Voice 512행은 규모가 작고 FLEURS와 함께 평가/진단용으로 보존한다.

## 현재 학습 포맷으로 변환할 때의 관문

1. **원천 resolver**: Arrow `data_id`, catalog ID, `audio_path`, `tar_path`, tar file, member를 한 행에 보존한다. tar 디렉터리 전체 glob은 금지한다.
2. **오디오 QC**: tar member 존재, decode, 실제 sample rate/channel/duration, 무음·clipping·과도한 길이를 검사한다. 8 kHz는 16 kHz로 올리되 `source_sample_rate=8000`을 남긴다.
3. **중복·split QC**: 위 lineage/path/PCM hash와 held-out 차단을 먼저 수행한다.
4. **타깃 QC**: Arrow 원전사는 보존하되 학습 타깃으로 바로 쓰지 않는다. 현재 정책대로 Qwen3-ASR 원출력과 Whisper 출력의 정규화 편집거리 비율 5% 이하만 승인한다. TN은 비교용이며 저장 타깃에 강제하지 않는다.
5. **정렬 QC**: 승인된 Qwen 원출력을 현재 tokenizer로 forced alignment하고, 동일 시각 토큰·초당 토큰 과밀·경계 초과를 기존 규칙으로 제외한다.
6. **스트림 구성**: session/speaker가 복원되는 자료는 같은 세션·화자 안에서만 pack한다. 복원되지 않으면 source shard와 길이 affinity 안에서 synthetic stream으로 명시한다.
7. **혼합 비율**: 첫 학습은 P0 source별 시간 상한과 8 kHz 총 비중 상한을 둔다. 8 kHz 비중은 승인 결과를 본 뒤 정하되, 처음부터 한국어 배치 대부분이 되게 하지 않는다.
8. **감사 가능성**: corpus별 입력 행 수, 승인 시간, 교사 합의율, alignment 탈락률, 8/16 kHz 비율, 중복 제거 수와 fingerprint를 뷰 summary에 기록한다.

## 권장 다음 실행

바로 전체 변환 작업을 제출하지 않는다. 먼저 P0 각 source에서 1,000행씩 결정적 표본을 뽑아 다음을 확인한다.

- Arrow→tar resolve 성공률 100%
- source별 실제 시간·sample rate·길이 분포
- Qwen/Whisper 합의율과 원전사 대비 차이
- current train/held-out과 locator·PCM hash 중복 0
- 8 kHz/16 kHz 각 20개 청취 카드

이 관문을 통과한 source만 100~200시간 cap으로 확장한다. 이 방식이면 새 저장소의 가치와 품질을 빠르게 측정하면서 현재 2,356시간 기준선을 오염시키지 않는다.

