---
type: source
status: active
created: 2026-09-22
updated: 2026-09-22
summary: speechlm_v1_encoder_alignment_2607 Arrow와 DI_LAB_CATALOG tar의 ASR 자료 구조·실측 표본·주의점을 기록한다.
sources:
  - /soundai/DB/speechlm/speechlm_v1_encoder_alignment_2607
  - /soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/DI_LAB_CATALOG
---

# speechlm_v1_encoder_alignment_2607 ASR 원천

## 조사 범위

- 조사일: 2026-09-21~22
- Arrow: `/soundai/DB/speechlm/speechlm_v1_encoder_alignment_2607`
- 오디오 tar: `/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/DI_LAB_CATALOG`
- 방법: `catalog_stats.csv`, Hugging Face Arrow의 첫 레코드 배치, tar 검증 JSONL과 일부 tar member를 읽기 전용으로 대조했다.
- 현재 승인 학습 뷰: `/soundai/users/tskim/VAPKT-data/data/selection/approved-qwen-verbatim-mono-v1.1-20260921`

## 구조

각 `SLM-SPEECH-NNNNNN`은 `train/data-*.arrow`, `dataset_info.json`, `state.json`을 가진다. ASR 행에서 실제로 필요한 값은 다음 두 문자열 필드에 들어 있다.

- `conversations`: 사용자 음성 태그와 assistant 전사
- `content_meta`, `add_info`: tar member, 원 오디오 경로, 원전사, 언어, 일부 공개 코퍼스의 길이

`content_meta`와 `add_info`는 JSON이 아니라 작은따옴표를 쓰는 Python literal인 행이 있으므로 `eval`이 아니라 `ast.literal_eval`로 읽어야 한다. tar member는 대체로 `SLM-SPEECH-NNNNNN_<chunk>/...`이고 같은 `<chunk>.tar`에 들어 있다.

이름과 달리 이 경로만으로 현재 VAPASR 학습용 forced alignment가 준비된 것은 아니다. 실측한 `cache-*.arrow` 열은 `_n_user`, `_n_bad`, `_bad_sample`뿐이며 lexical token/word timestamp나 encoder feature는 없다. 현재 시퀀스 규약에 쓰려면 별도로 Qwen 타깃 추출과 forced alignment가 필요하다.

## 실측된 오디오 특성

- AI Hub 한국어 카탈로그 표본은 대부분 **8 kHz, mono**였다. tar validation에서 `000003`, `004`, `005`, `011`, `015`, `016`, `027`, `029`, `030`, `031`, `034`를 확인했다.
- `000083/084` People's Speech, `000091` YODAS, `000093` MNSC는 표본상 **16 kHz, mono**다.
- `000087` Earnings22는 이 tar에서 **8 kHz, mono** 사본이다.
- tar validation의 `status=ok`는 디코딩·sample rate 검증 근거이지 전사 정확도 근거가 아니다.

## 카탈로그 주의점

1. **디렉터리 glob 금지.** `SLM-SPEECH-000035`는 MLS 카탈로그지만 같은 tar 디렉터리의 초기 조각과 validation에는 `commonvoice_21.0` 파일이 존재했고, 이는 `000078`의 tar member와도 일치했다. 반면 `000035` Arrow 표본은 chunk 476의 MLS였다. Arrow의 `tar_path`로 참조되는 member만 사용해야 한다.
2. AI Hub 폴더에는 `assistant_SLM-SPEECH-...tar` 같은 형제 파일도 있다. 이 역시 Arrow가 가리키지 않으면 입력으로 쓰지 않는다.
3. `catalog_stats.csv`의 `kept`는 경로/구조 검증 뒤 행 수다. Qwen–Whisper 전사 합의나 현재 타이밍 밀도 QC를 통과했다는 뜻이 아니다.
4. `dataset_info.json`에는 재사용 판단에 충분한 license/homepage가 채워져 있지 않다. AI Hub와 공개 코퍼스의 이용 조건은 원 배포처 기준으로 별도 기록해야 한다.

## 단일 화자 조건 조사

2026-09-22에 공개 설명과 로컬 구조를 대조했다.

- AIHub 한국어 음성은 2인 대화를 수집했지만 배포 음성을 발화 단위로 segmentation했다고 명시한다. 로컬 000004/000005도 발화별 wav와 단일 화자 폴더 구조다.
- AIHub 복지 콜센터의 공식 파일명 규칙에는 발화자 `A`/`B`가 들어가며, 로컬 000027 파일명도 이 규칙을 따른다.
- MLS 논문은 한 녹음에 여러 화자가 있는 자료를 제거했다고 명시한다.
- Common Voice는 한 기여자가 읽은 문장별 clip과 익명 speaker ID를 기본 단위로 한다.
- 반대로 AIHub 주요 영역별 회의는 3인 이상과 말겹침을 포함한다고 명시한다. mono라는 이유만으로 단일 화자로 볼 수 없으므로 000034는 제외한다.

외부 근거 URL과 최종 분류는 [[output-speechlm-single-speaker-manifest]]에 기록했다.

## 주요 행 수

`catalog_stats.csv`의 `kept` 기준이다. 이는 시간이 아니라 발화 행 수다.

| ID | 원천 | kept |
|---|---|---:|
| 000003 | AI Hub 방송 콘텐츠 대화체 | 4,971,355 |
| 000004 | AI Hub 자유대화 일반 | 2,003,824 |
| 000005 | AI Hub 자유대화 노인 | 972,646 |
| 000011 | AI Hub 저음질 전화망 | 4,437,351 |
| 000015 | AI Hub 전문분야 인터뷰 | 684,200 |
| 000016 | AI Hub 채용면접 | 43,077 |
| 000027 | AI Hub 복지 콜센터 | 1,686,365 |
| 000029 | AI Hub 상담 | 1,583,840 |
| 000030 | AI Hub 한국인 대화 | 2,547,047 |
| 000031 | AI Hub 고객 응대 | 2,082,000 |
| 000034 | AI Hub 회의 | 1,803,036 |
| 000035 | MLS English | 10,808,037 |
| 000078 | Common Voice 21 English | 1,131,708 |
| 000083 | People's Speech clean | 1,495,619 |
| 000084 | People's Speech dirty | 2,121,450 |
| 000087 | Earnings22 | 38,347 |
| 000091 | YODAS | 34,556,069 |
| 000093 | Multitask National Speech Corpus v1 | 2,276,881 |

전체 ASR 카탈로그와 현재 학습셋의 중복·채택 우선순위는 [[output-speechlm-arrow-asr-review]]에 정리한다.
