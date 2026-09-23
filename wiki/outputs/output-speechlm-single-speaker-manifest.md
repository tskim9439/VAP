---
type: output
status: active
created: 2026-09-22
updated: 2026-09-22
summary: speechlm 2607에서 단일 화자 클립 조건을 만족하는 26개 카탈로그와 새 Arrow manifest 생성 규약을 고정한다.
sources:
  - [[source-speechlm-v1-encoder-alignment-2607]]
  - [[output-speechlm-arrow-asr-review]]
---

# speechlm 2607 단일 화자 ASR manifest

## 결론

`speechlm_v1_encoder_alignment_2607` 전체를 학습 데이터로 사용하지 않는다. 공개 DB 설명과 원본 Arrow/tar 구조를 함께 확인해, **한 파일이 한 화자의 발화 조각인 근거가 있는 26개 카탈로그만** 1차 manifest에 넣는다. mono는 한 채널이라는 뜻일 뿐 한 화자라는 뜻은 아니므로, 회의·방송·웹 수집 자료는 mono여도 화자 분리 근거가 없으면 제외한다.

현재 승인 학습셋과 겹치는 KsponSpeech, LibriSpeech, AMI, Switchboard, AIHub 134 계열도 제외한다. FLEURS는 미노출 평가셋으로 보존하고 합성·TTS·번역 파생 자료는 제외한다.

## 1차 채택 목록

| ID | 세부 DB | 언어 / 원 표본률 | 채택 근거 |
|---|---|---|---|
| 000004 | AIHub 일반 자유대화 | ko / 8 kHz | 공식 설명의 발화 단위 segmentation |
| 000005 | AIHub 노인 자유대화 | ko / 8 kHz | 발화 단위 파일 |
| 000008 | AIHub 외래어·한영 혼합 발화 | ko / 8 kHz | 단일 화자 prompt clip |
| 000009 | AIHub 노인 명령어 | ko / 8 kHz | 단일 화자 prompt clip |
| 000011 | AIHub 저음질 전화망 | ko / 8 kHz | 화자 발화 단위 wav |
| 000012 | AIHub 차량 내 대화·명령 | ko / 8 kHz | 화자 발화 단위 wav |
| 000013 | AIHub 다양한 장소 소음 | ko / 8 kHz | 한 음성에 잡음을 더한 clip |
| 000014 | AIHub 아동 음성 | ko / 8 kHz | 단일 화자 prompt clip |
| 000015 | AIHub 전문분야 인터뷰 | ko / 8 kHz | 화자 발화 단위 wav |
| 000016 | AIHub 채용면접 | ko / 8 kHz | 질문·답변별 발화 조각 |
| 000017 | AIHub 주소 발화 | ko / 8 kHz | 단일 화자 prompt clip |
| 000021 | AIHub 소음 명령어 | ko / 8 kHz | 한 음성에 잡음을 더한 clip |
| 000022 | AIHub 극한소음 음성인식 | ko / 8 kHz | 한 음성에 잡음을 더한 clip |
| 000023 | AIHub 대학 강의 | ko / 8 kHz | 단일 강연자 발화 조각 |
| 000024 | AIHub 뉴스 앵커 | ko / 8 kHz | 단일 앵커 발화 조각 |
| 000025 | AIHub 문학 낭독 | ko / 8 kHz | 단일 낭독자 발화 조각 |
| 000026 | AIHub 숫자 패턴 발화 | ko / 8 kHz | 단일 화자 prompt clip |
| 000027 | AIHub 복지 콜센터 | ko / 8 kHz | 파일명에 A/B 발화자 구분이 명시됨 |
| 000029 | AIHub 상담 음성 | ko / 8 kHz | speaker 디렉터리별 발화 조각 |
| 000031 | AIHub 고객 응대 | ko / 8 kHz | speaker 디렉터리별 발화 조각 |
| 000033 | AIHub 한국어 강의 | ko / 8 kHz | speaker 디렉터리별 발화 조각 |
| 000035 | Multilingual LibriSpeech English | en / 8 kHz 사본 | 공식적으로 다화자 녹음을 제거한 단일 낭독자 자료; `/mls_english/`만 |
| 000063 | AIHub 연령대별 은어·속어 발화 | ko / 44.1 kHz | 단일 화자 prompt clip |
| 000064 | Common Voice Korean | ko / 8 kHz 사본 | 자원자가 문장을 읽어 제출한 개별 clip |
| 000078 | Common Voice 21 English | en / 8 kHz 사본 | 자원자가 문장을 읽어 제출한 개별 clip |
| 000093 | MNSC v1 ASR PART1 | en / 16 kHz | 낭독 파트만 경로 필터; PART6 대화는 제외 |

AIHub 일반 자유대화는 2인이 대화해 수집했지만 배포 구조가 **발화 단위 음성파일**이라고 명시한다. MLS 논문은 여러 화자가 들어간 오디오를 제거했다고 설명하며, Common Voice 논문은 각 행이 한 익명 화자가 읽은 한 clip이라고 정의한다. 복지 콜센터는 공식 파일명 규칙에 A/B 화자 코드가 있다.

## 보류 목록

- `000003` 방송 대화, `000028` 어린이 방송: 한 파일 한 화자 보장이 없다.
- `000030` 한국인 대화, `000034` 회의: 원자료가 다화자·원거리·겹침을 포함한다.
- `000083`, `000084` People's Speech와 `000091` YODAS: 웹 수집 clip의 단일 화자 보장이 없다.
- `000087` Earnings22: 실적발표 통화 자체는 다화자다. 현재 8 kHz chunk가 speaker-change segmentation 결과인지 provenance 확인 뒤 승격한다.
- `000093` PART6: 대화 파트는 채널·화자 분리 설명을 확인하기 전까지 제외한다.

보류 자료는 음질이 나빠서가 아니라, 이번의 **파일당 단일 화자** 계약을 문서만으로 증명할 수 없어서 제외한다.

## 중복·평가·합성 제외

- 현재 학습셋 중복: `000019`, `000020`, `000032`, `000080`, `000085`, `000092`.
- 미노출 평가 보존: `000062`, `000079` FLEURS.
- 합성/TTS/번역 파생: `000018`, `000036`, `000037`, `000067`~`000073`.

## 새 Arrow 규약

전체 산출물 경로는 다음으로 고정한다.

`/soundai/users/tskim/VAPKT-data/data/speechlm-asr-single-speaker-v1-20260922`

각 행은 아래 필드를 갖는다.

- 식별·출처: `manifest_version`, `catalog_id`, `corpus`, `data_id`, `source_arrow`, `source_row_index`
- 오디오: `audio_tar`, `audio_member`, `source_audio_path`, `sample_rate`, `num_channels`, `num_frames`, `duration_sec`
- 라벨: `original_transcript`, `language`, `split`
- 화자 감사: `speaker_id`, `speaker_separation_basis`

`original_transcript`는 원본 Arrow 라벨을 그대로 보존하며 TN이나 Qwen 재전사를 적용하지 않는다. `audio_tar`는 카탈로그 디렉터리 glob이 아니라 각 Arrow 행의 `tar_path`로 결정한다. 이 규칙은 000035 폴더에 섞여 있는 Common Voice 고아 tar가 MLS로 유입되는 것을 막는다.

duration은 같은 tar shard의 `*.duration.json`에서 결합하고, sample rate와 mono 여부는 기존 전체 tar validation 산출물의 카탈로그 포맷을 사용한다. 빌더는 25만 행 단위 zstd Arrow stream으로 저장하며, `summary.json`에 카탈로그별 행 수·시간·duration/tar 누락 수를 기록한다.

재현 코드는 `experiments/build_speechlm_single_speaker_arrow.py`, 병렬 실행기는 `experiments/run_speechlm_single_speaker_arrow.sh`, 정책 파일은 `experiments/speechlm_single_speaker_catalogs.json`이다. 26개 DB × 3행 스모크에서는 78행, 0.153시간을 만들었고 tar/duration 누락은 0이었다.

전체 빌드는 완료됐다.

- 26개 catalog, 47,662,558행, 108,512.79시간
- zstd Arrow 239개, 4,849,313,432 bytes
- `missing_duration=0`, `missing_tar=0`
- 전체 239개 Arrow를 다시 열어 센 실제 행 수도 47,662,558로 summary와 일치
- schema 종류 1개, catalog별 샘플의 빈 전사·잘못된 sample rate/channel/duration·없는 tar 경로 0건

MLS English가 10,808,037행·44,659.74시간으로 가장 크므로, 이 source manifest를 그대로 균등 sampling하지 않는다. 후속 품질 승인 뷰에서 source별 시간 cap 또는 sampling weight를 반드시 적용한다.

## 이후 학습 투입 관문

ASR 재전사·정렬 실행 준비와 tar 사전 인덱싱은 [[output-speechlm-asr-alignment-preparation]]에서 추적한다.

이 Arrow는 **형식과 화자 구성 후보를 확정한 source manifest**이지 품질 승인셋은 아니다. 108,513시간 전체를 곧바로 학습한다는 뜻도 아니다. 학습 전에는 현재 파이프라인과 똑같이 다음을 거친다.

1. 기존 train 및 held-out과 locator/PCM hash 중복 제거
2. Qwen·Whisper 전사 합의 QC
3. Qwen 원출력 기반 forced alignment
4. duration·무음·clipping·token density QC
5. source별 시간 상한과 8 kHz 혼합 비율 결정

## 외부 근거

- AIHub 한국어 음성: https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=ty&currMenu=116&dataSetSn=123&topMenu=
- AIHub 저음질 전화망: https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=realm&currMenu=115&dataSetSn=571&topMenu=100
- AIHub 복지 콜센터: https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=&currMenu=&dataSetSn=470&topMenu=
- AIHub 채용면접: https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=&currMenu=115&dataSetSn=71592&topMenu=100
- AIHub 주요 영역별 회의: https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=realm&currMenu=&dataSetSn=464&topMenu=
- MLS 논문: https://www.isca-archive.org/interspeech_2020/pratap20_interspeech.pdf
- Mozilla Common Voice: https://commonvoice.mozilla.org/en/datasets
- Common Voice 논문: https://aclanthology.org/2020.lrec-1.520.pdf

외부 페이지 확인일은 2026-09-22다.
