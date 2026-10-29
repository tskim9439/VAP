---
type: source
status: active
created: 2026-10-29
updated: 2026-10-29
summary: NVIDIA Nemotron 3.5 Transcribe 기술 보고서(2026-10-28) — 오프라인 SALM(인코더 0.6B + 고정 화자 분리 가지 + 30B-A3B LLM), AMI 회의 통째 cpWER 10.29/17.34, 번역은 단일 화자 오프라인, 스트리밍은 향후 과제
sources:
  - https://research.nvidia.com/labs/nemotron/NVIDIA-Nemotron-3.5-Transcribe/
  - https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf
raw_authors:
  - tskim
---

# Nemotron 3.5 Transcribe 기술 보고서 (NVIDIA, 2026-10-28)

- 2026-10-29 에 읽었다. 사용자 요청은 "이 자료도 참고해서 아이디어를 얻어봐"였다.
  - 67 쪽 PDF 의 본문 전체를 텍스트로 뽑아 읽었다. PDF 는 볼트에 넣지 않았고, 위 URL 에 있다.
- 모델 이름은 Nemotron-3.5-Transcribe-30B-A4B 다. 가중치는 BF16·FP8·NVFP4·GGUF 로 공개됐다(OpenMDW-1.1). 40 개 언어를 다룬다.
- 비교와 아이디어는 [[output-nemotron35-transcribe-comparison-20261029]] 에 정리했다.

## 구조 (2 절, 표 1)
- **오프라인 SALM 이다.** 인코더 출력을 프롬프트의 `<|audio|>` 자리에 넣는다. 오디오 1 초가 LLM 입력 12.5 위치다.
- **Parallel Expert Encoder** 는 두 가지로 이뤄진다.
  - ASR 가지: Transformer 32 층·1280 차원, 약 0.6B, 80 ms 프레임이다.
    - TDT 로 미리 학습한 다국어 인코더에서 시작했다.
    - 학습 때는 30 s 청크마다 따로 돈다(청크 사이 어텐션 없음).
  - 화자 분리 가지: Nemotron 3 Diarization(스트리밍 Sortformer, 약 100M, 최대 8 명)이다. **학습 내내 고정한다.**
  - 주입 방식:
    - 활동 확률을 0.5 로 이진화하고 정규화한다.
    - **고정 사인 커널(8 × 1280)** 로 프레임마다 화자 임베딩을 만든다.
    - 이것을 정규화한 ASR 표현에 더한 뒤 Linear(1280 → 2688) 하나로 LLM 에 넣는다.
  - 학습 예에 정답 화자 구간이 있으면 화자 분리 예측 대신 정답 구간을 넣는다.
  - 학습 때 화자 분리 가지도 30 s 청크로 돌아서 화자 번호가 청크마다 따로다.
    - 녹음 전체에서 화자 표지를 일관되게 유지하는 일은 LLM 이 맡는다.
  - 추론 때는 40 s 중심 청크에 좌우 4 s 문맥을 붙인다. 화자 분리 가지는 화자 캐시와 FIFO 를 청크 사이에 이어 간다.
- **LLM** 은 Nemotron 3.5 Lightning 30B-A3B 다.
  - Mamba-2 23 층, MoE 23 층, 어텐션 6 층이다. 전문가는 128 개이고 top-6 이다.
  - 모델 전체는 34B, 활성 파라미터는 4.3B 다.
- **MTP**: 깊이 2, 공유 층 하나(1.3B)를 함께 학습했고, 추측 디코딩의 초안 모델로도 쓴다.
- **단어 시각**: 디코드가 끝난 뒤 CTC 어댑터(50M)로 강제 정렬한다.

## 학습 (3 절)
- **다화자 학습**:
  - 형식은 SOT 다. 턴마다 `<spk:k>` 를 쓰고 녹음 내내 같은 표지를 쓴다.
  - 데이터: Fisher·AMI·ICSI·DiPCo·AISHELL-4·AliMeeting 과 MLS 모의 혼합이다.
  - 길이는 몇 초부터 회의 통째까지다(최대 1 h, 65,536 토큰). 화자는 최대 8 명이다.
- **섞음 가중**:
  - 시간이나 예 수가 아니라 **답(감독) 토큰 양**으로 정하고, 온도 T = 0.5 로 뽑는다.
  - 번역 데이터는 크기에 맞는 가중의 75 % 만 쓴다. 영어로 가는 번역은 번역의 20 % 로 묶었다.
    - 초기 체크포인트가 다국어 ASR 데이터를 번역해 버렸기 때문이다.
- **RL**: 끝에 짧게 했다. 보상은 1 − WER 이다.
  - 화자 귀속 예는 화자 표지까지 포함한 직렬 전사의 WER 을 쓴다.
  - 2 만 예 가운데 화자 귀속 예는 201 개뿐이다.
- **생성 반복**: 후처리로 접는다(길게 정확히 반복되는 부분을 하나로).
  - 중국어 회의는 이 후처리가 없으면 cpCER 이 AISHELL-4 32.91 → 45.55, AliMeeting 34.95 → 55.37 이다.

## 결과
### 화자 귀속 전사 (표 13·14, 회의 통째, cpWER %)

| 시스템 | AMI IHM-mix | AMI SDM | CH109 | Mixer 6 | NOTSOFAR IHM | NOTSOFAR SC | 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nemotron 3.5 Transcribe | 10.29 | 17.34 | 11.50 | 14.80 | 14.94 | 25.91 | 15.80 |
| DiCoW v3.3 + DiariZen-v2 | 14.48 | 18.24 | 10.80 | 14.74 | 17.91 | 24.80 | 16.83 |
| Multi-talker Parakeet + Nemotron 3 Diarization | 13.38 | 20.33 | 13.00 | 16.92 | 18.19 | 30.42 | 18.71 |
| MOSS-Transcribe-Diarize | 15.40 | 24.72 | 10.72 | 13.25 | 22.52 | 28.92 | 19.26 |
| VibeVoice-ASR | 22.13 | 32.12 | 12.08 | 16.45 | 25.52 | 37.21 | 24.25 |

- 모든 시스템을 같은 데이터·채점으로 쟀다. 정규화는 Open ASR Leaderboard 방식이다.
- 비교한 네 시스템의 지연 설정은 표에 없다.
- 조각 길이에 거의 둔감하다. AMI IHM-mix 는 10–20 s 10.80, 60–90 s 9.77, 120–180 s 10.05, 통째 10.29 다.
- LibriSpeechMix 1·2·3 명은 1.58·4.14·8.65 다(이 셋으로는 학습하지 않았다).
- 화자 분리 품질에 대한 민감도(예측 활동 대 정답 활동) 절제는 보고하지 않았다.

### 번역 (4.10 절, 부록 표 23, BLEU)
- 오프라인·단일 화자 과제다. 지시문 "Please translate the given speech to {target language}." 로 목표 언어 하나를 정한다.
- FLEURS 평균은 X→영 29.20, 영→X 26.54 다. CoVoST 2 는 33.67, 26.34 다.
- 우리 언어쌍(FLEURS)은 독→영 39.96, 영→독 33.93, 한→영 24.23, 영→한 27.68 이다.
- 화자 귀속 전사와 번역을 묶은 평가는 없다.

### 속도
- vLLM 으로 H100 한 장에서 MTP 추측 디코딩을 쓴다.
- 10 분 녹음 단일 요청이 NVFP4 에서 2.9 s 걸린다.

## 한계 (보고서가 직접 적은 것)
- 공개 모델과 평가는 오프라인 전사다. 스트리밍 전사는 지원하지 않으며 향후 과제로 남겼다.
- 화자는 최대 8 명이다.
- 중국어 원거리 회의는 약하다(학습한 중국어 다화자 데이터가 거의 없다).

## 관련 페이지
- 외부 기준 표: [[output-ami-external-comparison-20261027]]
- 화자 분리 통합: [[output-diarization-integration-20261025]]
