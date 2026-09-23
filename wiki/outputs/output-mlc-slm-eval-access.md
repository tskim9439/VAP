---
type: output
status: active
created: 2026-09-18
updated: 2026-09-18
summary: MLC-SLM 2025 Eval 정답은 즉시 공개됐지만 대응 음성은 Nexdata 신청·승인이 필요하며 평가 전용 사용 조건이 적용됨
sources:
  - https://huggingface.co/datasets/bsmu/MLC-SLM-Eval
  - https://www.nexdata.ai/competition/mlc-slm
  - '[[output-phase2-independent-evaluation-plan]]'
---

# MLC-SLM Eval 확보 가능성

2026-09-18 확인 결과, 2025 MLC-SLM Eval은 **정답 주석과 음성의 접근 경로가 분리**되어 있다.

- `bsmu/MLC-SLM-Eval`: Eval-1/Eval-2의 oracle segmentation, speaker label, transcription을 공개한다. 저장소는 약 6.55MB이며 API 파일 목록을 확인한 결과 11개 언어와 영어 5개 accent의 `.txt` 224개만 있고 오디오는 없다. 데이터 카드의 라이선스 표시는 CC BY-SA 4.0이다.
- 대응 오디오: 공식 Nexdata 페이지의 sponsored dataset 등록 폼으로 신청해야 한다. 안내상 승인 후 이메일로 다운로드 링크가 오며 최대 7일이 걸린다.
- 오디오 사용 조건: Data Use Agreement에 따라 접근 제어가 필요하고 재배포 및 평가셋을 학습·미세조정에 사용하는 행위가 제한된다. 공개 주석의 라이선스 표시를 오디오까지 자동으로 확장해 해석하지 않는다.
- 현재 서버 `/soundai`·`/lustre` 검색에서는 MLC-SLM 자료를 찾지 못했다.

Phase 2에는 Eval-2가 더 직접적인 객관 평가셋이다. 장시간 대화에 oracle 화자/분할 없이 추론하고 MeetEval의 tcpWER/tcpCER로 채점할 수 있다. 신청이 승인되면 오디오 hash·라이선스 메타데이터를 남기고 locked test로 분리해야 한다.

## VibeVoice 벤치마크와의 관계

VibeVoice-ASR와 VibeVoice-ASR-Streaming 보고서는 평가표에서 split 이름 대신
`MLC-Challenge`라고만 쓴다. 따라서 보고서 문구만으로 `Eval-1`과 `Eval-2`를
합쳐 평가했다고 해석하면 안 된다.

- MLC-SLM 공식 정의상 `Eval-1`은 oracle segmentation·speaker label을 제공하는
  Task 1 ASR 세트이고, `Eval-2`는 이를 제공하지 않는 Task 2 diarization+ASR
  세트다.
- VibeVoice는 `MLC-Challenge`에 대해 DER, cpWER/cpCER, tcpWER/tcpCER를 함께
  보고하며 원음에서 who/when/what을 예측한다. 이 주 평가는 과업 정의상
  `Eval-2`에 대응한다고 보는 것이 타당하다. 다만 논문이 split을 명시하지
  않았으므로 이는 과업·지표에 근거한 판단이다.
- VibeVoice-ASR는 MLC-SLM **training split**을 SFT에 사용했다고 별도로 밝힌다.
  그러므로 평가 오디오를 학습에 사용했다는 뜻은 아니다.

VibeVoice baseline 재현의 필수 대상은 우선 `Eval-2 audio`다. `Eval-1 audio`도
함께 신청하면 oracle 분할 기반 recognition-only 비교와 데이터 무결성 대조에
쓸 수 있으므로, 신청 문구에는 두 세트를 모두 넣는 편이 안전하다.

## Sponsored Dataset 신청 문구

Nexdata 신청 폼의 `Comments (Please list dataset requested)`에는 다음 문구를
사용한다.

> I would like to request access to the audio files for the Interspeech 2025
> Multilingual Conversational Speech Language Model (MLC-SLM) Challenge
> Evaluation Dataset, including both Eval-1 and Eval-2. We already have access
> to the publicly released evaluation annotations and would like to use the
> corresponding audio solely for non-commercial research and evaluation. Our
> purpose is to reproduce the MLC-Challenge benchmark results reported for
> VibeVoice-ASR and VibeVoice-ASR-Streaming, and to evaluate our own streaming
> speaker-attributed ASR system on multilingual conversational speech. The
> evaluation data will not be used for model training, fine-tuning, data
> augmentation, or hyperparameter selection. It will be stored on an
> access-controlled research server, accessed only by authorized project
> members, and will not be redistributed or shared with third parties. We agree
> to comply with the applicable Data License Agreement and any additional usage
> restrictions. Please also provide the download instructions and the license
> or Data Use Agreement that specifically applies to the Eval-1 and Eval-2
> audio files.

근거: [VibeVoice-ASR-Streaming Technical Report](https://arxiv.org/html/2609.02812v2),
[VibeVoice-ASR Technical Report](https://arxiv.org/html/2601.18184),
[MLC-SLM Challenge summary](https://arxiv.org/html/2509.13785).
