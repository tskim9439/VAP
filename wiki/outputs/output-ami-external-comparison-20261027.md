---
type: output
status: active
created: 2026-10-27
updated: 2026-10-30
summary: AMI cpWER 외부 비교 — 직접 재면 Parakeet + N3 Diarization 15.9·20.5(보고 재현) 대 우리 E11 30.5·43.5(재귀속 23.1·29.6), VibeVoice 7B 화자 쏠림
sources:
  - '[[output-diarization-integration-20261025]]'
  - '[[output-vibevoice-stream-comparison-20261020]]'
  - '[[source-vibevoice-asr-streaming-iclr2027]]'
  - '[[source-nemotron-3-5-transcribe-tech-report]]'
raw_authors:
  - tskim
---

# AMI 외부 비교: 우리 대 Multitalker Parakeet · VibeVoice-ASR (2026-10-27)

사용자 요청은 "https://openreview.net/attachment?id=r6QZahgnJY&name=pdf 이것과 VibeVoice-ASR, "Multitalker Parakeet SDM" 과는 비교해서는 어떠한지 정리해줘"였다.
- OpenReview 문서는 **VibeVoice-ASR-Streaming 의 ICLR 2027 투고본**이다. 사용자가 2026-10-28 `raw/inbox/` 에 올렸다 → [[source-vibevoice-asr-streaming-iclr2027]].
  - AMI 수치는 arXiv 2609.02812 와 같다.
  - 사용자가 말한 "1.7 B 모델"은 문서에서 1.5B(Qwen2.5-1.5B 백본)다.
- 외부 수치는 2026-10-27 원문(HF 카드 raw README, arXiv HTML 표)에서 직접 뽑았다.

## 비교표 (AMI eval cpWER %, 낮을수록 좋다 — 우리 행은 2026-10-30 기준)
- 이 표를 처음 만든 2026-10-27 에는 E6 이 최신이라 E6 창 수치를 실었다.
- E11(고정 N3 Diarization 활동, 2026-10-30)부터는 외부 수치와 같은 조건(회의 통째, 표준 정규화 + meeteval)으로 잰 E11 로 견준다. E6 이 나오는 아래 절들은 그때의 기록이다.
- 스트리밍과 오프라인으로 나누고, 같은 평가 단위끼리 묶어 IHM(-mix) 오름차순으로 정렬했다(2026-10-30, 사용자 "표를 Offline/Streamable 을 구분하고 끼리끼리 뭉쳐서 정렬해서 쉽게 볼 수 있도록 정리해줘"). **굵은 행**이 우리 모델이다. 우리 모델 전체는 [[output-ami-results-summary-20261030]] 에 있다.

### 스트리밍 — 회의 통째·세션
| 시스템 | 크기 | 지연 | 화자 분리 | IHM(-mix) | SDM | 출처·조건 |
|---|---|---|---|---:|---:|---|
| SSA-v2 + Nemotron 3 Diarization(arXiv 2609.10265 표 1) | 0.6B × 화자 수 + 100M | 스트리밍(표에 지연 미기재) | Nemotron 3 Diarization | 13.38 | 20.33 | 세션. 정답 화자 구간이면 12.41 · 17.44 |
| Multitalker Parakeet v1 + N3 Diarization(우리 틀로 직접 잼) | 0.6B × 화자 수 + 100M | 1.12 s | Nemotron 3 Diarization | 15.86 | 20.50 | 회의 통째, 표준 — 아래 "직접 잰 결과" |
| Multitalker Parakeet v1 + Streaming Sortformer v2(우리 틀로 직접 잼) | 0.6B × 화자 수 + 117M | 1.12 s | Streaming Sortformer v2 | 17.59 | 34.59 | 같음 |
| Multitalker Parakeet v1(카드) | 0.6B × 화자 수 + 117M | 1.12 s | Streaming Sortformer v2 | 21.26 | 37.44 | HF 카드 — split·정규화·세션 여부 미기재 |
| SSA-v1(arXiv 2609.10265 표 1) | 0.6B × 화자 수 + 117M | 스트리밍(표에 지연 미기재) | Sortformer v2.1 | 22.10 | 39.91 | 세션. 정답 화자 구간이면 15.22 · 20.27 |
| **우리 E11 + 출력 재귀속** | 인코더 0.6B + LLM 1.7B | δ10 평균 0.92 s | 고정 N3 Diarization 활동 + 출력 재귀속(학습 없는 후처리) | **23.07** | **29.62** | 회의 통째, 표준 정규화 + meeteval — [[output-n3d-branch-20261029]] 6.7 절 |
| **우리 E11** | 인코더 0.6B + LLM 1.7B | δ10 평균 0.92 s | 고정 N3 Diarization 활동 | **30.50** | **43.48** | 같음 — 같은 문서 6.3 절 |

### 스트리밍 — 480 s 이하 조각(VibeVoice-Streaming 논문 조건: 녹음을 겹치지 않는 480 s 이하 조각으로 나눠 잰다)
| 시스템 | 크기 | 지연 | 화자 분리 | IHM(-mix) | SDM | 출처·조건 |
|---|---|---|---|---:|---:|---|
| VibeVoice-ASR-Streaming 7B, 22 프레임(OpenReview 표 2 = arXiv 2609.02812) | 8.7B | 기대 평균 2.0 s(첫 출력 3.5 s) | 종단(모델이 직접) | 27.48 | 39.01 | WER 19.83 · 29.81 |
| Azure Conversation Transcriber(같은 표 2) | 서비스 | 실측 8.21–9.12 s | 서비스 | 31.90 | 41.40 | WER 24.01 · 33.27 |
| VibeVoice-ASR-Streaming 7B, 15 프레임(표 5) | 8.7B | 1.53 s | 종단 | 35.40 | 42.49 | WER 20.82 · 31.88 |
| VibeVoice-ASR-Streaming 1.5B, 22 프레임(표 5) | LLM Qwen2.5-1.5B | 2.00 s | 종단 | 45.15 | 56.83 | WER 22.85 · 34.43 |
| VibeVoice-ASR-Streaming 1.5B, 15 프레임(표 5) | LLM Qwen2.5-1.5B | 1.53 s | 종단 | 46.48 | 59.49 | WER 23.39 · 35.90 |
| VibeVoice-ASR-Streaming 7B, 22 프레임(우리 틀로 직접 잼) | 8.7B | 2.0 s | 종단 | 48.17 | 43.16 | 표준 — 아래 "직접 잰 결과" |

- 우리 모델은 이 조건으로 재지 않았다. 더 어려운 회의 통째로 잰 값은 위 묶음에 있다.

### 스트리밍 — 20–90 s 창(우리 모델만, 2026-10-27 기록)
| 시스템 | 크기 | 지연 | 화자 분리 | IHM(-mix) | SDM | 출처·조건 |
|---|---|---|---|---:|---:|---|
| **우리 E5(정답 활동 입력, 상한)** | 인코더 0.6B + LLM 1.7B | δ10 평균 0.92 s | 정답 | 23.39 | 29.24 | 창, 우리 정규화 |
| **우리 E6 final** | 인코더 0.6B + LLM 1.7B | δ10 평균 0.92 s | 인과 헤드(예측) | 37.68 | 45.35 | 창, 우리 정규화. 회의 통째(E7a 헤드 + AOSC)는 67.34·73.10 |

- 창은 창마다 화자를 짝지어 회의 통째·세션보다 쉬운 조건이다. 외부 수치와 바로 견주지 않는다.

### 오프라인
| 시스템 | 크기 | 지연 | 화자 분리 | IHM(-mix) | SDM | 출처·조건 |
|---|---|---|---|---:|---:|---|
| Nemotron 3.5 Transcribe(N3.5T 보고서 표 14, 2026-10-29 추가) | 34B(활성 4.3B) | 오프라인(40 s 청크 + 좌우 4 s) | 고정 N3 Diarization 가지(한 체크포인트) | 10.29 | 17.34 | 회의 통째, Open ASR 정규화 — [[source-nemotron-3-5-transcribe-tech-report]] |
| DiCoW-v3.3(2609.10265 표 1) | Whisper 계열 | 오프라인 | 예측(DiariZen-v2) | 14.48 | 18.24 | 세션. 정답 화자 분리면 11.33 · 14.78 |
| MOSS-Transcribe-Diarize(N3.5T 보고서 표 14) | – | 오프라인 | 종단 | 15.40 | 24.72 | 회의 통째, NVIDIA 평가 |
| VibeVoice-ASR 7B(arXiv 2601.18184 표 2) | 8.7B | 오프라인(60 분 한 번에) | 종단 | 20.41 | 28.82 | 자체 평가. WER 18.81 · 24.65, DER 11.92 · 13.43 |
| VibeVoice-ASR 비스트리밍(Streaming 논문 표 4 — 스트리밍 학습의 시작점) | – | 오프라인 | 종단 | 20.98 | 33.88 | 480 s 이하 조각. WER 19.08 · 26.28 |
| VibeVoice-ASR 7B(N3.5T 보고서 표 14) | 8.7B | 오프라인 | 종단 | 22.13 | 32.12 | 회의 통째, NVIDIA 평가 |

### 같은 시스템인데 출처마다 다른 값
- **VibeVoice-ASR 오프라인 SDM:** 28.82(자체), 42.19(NVIDIA 평가 셋), 32.12(N3.5T 보고서 표 14, 회의 통째), 28.42(arXiv 2609.15218, 2·5 분 조각 평균), 36.09(제3자 재현, GitHub VibeVoice issue #237).
  - 채점 조건에 따라 14 pt 가까이 흔들린다.
- **Parakeet 카드 IHM 21.26:** 제3자 재현(transcribe.cpp — AMI IHM test 16 회의, meeteval, 회의 통째)의 NeMo 참조 21.39 와 맞는다.
  - 2609.10265 는 IHM 을 "Individual Headset Mix"(= 우리 IHM-mix)로 정의한다.
- **SSA-v2 + Nemotron 3 Diarization(13.38 / 20.33)** 은 N3.5T 보고서 표 14 에서 "Multi-talker Parakeet + Nemotron 3 Diarization" 으로 같은 값이 나온다. DiCoW-v3.3 + DiariZen-v2(14.48 / 18.24)도 두 문서가 같다.
- **카드의 평가 세션 수 표**(AMI IHM 219 등)는 화자 분리 모델 카드의 DIHARD·CALLHOME 수를 옮겨 온 것으로 보여 근거로 쓰지 않는다.

## 같은 LLM 크기: 우리 1.7B 대 VibeVoice-ASR-Streaming 1.5B (2026-10-28)
- **2026-10-30 갱신:** 우리 E11 은 더 어려운 회의 통째(표준)로 30.50·43.48 이다.
  - 1.5B(480 s 조각, 보고값 45.15·56.83)보다 IHM-mix 14.7·SDM 13.4 낮다. 출력 재귀속(23.07·29.62)이면 22.1·27.2 낮다.
  - 아래 표는 2026-10-28 의 E6 창 기준 기록이다.

| AMI(eval) | 우리 E6 final(LLM 1.7B, δ10 평균 0.92 s) | VibeVoice-Streaming 1.5B(2.00 s) | VibeVoice-Streaming 7B(2.00 s) |
|---|---:|---:|---:|
| IHM(-mix) cpWER | 37.68(표준 정규화 37.09) | 45.15 | 27.48 |
| SDM cpWER | 45.35(표준 정규화 44.73) | 56.83 | 39.01 |
| IHM(-mix) WER(우리 agn) | 22.54(21.49) | 22.85 | 19.83 |
| SDM WER(우리 agn) | 28.06(27.16) | 34.43 | 29.81 |

- **비슷한 크기에서는 우리가 앞선다.** cpWER 가 SDM 에서 12, IHM 에서 8 낮고, 지연은 절반 아래다.
- **받아쓰기:** SDM 은 7 pt 낫고 IHM 은 비슷하다(0.3–1.4 낫다).
- **조건 차이:**
  - 우리 수치는 창(20–90 s, 창마다 화자 짝) cpWER 이고, 그쪽은 480 s 이하 조각이다. 조각이 길수록 화자 귀속이 어렵다.
  - 회의 통째 수치(96172 E7a·96461 E6 기준선)가 나오면 이 표를 다시 맞춘다.
- **7B 와 견주면 SDM cpWER 가 6 높다.** VibeVoice 쪽은 1.5B → 7B 에서 cpWER 가 12.8 pt(다섯 셋 평균) 내렸다. 화자 귀속이 크기에 민감하다.

## 회의 통째(세션) 첫 결과 — 2026-10-28(E6 기준, 지난 기록)
| AMI eval cpWER(표준 정규화 + meeteval) | IHM(-mix) | SDM | 평가 단위 |
|---|---:|---:|---|
| 우리 E6 LLM + E7a 헤드 + AOSC(δ10, 0.92 s) | 67.34 | 73.10 | 회의 통째(34 회의) |
| 우리 E6(같은 모델, 창) | 37.09 | 44.73 | 20–90 s 창 |
| VibeVoice-ASR-Streaming 1.5B(2.0 s) | 45.15 | 56.83 | 480 s 이하 조각 |
| VibeVoice-ASR-Streaming 7B(2.0 s) | 27.48 | 39.01 | 480 s 이하 조각 |
| Multitalker Parakeet v1(1.12 s) | 21.26 | 37.44 | 세션(제3자 재현은 회의 통째) |
| NVIDIA SSA-v2 + Nemotron 3 Diarization | 13.38 | 20.33 | 세션 |
| Nemotron 3.5 Transcribe(오프라인, 2026-10-29 추가) | 10.29 | 17.34 | 회의 통째 |

- 회의 통째에서 우리 귀속이 무너져(46 pt) 창 단위의 우위가 사라졌다. 받아쓰기(agn IHM 21.1·SDM 27.4)는 그대로다. 원인과 대책은 [[output-diarization-integration-20261025]] 7 절 결과 3 에 있다.

## 회의 통째 — 고정 화자 분리 전문가(E11) 결과 (2026-10-30)
| AMI eval cpWER(표준 정규화 + meeteval) | IHM(-mix) | SDM | 평가 단위 |
|---|---:|---:|---|
| **우리 E11**(E9 → Nemotron 3 Diarization 활동으로 학습, δ10, 0.92 s) | **30.50** | 43.48 | 회의 통째(16 회의) |
| 우리 E9 + 전문가(학습 없음) | 33.32 | **42.75** | 회의 통째 |
| 우리 E9 · E7a 헤드 + AOSC | 53.47 | 66.35 | 회의 통째 |
| VibeVoice-ASR-Streaming 1.5B(2.0 s) | 45.15 | 56.83 | 480 s 이하 조각 |
| VibeVoice-ASR-Streaming 7B(2.0 s) | 27.48 | 39.01 | 480 s 이하 조각 |
| Azure Conversation Transcriber(8.2–9.1 s) | 31.90 | 41.40 | – |
| Multitalker Parakeet v1(1.12 s) | 21.26 | 37.44 | 세션 |
| NVIDIA SSA-v2 + Nemotron 3 Diarization | 13.38 | 20.33 | 세션 |
| Nemotron 3.5 Transcribe(오프라인) | 10.29 | 17.34 | 회의 통째 |

- 화자 활동 출처를 고정 전문가(Nemotron 3 Diarization)로 바꾸자 회의 통째 귀속 붕괴가 거의 사라졌다(E9 대비 eval −20–24). 자세한 결과는 [[output-n3d-branch-20261029]] 6 절에 있다.
- 같은 크기의 VibeVoice-Streaming 1.5B 보다 13–15 낮다. 7B 에는 3.0·4.5, Parakeet v1 에는 9.2·6.0 뒤진다.
- 받아쓰기(Whisper 정규화 agn)는 IHM-mix 20.64·SDM 26.44 로 VibeVoice 7B WER(19.83·29.81)과 비슷하다. 남은 차이는 귀속이다(cp − agn 9.9·17.0 대 7.7·9.2).

## 같은 조건으로 직접 재기 (2026-10-30, 준비)
사용자: "지금 우리가 사용하고 있는 베이스라인 모델들의 성능 수치는 그냥 논문에 레포트된 수치이지?" → 맞다. 위 표의 외부 수치는 모두 남이 보고한 값이다 → "이 둘을 준비해줘".
- **도구** — `experiments/baseline_ami_ext.py`
  - 우리 회의 통째 평가(`v3_eval_task.sh amifull`, δ10)와 같은 회의·오디오(mono 혼합 16 kHz)·참조(전사 없는 구간 규칙까지)·채점이다.
  - 행 꼴이 `cst_stage3_eval` 과 같아 `summarize`·`ami_report.py`·`meeteval_cpwer.py` 를 그대로 쓴다.
  - 가설 단어도 참조와 같은 규칙으로 마스크한다(시각 t 의 단어는 청크 int(t / 80 ms) + δ 가 전사 없는 구간이면 뺀다).
- **시스템**(작업 `raw/sources/experiments/2026-10-30-ami-ext/ami_ext_task.sh`, 묶음 `ext-vibevoice`·`ext-parakeet`, 각 8 몫)
  - VibeVoice-ASR-Streaming-7B(@60d858b, 서버에 있음)
    - `vv7b-full`: 회의 통째(우리 조건). LLM 문맥이 131k 위치라 50 분 회의도 한 번에 들어간다.
    - `vv7b-seg480`: 480 s 이하 조각마다 따로(논문 조건 — 보고값 27.48 / 39.01 재현 점검). 조각 끝은 480 s 전 60 s 안의 가장 늦은 무음 지점이다.
  - Multitalker Parakeet v1(@8749fc7, NVIDIA Open Model License)
    - NeMo 3.1.0 설치본(6989ae9fc)에 있는 다화자 모델·병렬 스트리밍 도구로 돈다. 공식 예제의 manifest 경로·설정 기본값을 옮겼다(1.12 s, `att_context_size [70,13]`).
    - `mtp-sortv2`: + Streaming Sortformer v2(@84edd51, CC-BY-4.0) — 카드 조건(보고값 21.26 / 37.44 재현 점검).
    - `mtp-n3d`: + Nemotron 3 Diarization — 우리 E11 과 같은 화자 분리기(캐시 264·264·222).
- **확인**
  - 맥 CPU 배선 점검(AMI ES2002a 38 s 클립): 두 화자 분리기 모두 4 명을 가르고 받아썼다. N3D 쪽이 첫 "Um well"·맞장구 "Mm-hmm" 까지 잡았다.
  - 순수 함수(조각 경계·7 명 이상 헝가리안 채점·화자 묶기)를 손으로 대조했다.
  - 외부 기준 행의 참조가 우리 E11 회의 통째 행과 같다(IS1009a SDM: 참조 단어 1,944 · 마스크 청크 434 · 화자 순서까지).
- **VibeVoice 7B 서버 스모크**(호스트 GPU 4·6, AMI SDM IS1009a 14 분 — `tmp/e11-check/ext`)

| | cp(우리 정규화) | meeteval(Whisper) | agn | 가설 화자 |
|---|---:|---:|---:|---|
| 회의 통째 | 83.28 | 82.32 | 30.30 | "Speaker 1" 1,876 · "Speaker 0" 12 단어 |
| 480 s 조각 둘 | 77.57 | 77.22 | 28.86 | 조각마다 "Speaker 1" 853·939 · "Speaker 0" 8·80 |
| (참고) 우리 E11 같은 회의 | 35.19 | – | 26.34 | 4 명 |

  - **받아쓰기는 논문과 맞고(SDM WER 29.81) 화자 귀속이 무너졌다.** 4 명 회의의 말을 거의 모두 한 화자로 냈다. "Speaker 0" 은 첫 비음성("Footsteps…")과 일부 말에만 썼다.
  - 호출은 공식 `demo/vibevoice_asr_streaming_inference_from_file.py` 와 같다(체크포인트의 청크 2.93 s·lookahead 0.53 s, 온도 0, 청크당 256 토큰, sdpa, bf16). 공식 로더(ffmpeg)도 음량을 정규화하지 않는다.
  - 디코드는 14 분 회의에 110.5 s(실시간 0.13 배)다. 묶음 전체는 8 GPU 에서 한 시간 남짓으로 어림한다.
  - 같은 회의 IHM-mix 회의 통째도 cp 77.42·agn 22.89·화자 2 명 놓침으로 같았다.
  - 확인한 것:
    - 체크포인트 @60d858b 는 HF 최신이다(가중치는 2026-09-02 한 번). 모델 코드(`modeling_vibevoice_asr.py`)의 마지막 변경(2026-08-27 "Add streaming ASR inference")도 소스 16fb2cb 에 들어 있다.
    - 입력 음량은 정상이다(IS1009a RMS SDM −31 dBFS·IHM-mix −25.5 dBFS).
    - 논문: "평가 녹음은 공개 체크포인트가 지원하는 최대 세션 길이에 맞춰 480 초로 제한"하고, 화자 수 같은 부가 정보는 주지 않는다. 회의 통째는 사용 범위 밖이고 480 s 조각이 논문 조건이다.
  - 회의 하나라 일반화하지 않는다. 묶음 `ext-vibevoice` 의 480 s 조각 단계가 전 회의에서 쏠림이 일반적인지 보여 준다.
- **사용자 할 일**: T5 `vapkt-models/` 의 두 .nemo 를 서버 `VAPKT-data/vapkt-baselines/` 아래로 올린다(azcopy, 사내망).
  - Parakeet → `vapkt-baselines/multitalker-parakeet-v1/`(2.49 GB, sha256 `afaefe89…`). T5 폴더 이름과 다르다 — 원래 이름은 업로드 전에 서버에서 조회해 NFS 에 "없음"이 남았을 수 있다.
  - `diar_streaming_sortformer_4spk-v2/`(471 MB, sha256 `b371afce…`)
  - 그다음 `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-evals.sh ext-parakeet)`. VibeVoice 는 바로 낼 수 있다: `… ext-vibevoice`.
  - 2026-10-30 사용자가 올렸다(서버 크기·sha256 이 HF 와 같다). `ext-parakeet` 은 job 98631(16 작업, 11:06 시작)이다.

## 직접 잰 결과 — Multitalker Parakeet (2026-10-30, job 98631)
| AMI 회의 통째, 표준 정규화 + meeteval cpWER | 지연 | IHM-mix eval | SDM eval | IHM-mix dev | SDM dev | 보고값(IHM / SDM) |
|---|---|---:|---:|---:|---:|---|
| Multitalker Parakeet v1 + Streaming Sortformer v2 | 1.12 s | 17.59 | 34.59 | 14.57 | 23.47 | 카드 21.26 / 37.44 |
| **Multitalker Parakeet v1 + Nemotron 3 Diarization** | 1.12 s | **15.86** | **20.50** | 13.01 | 17.60 | NVIDIA(같은 조합) 13.38 / 20.33 |
| 우리 E11(같은 N3 Diarization 활동) | 0.92 s | 30.50 | 43.48 | 29.74 | 36.51 | – |
| 우리 E9 + 전문가(학습 없음) | 0.92 s | 33.32 | 42.75 | 34.18 | 38.91 | – |
| 우리 E11 + 출력 재귀속(학습 없음, 2026-10-30) | 0.92 s | 23.07 | 29.62 | 21.21 | 26.54 | – |
| 우리 E9 + 전문가 + 출력 재귀속 | 0.92 s | 23.53 | 29.28 | 21.81 | 26.48 | – |

- 산출: `results/ext-ami-mtp-sortv2`·`ext-ami-mtp-n3d`(행 68 개씩 — 참조는 우리 E11 회의 통째 행과 같다). 8 GPU 에서 두 조합 모두 약 15 분이다.
- **우리 평가 틀이 보고값을 재현한다.** N3 Diarization 조합 SDM 20.50 은 NVIDIA 보고 20.33 과 같다. Sortformer v2 조합은 카드보다 3–4 낮다(참조·정규화 차이로 본다).
- **같은 화자 분리기(N3 Diarization)를 써도 Parakeet 가 우리보다 IHM-mix 14.6·SDM 23.0 낮다.** 차이는 이제 화자 분리 품질이 아니라 활동을 쓰는 방식과 받아쓰기에서 온다.
  - Parakeet 는 화자마다 인스턴스를 두고 그 화자 활동을 인코더에 주입해 따로 디코드한다(화자 수 × 0.6B). 우리는 한 스트림 LLM 이 슬롯을 고른다.
  - 받아쓰기도 뒤진다. Parakeet 의 SDM cpWER(20.50)이 우리 SDM 받아쓰기 오류(Whisper 정규화 agn 26.44)보다 낮다.
- **출력 재귀속(2026-10-30) 뒤 차이는 IHM-mix 7.2·SDM 9.1 로 줄었다.**
  - 재귀속은 활동을 어긴 단어만 활동 슬롯으로 옮기는 학습 없는 후처리다. 스트리밍 출력에 그대로 붙는다([[output-n3d-branch-20261029]] 6.7 절).
  - 귀속 몫이 2–3 pt 만 남아 남은 차이는 대부분 받아쓰기다. Sortformer v2 조합보다는 SDM 이 5.0 낮다.
- dev 는 N3 Diarization 이 AMI dev 로 학습해 섞여 있다. 판단은 eval 로 한다.
- 외부 시스템의 agn(시각순 화자 무시)은 비교에 쓰지 않는다. 단어 시각을 SegLST 문장 구간 안에 고르게 놓은 근사라 겹침 구간 순서가 틀어져 부풀려진다(cp 가 agn 보다 낮은 회의가 나온다). cp·meeteval cpWER 은 시각과 무관하다.
- 평가 노드의 로컬 env 복사본에 meeteval 이 없어 묶음의 meeteval 단계가 실패했다. 서버 호스트에서 NFS env 로 다시 매겼고, 작업 스크립트는 meeteval 만 NFS env 로 돌게 고쳤다.

## 직접 잰 결과 — VibeVoice-ASR-Streaming 7B (2026-10-30, job 98672)
| 480 s 조각(논문 조건), 표준 정규화 + meeteval cpWER | IHM-mix eval | SDM eval | IHM-mix dev | SDM dev | 논문 보고(IHM / SDM) |
|---|---:|---:|---:|---:|---|
| VibeVoice-ASR-Streaming 7B(우리 틀) | 48.17 | 43.16 | 50.80 | 43.97 | 27.48 / 39.01 |
| 받아쓰기(agn, 우리 정규화) | 22.03 | 30.77 | 22.87 | 29.10 | WER 19.83 / 29.81 |

- 산출 `results/ext-ami-vv7b-seg480`(조각 318 개)·`ext-ami-vv7b-full`(회의 통째).
- **회의 통째(사용 범위 밖)**: 표준 cpWER IHM-mix eval 82.07·SDM 63.14(dev 60.75·78.80), agn(우리 정규화) 22.36·31.55.
  - 받아쓰기는 조각과 같다. 긴 회의에서는 화자를 놓친다(화자 놓침 eval IHM-mix 23·SDM 10, dev 14·20). IS1009a 스모크와 같은 꼴이다.
  - 논문 조건이 아니므로 비교에는 480 s 조각을 쓴다.
- **받아쓰기는 논문과 맞고 화자 귀속이 차이 난다.** 가설 화자는 조각마다 평균 2.7–3.2 명(참조 4 명)이고, 화자 놓침이 IHM-mix eval 96·SDM eval 58 이다. 최다 화자 단어 몫은 중앙값 0.51–0.69, 0.9 이상인 조각이 IHM-mix eval 76 개 중 26 개다.
- **SDM 은 논문보다 4.2 높고 IHM 은 20.7 높다.** IHM 차이가 유독 크다.
- **"논문 AMI-IHM 이 다른 입력" 가설 검토(2026-10-30, 사용자 요청)** — 기각에 가깝다.
  - 논문 본문: Google STT 기준선에 "녹음마다 16 kHz mono, 최대 화자 수 = 그 녹음의 참조 화자 수"를 준다 → AMI-IHM 도 여러 화자가 섞인 한 채널이다. IHM cpWER(27.48)이 WER(19.83)보다 7.6 높은 것도 다화자 입력과 맞다.
  - 평가 범위 "AMI-IHM·SDM·MLC 영어 30.80 h, 단어 256,419" 는 AMI 테스트 회의 전체를 480 s 이하로 나눈 것과 맞다(회의마다 앞 480 s 만이면 AMI 가 4 h 남짓이라 맞지 않는다). 비용 표의 "237 s AMI-IHM 녹음"도 나눈 마지막 조각으로 보인다.
  - 혼합 방식 대조(헤드셋 4 채널을 그대로 더한 오디오, `--audio headset-sum`, IHM-mix 4 회의 480 s 조각 18 개 — 호스트 GPU 2): cp 39.38 대 공식 Mix-Headset 41.33, agn 24.58 대 24.05. 차이 2 pt 로 논문과의 21 pt 를 설명하지 못한다.
  - **대신 화자 표지가 불안정하다.** 같은 조각이 입력 혼합만 조금 달라도 화자를 가르다가 한 명으로 몰리기를 오간다(예: EN2002c@957 17.1 ↔ 59.1, IB4001@480 68.4 ↔ 14.5). 회의 첫 조각은 두 입력 모두 자주 무너진다(ES2004d@0·IS1009a@0 cp 80 안팎, 가설 1–2 명).
  - 정리: 공개 체크포인트·공식 추론 코드로 논문 IHM 수치는 재현되지 않는다. 입력 종류·혼합 방식으로는 설명되지 않고, 화자 표지가 작은 입력 차이에 흔들릴 만큼 불안정하다.
- **Flash Attention 대조(2026-10-30, 사용자 "별도 env 를 만들어 대조해보자")** — 이것으로도 설명되지 않는다.
  - 별도 env `VAPKT-data/baselines/vibevoice-fa-v1/venv`: vapasr env 위 venv 에 torch 2.8.0 + 공식 flash-attn 2.8.3.post2 휠(`raw/sources/experiments/2026-10-30-ami-ext/setup_vibevoice_fa_env.sh`). 학습 env 는 그대로다.
  - 커널 점검: flash_attn_func 대 sdpa 최대 차 0.002(bf16).
  - 같은 IHM-mix 조각 18 개(4 회의, 호스트 GPU, `baseline_ami_ext.py --attn`):

| 설정 | cp(우리 정규화) | agn |
|---|---:|---:|
| torch 2.9 + sdpa(클러스터 결과) | 41.33 | 24.05 |
| torch 2.8 + sdpa | 36.60 | 24.37 |
| torch 2.8 + Flash Attention 2 | 42.07 | 24.34 |

  - 받아쓰기는 그대로고 cp 만 흔들린다. 몇 조각이 화자를 가르다(cp 15–27) 한 명으로 몰리기(cp 67–88)를 오간다. 설정 쌍마다 2–4 조각이고 방향도 제각각이다(예: IB4001@480 14.7 ↔ 68.4, IS1009a@479.84 25.2 ↔ 72.1).
  - Flash Attention 이 결과를 논문 쪽으로 옮기지 않는다. 논문 IHM 27.48 과의 차이는 커널·torch 판으로 설명되지 않는다.
- 같은 크기대 비교가 아니라도, 우리 E11 은 더 어려운 회의 통째 조건에서 IHM-mix 30.50·SDM 43.48 이다(SDM 은 비슷, IHM-mix 는 우리가 낮다).

## 받아쓰기 몫(화자 무시)
| | IHM(-mix) | SDM |
|---|---:|---:|
| 우리 E11 agn(회의 통째, Whisper 정규화, 2026-10-30) | 20.64 | 26.44 |
| VibeVoice-ASR-Streaming 7B agn(우리 틀로 직접 잼, 480 s 조각, 우리 정규화) | 22.03 | 30.77 |
| 우리 E6 agn(창, 2026-10-27) | 22.54 | 28.06 |
| 우리 E5 agn(정답 활동, 창) | 22.40 | 27.84 |
| VibeVoice-ASR-Streaming WER(2.0 s) | 19.83 | 29.81 |
| VibeVoice-ASR 오프라인 WER | 18.81 | 24.65 |

- NVIDIA 는 정답 화자 구간에서 SDM cpWER 17.4–20.3 이다. 그쪽 받아쓰기 오류는 그보다 낮다는 뜻이고, 우리 agn 28 보다 한참 낮다.
- 다만 정규화가 다르다.
  - 우리 영어 채점(`vapasr.data.textnorm.score_en`)은 소문자·문장부호·숫자만 정리하고 채움말(um·uh·mm-hmm)을 그대로 센다.
  - 표준 정규화(Whisper·CHiME 계열)는 채움말을 지운다. AMI 는 채움말이 많아 이 차이가 몇 pt 날 수 있다.

### 표준 정규화로 다시 잰 E6 final(2026-10-27, `results/stage3-eval-ami-q17-e6-final-wn`)
| AMI 창 cpWER(δ10) | 우리 정규화 | Whisper 정규화(단어마다) | Whisper + meeteval(화자 통째) | agn(우리 → Whisper) |
|---|---:|---:|---:|---:|
| IHM-mix eval | 37.68 | 37.01 | 37.09 | 22.54 → 21.49 |
| IHM-mix dev | 43.15 | 42.31 | 42.31 | 22.76 → 21.61 |
| SDM eval | 45.35 | 44.69 | 44.73 | 28.06 → 27.16 |
| SDM dev | 48.87 | 48.27 | 48.36 | 26.64 → 25.61 |

- **정규화 차이는 작다.** cpWER 는 0.5–0.8, agn 은 약 1 내린다. 참조 단어 수도 0.6 % 만 줄었다(IHM-mix eval 88,206 → 87,671).
  - 외부 수치와의 차이는 정규화로 설명되지 않는다.
- **우리 cp 계산은 meeteval cpWER 와 같다.** 우리 정규화로 meeteval 을 돌리면 행의 cp 와 소수 둘째 자리까지 같다.
- **단어마다 대 문장 단위 정규화는 0.1 안에서 같다.**
- **참조에 없는 가설 화자(헛화자)가 IHM-mix eval 창 580 개에서 408 명이다**(창마다 0.7 명꼴).
  - meeteval 의 falarm_speaker 합이다. 그 화자의 말은 모두 삽입으로 센다.

## 판단 (2026-10-30 갱신 — E11 기준, 회의 통째·표준 정규화 + meeteval, eval)
1. **같은 화자 분리기(N3 Diarization)를 쓰는 Parakeet(화자별 디코드, 15.86·20.50)에 E11 은 IHM-mix 14.6·SDM 23.0 뒤진다.** 출력 재귀속(학습 없는 후처리)으로 7.2·9.1 까지 줄었다.
2. **재귀속 뒤 남은 차이는 대부분 받아쓰기다.** 귀속 몫은 2.35·3.05 만 남는다. 우리 Whisper agn(20.64·26.44)만으로도 Parakeet + N3 Diarization 의 cpWER 보다 높다.
3. **Sortformer v2 조합(17.59·34.59)보다는 재귀속 뒤 SDM 이 5.0 낮다.** IHM-mix 는 5.5 높다.
4. **VibeVoice-ASR-Streaming 은 우리 틀로 재면 우리보다 높다.** 7B 는 480 s 조각에서 48.17·43.16 이다. 논문 수치(27.48·39.01)는 재현되지 않는다(입력 종류·혼합 방식·Flash Attention 모두 아니다). 같은 LLM 크기 1.5B 보고값(45.15·56.83)보다도 E11 이 회의 통째에서 14.7·13.4 낮다.
5. 화자별 디코드(Parakeet 꼴)는 우리 목표(한 스트림 E2E) 밖이라 비교 기준으로만 쓴다.

### 지난 판단 (2026-10-27, E6 창 기준 — 기록)
1. **같은 지연(약 1 s) 스트리밍 기준(Parakeet v1 SDM 37.4)보다 8 pt 뒤다.** 그것도 더 쉬운 창 cpWER 로 잰 값이다. NVIDIA 의 새 결과(SSA-v2 SDM 20.3)와는 25 pt 차이다.
2. **VibeVoice-ASR-Streaming(7B, 지연 2.0 s)과 견주면 받아쓰기는 비슷하고 귀속이 두 배다.**
   - 받아쓰기: SDM agn 28.1 대 WER 29.8. 우리는 지연이 절반 아래다.
   - 귀속: 우리 cp − agn 은 15–17, VibeVoice 의 cpWER − WER 은 7.7–9.2 다.
3. **귀속이 가장 큰 차이다.** NVIDIA 가 SSA-v1 → v2 에서 SDM 39.9 → 20.3 으로 줄인 것도 주로 화자 분리를 바꾼 덕이다(Sortformer v2.1 → Nemotron 3 Diarization, 그 AMI SDM DER 12.8 @1.04 s).
   - 우리 인과 헤드의 창 DER 은 SDM eval 27 이다(E1c).
   - E8(헤드에 받아쓰기 지연만큼 앞 보이기)과 화자 분리 데이터·모델 보강이 같은 방향이다.
4. **받아쓰기도 실제로 뒤진다.** 표준 정규화로 다시 재도 SDM agn 은 27.2 다.
   - NVIDIA 는 정답 화자 구간에서 SDM cpWER 17.4–20.3 이다. 그쪽 받아쓰기 오류는 그 아래다.
   - 그쪽은 세션 단위·화자별 디코드라 조건이 다르지만, 8 pt 안팎의 받아쓰기 차이가 남는다.

## 같은 조건으로 맞추려면(2026-10-27 제안 — 회의 통째·표준 정규화 + meeteval 은 2026-10-28·30 에 모두 했다)
- AMI eval 16 회의를 **회의 통째**로 잰다(96172 E7a·96461 E6 기준선).
- **meeteval cpWER + 표준 정규화**(채움말 제거)로 다시 채점한다. 예측은 `predictions-rank*.jsonl` 에 있어 CPU 로 다시 매길 수 있다.
- 지연은 우리 "평균 80δ + 40r ms"(δ10 은 0.92 s)와 Parakeet "청크 1.12 s", VibeVoice-Streaming "기대 평균 2.0 s"로 정의가 서로 다르다는 것을 표에 적는다.

## 표준 정규화 재채점 도구 (2026-10-27, 사용자 "표준 정규화로 다시 채점하는 것도 만들어줘")
- `bash raw/sources/experiments/2026-10-27-std-rescore/rescore_std.sh ami|amifull <결과 폴더 이름>`(mxc 호스트 CPU, 디코드 없음, 원 폴더는 읽기만)
  1. `cst_stage3_eval.py run --from-predictions <원 폴더> --en-norm whisper` → `<원 폴더>-wn`.
     - 창·회의 점수(cp·agn·귀속·EOT)를 단어마다 Whisper EnglishTextNormalizer 로 다시 매긴다.
     - 이어서 `ami-report.txt`(공식 dev·eval 별)를 만든다.
  2. `experiments/meeteval_cpwer.py --norm whisper`·`--norm ours`.
     - 화자 발화를 통째로(문장 단위) 정규화해 meeteval cpWER 로 잰다. 외부 수치와 같은 도구다.
     - 결과는 `<원 폴더>-wn/meeteval-{whisper,ours}.{txt,json}` 에 남는다.
- **정규화:** `whisper_normalizer` 0.1.15(서버 env 에 이미 있음)의 EnglishTextNormalizer 다.
  - 채움말 hmm·mm·mhm·mmm·uh·um 을 지운다(mm-hmm 도 지워진다).
  - colour → color, twenty → 20, Mr → mister, won't → will not 로 바꾼다.
- **채점 코드:** `vapasr.data.cst_eval.set_en_norm`(기본 ours — 예전 값 그대로), `cst_stage3_eval --en-norm`(ours 면 config 키 없음).
- **외부 기준 자동 첨부(2026-10-28):** `meeteval_cpwer.py` 보고 끝에 외부 AMI 기준 줄이 붙는다(사용자 "앞으로 cpWER 보고할 때 다른 모델들의 수치도 함께 포함해서 비교하기 쉽도록 해줘").
  - 수치는 `raw/sources/experiments/2026-10-27-std-rescore/external_ami.json` 하나에 둔다. 이 문서의 비교표와 같은 값이다.
  - 새 외부 수치를 찾으면 두 곳을 함께 고친다.
- **단어마다 대 문장 단위:** 단어마다 하면 시각을 지켜 agn 을 함께 잴 수 있다. 문장 단위(meeteval)는 여러 단어 숫자까지 표준대로다.
  - 두 값의 차이는 숫자 표기 정도여야 한다. 실제 차이는 재채점 결과로 확인한다.

## 출처
- Multitalker Parakeet 카드: https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1/raw/main/README.md (commit 8749fc71fd)
- NVIDIA SSA-v2: https://arxiv.org/html/2609.10265v2 표 1
- Nemotron 3.5 Transcribe: https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf 표 13·14(2026-10-29 확인) — 우리 기여 검토는 [[output-nemotron35-transcribe-comparison-20261029]]
- VibeVoice-ASR: https://arxiv.org/html/2601.18184v2 표 2
- VibeVoice-ASR-Streaming: https://arxiv.org/html/2609.02812v2 표 2·5·7
- arXiv 2609.15218 표 2, VibeVoice GitHub issue #237, transcribe.cpp 재현(https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/multitalker-parakeet-streaming-0.6b-v1.md)
- 우리 수치: [[output-diarization-integration-20261025]] 7 절 결과 1(job 95936, `VAPKT-data/logs/e6-compare.log`)
