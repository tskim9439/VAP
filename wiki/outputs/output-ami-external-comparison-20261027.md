---
type: output
status: active
created: 2026-10-27
updated: 2026-10-27
summary: AMI cpWER 외부 비교 — Parakeet SDM 37.4·NVIDIA SSA-v2 20.3·VibeVoice 대 우리 E6 창 SDM 45.4(표준 정규화 44.7); 귀속이 가장 크고 받아쓰기도 뒤진다
sources:
  - '[[output-diarization-integration-20261025]]'
  - '[[output-vibevoice-stream-comparison-20261020]]'
raw_authors:
  - tskim
---

# AMI 외부 비교: 우리 대 Multitalker Parakeet · VibeVoice-ASR (2026-10-27)

사용자 요청은 "https://openreview.net/attachment?id=r6QZahgnJY&name=pdf 이것과 VibeVoice-ASR, "Multitalker Parakeet SDM" 과는 비교해서는 어떠한지 정리해줘"였다.
- OpenReview 문서는 브라우저 확인 화면(403) 때문에 받지 못했다. 사용자가 PDF 나 제목을 주면 덧붙인다.
- 외부 수치는 2026-10-27 원문(HF 카드 raw README, arXiv HTML 표)에서 직접 뽑았다.

## 비교표 (AMI cpWER %, 낮을수록 좋다)
| 시스템 | 크기 | 지연 | 화자 분리 | IHM(-mix) | SDM | 출처·조건 |
|---|---|---|---|---:|---:|---|
| **우리 E6 final** | 인코더 0.6B + LLM 1.7B | δ10: 0.8–1.04 s | 인과 헤드(예측) | **37.68** | **45.35** | AMI eval, **창(20–90 s) cpWER** — 세션보다 쉽다, 자체 정규화(채움말 남김) |
| 우리 E6 avg3 | 같음 | 같음 | 같음 | 35.53 | 44.15 | 같음 |
| 우리 E5(정답 활동 입력, 상한) | 같음 | 같음 | 정답 | 23.39 | 29.24 | 같음 |
| Multitalker Parakeet v1(카드) | 0.6B × 화자 수 + 117M | 1.12 s | Streaming Sortformer v2 | 21.26 | 37.44 | HF 카드 — split·정규화·세션 여부 미기재 |
| SSA-v1(arXiv 2609.10265 표 1) | 같음 | 스트리밍(표에 지연 미기재) | Sortformer v2.1 | 22.10 | 39.91 | 정답 화자 구간이면 15.22 · 20.27 |
| **SSA-v2 + Nemotron 3 Diarization**(같은 표) | 0.6B × 화자 수 + 100M | 스트리밍 | Nemotron 3 Diarization | **13.38** | **20.33** | 정답 화자 구간이면 12.41 · 17.44 |
| VibeVoice-ASR-Streaming 7B(arXiv 2609.02812 표 2) | 8.7B | 기대 평균 2.0 s(첫 출력 3.5 s) | 종단(모델이 직접) | 27.48 | 39.01 | 녹음 480 s 상한, WER 19.83 · 29.81 |
| VibeVoice-ASR 7B 오프라인(arXiv 2601.18184 표 2) | 8.7B | 오프라인(60 분 한 번에) | 종단 | 20.41 | 28.82 | WER 18.81 · 24.65, DER 11.92 · 13.43 |
| DiCoW-v3.3(2609.10265 표 1) | Whisper 계열 | 오프라인 | 정답 / 예측 | 11.33 / 14.48 | 14.78 / 18.24 | 참고 상한 |

### 같은 시스템인데 출처마다 다른 값
- **VibeVoice-ASR 오프라인 SDM:** 28.82(자체), 42.19(NVIDIA 평가 셋), 28.42(arXiv 2609.15218, 2·5 분 조각 평균), 36.09(제3자 재현, GitHub VibeVoice issue #237).
  - 채점 조건에 따라 14 pt 가까이 흔들린다.
- **Parakeet 카드 IHM 21.26:** 제3자 재현(transcribe.cpp — AMI IHM test 16 회의, meeteval, 회의 통째)의 NeMo 참조 21.39 와 맞는다.
  - 2609.10265 는 IHM 을 "Individual Headset Mix"(= 우리 IHM-mix)로 정의한다.
- **카드의 평가 세션 수 표**(AMI IHM 219 등)는 화자 분리 모델 카드의 DIHARD·CALLHOME 수를 옮겨 온 것으로 보여 근거로 쓰지 않는다.

## 받아쓰기 몫(화자 무시)
| | IHM(-mix) | SDM |
|---|---:|---:|
| 우리 E6 agn(창) | 22.54 | 28.06 |
| 우리 E5 agn(정답 활동) | 22.40 | 27.84 |
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
- **가설 화자가 참조보다 많은 창(헛화자)이 창의 약 70 % 다**(IHM-mix eval 408/580).

## 판단
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

## 같은 조건으로 맞추려면(제안)
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
- **단어마다 대 문장 단위:** 단어마다 하면 시각을 지켜 agn 을 함께 잴 수 있다. 문장 단위(meeteval)는 여러 단어 숫자까지 표준대로다.
  - 두 값의 차이는 숫자 표기 정도여야 한다. 실제 차이는 재채점 결과로 확인한다.

## 출처
- Multitalker Parakeet 카드: https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1/raw/main/README.md (commit 8749fc71fd)
- NVIDIA SSA-v2: https://arxiv.org/html/2609.10265v2 표 1
- VibeVoice-ASR: https://arxiv.org/html/2601.18184v2 표 2
- VibeVoice-ASR-Streaming: https://arxiv.org/html/2609.02812v2 표 2·5·7
- arXiv 2609.15218 표 2, VibeVoice GitHub issue #237, transcribe.cpp 재현(https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/multitalker-parakeet-streaming-0.6b-v1.md)
- 우리 수치: [[output-diarization-integration-20261025]] 7 절 결과 1(job 95936, `VAPKT-data/logs/e6-compare.log`)
