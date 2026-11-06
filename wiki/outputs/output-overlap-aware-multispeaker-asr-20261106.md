---
type: output
status: active
created: 2026-11-06
updated: 2026-11-06
summary: 겹친 발화를 복원하는 화자 조건부 인코딩, 목소리 참조, 분리 전처리, 직렬화 학습을 비교하고 E15b의 실험 우선순위를 정리
sources:
  - '[[output-agn-reduction-20261104]]'
  - '[[output-e13-gate-trans-20261102]]'
  - '[[source-nemotron-3-5-transcribe-tech-report]]'
  - https://arxiv.org/abs/2609.10265
  - https://arxiv.org/abs/2601.19194
  - https://arxiv.org/abs/2510.03723v2
  - https://www.isca-archive.org/interspeech_2025/subramanian25b_interspeech.pdf
  - https://arxiv.org/abs/2604.22467v2
  - https://arxiv.org/abs/2603.17769
  - https://arxiv.org/abs/2608.22796
  - https://arxiv.org/abs/2610.25824
---

# 겹친 발화의 단어를 복원하는 다화자 ASR 기법

## 질문

단일 채널 스트리밍 ASR에서 두 사람의 말이 겹칠 때 두 전사를 모두 살리는 데 어떤 기법을 쓸 수 있는가. 공개 자료는 2026-11-06 확인했다.

## 현재 병목

AMI eval에서 N3D 활동을 공통으로 준 E13(출력 재귀속 포함)과 Multitalker Parakeet을 비교했다. 두 화자가 겹친 구간의 단어 놓침률(삭제와 대치의 합)은 IHM-mix에서 **42.7% 대 21.3%**, SDM에서 **48.9% 대 29.8%**다. 두 화자 및 세 화자 이상 겹침을 합치면 전체 cpWER 격차 중 5.02 / 4.43%p를 차지한다. E13의 게이트 진단에서는 방출 시점에 해당 슬롯이 닫힌 참조 단어가 전체의 3.6% / 3.2%였다. 게이트 회복과 함께, 슬롯이 열렸을 때 각 화자의 말을 구별할 음향 특징이 필요하다. [[output-agn-reduction-20261104|AMI 오류 예산 10절]] · [[output-e13-gate-trans-20261102]]

## 공개 기법과 실험 근거

| 기법 | 겹친 발화에 쓰는 방법 | 보고된 근거와 조건 |
|---|---|---|
| **화자 조건부 인코딩** — Multitalker Parakeet(SSA-v2), DiCoW | 혼합 음성의 인코더 내부에 화자 활동 정보를 넣고, 대상 화자별로 음향 특징과 전사를 구한다. | 2026년 스트리밍 비교에서 SSA-v2 + N3D는 AMI SDM cpWER 20.33%를 보고했다. 화자별 인식은 화자 수가 늘면 연산과 메모리도 늘어난다. [스트리밍 구조 비교, 표 1](https://arxiv.org/html/2609.10265) |
| **화자별 특징과 공유 디코더** — SA-DiCoW | 화자별 조건부 특징을 구한 뒤 이어 붙여 디코더 하나에 준다. | LibriMix 3인 합성에서는 DiCoW의 cpWER 32.1%를 17.2%로 낮췄고, AMI SDM의 정답 활동 조건에서는 16.3%에서 18.1%로 높아졌다. 30초 청크를 쓰는 구조 비교다. [SA-DiCoW, 표 3·4](https://arxiv.org/html/2510.03723v2) |
| **화자 참조 음성** — SE-DiCoW | 두 사람의 활동 마스크가 비슷할 때, 같은 녹음에서 얻은 대상 화자의 음향 단서를 인코더 교차 어텐션에 넣는다. | DiCoW v3.3 대비 Libri3Mix Clean tcpWER는 정답 활동에서 16.0→9.7%, DiariZen 활동에서 31.6→29.3%였다. DiariZen 활동을 쓴 AMI SDM은 18.6→18.5%, IHM-mix는 15.2→15.3%였다. 논문은 참조 구간을 찾을 때 녹음 전체를 훑었다. [SE-DiCoW, §2.2·표 1](https://arxiv.org/html/2601.19194v1) |
| **연속 음성 분리(CSS) + 다화자 ASR** | 단일 채널에서 겹친 파형을 여러 스트림으로 분리해 인식기에 준다. | 단일 채널 LibriCSS의 40% 겹침 조건에서 CSS를 붙인 CT-tSOT의 화자 무관 WER가 16.47→14.97%로 내려갔다. 무겹침에서는 오류가 늘었고, 이 분리기는 0.8초 처리 지연을 보고했다. [Subramanian 외, 표 1·§3.2.1](https://www.isca-archive.org/interspeech_2025/subramanian25b_interspeech.pdf) |
| **출력 순서 학습** — t-SOT, Shuffle, DM-ASR | t-SOT는 여러 화자의 토큰을 시간순으로 섞고, Shuffle은 각 화자 내부의 순서를 보존하는 복수의 섞기 순서를 학습 손실에서 허용한다. DM-ASR은 화자와 시각 조건을 명시해 하나의 LLM으로 차례로 전사한다. | Shuffle의 개선 근거는 합성 LibriSpeech 계열 실험이며, DM-ASR v2의 온라인 단위는 문장이다. [t-SOT](https://arxiv.org/abs/2202.00842) · [Shuffle, 표 2](https://arxiv.org/html/2603.17769v1) · [DM-ASR v2](https://arxiv.org/abs/2604.22467v2) |

학습 데이터에서도 겹침을 직접 제어할 수 있다. DiaScriber는 턴 경계를 이동해 두 화자가 겹치는 길이와 비율을 바꾸고, 추가 발화를 섞어 세 화자 이상을 구성했다. VAPKT에서는 [[output-agn-reduction-20261104]]에 기록한 `sim_en-ovl`을 출발점으로, 겹침 지속 시간과 화자 간 음량 차이를 별도 축으로 나눠 작은 목소리·맞장구·끼어들기를 평가할 수 있다. 전체 지표가 비슷해도 겹침 조건별 성능이 크게 달라질 수 있다. [DiaScriber, §2.2](https://arxiv.org/html/2608.22796v1) · [MS-Bench, §4.2](https://arxiv.org/html/2610.25824v1)

## E15b에서 확인할 순서

1. **기본 E15b 효과를 분리한다.** N3D와 디코드 지연을 고정하고, E13·E15b·Parakeet을 같은 AMI eval 정규화로 비교한다. E15b에서 겹침 증강을 함께 바꾸므로, 구조의 효과를 분리하려면 E13에도 같은 겹침 데이터를 주는 대조가 필요하다. 2026-11-06 위키 기록에는 E15b 스모크와 제출까지 있다. [[output-agn-reduction-20261104]]
2. **두 활동 신호가 비슷할 때 목소리 참조를 시험한다.** E15b의 `SlotInject`는 대상과 나머지 슬롯 활동으로 공통 음향 특징을 조건화한다. 두 슬롯의 활동이 같은 프레임에서는 이 연산이 같은 조건을 받는다. 주변 문맥이나 디코더 슬롯은 여전히 단서가 될 수 있다. 긴 겹침에서 놓침이 남으면, 이미 지나간 구간 중 슬롯별로 상대 화자의 간섭이 적은 구간을 골라 음색 임베딩을 캐시하고 활동 입력과 함께 사용해 볼 수 있다. 참조를 확보하기 전 구간은 따로 집계한다. 이는 SE-DiCoW에서 착안한 스트리밍 가설이다. [SE-DiCoW, §2.2](https://arxiv.org/html/2601.19194v1) · [[output-agn-reduction-20261104]]
3. **겹침 복원과 화자 귀속을 따로 잰다.** 참조상 동시 활성 화자 수(1/2/3+)와 약한 쪽 화자의 음량을 기준으로 단어 삭제·대치·삽입을 분리하고, cpWER와 방출 지연을 함께 본다. 예측 활동과 정답 활동의 차이는 diarization의 영향을, 분리된 AMI 헤드셋 채널을 따로 인식한 값은 혼합이 없는 참고 성능을 보여준다. AMI 영어 실험과 KO↔EN 대화는 분리해 보고한다. [[output-agn-reduction-20261104]] · [[conversational-simultaneous-s2tt]]

## 불확실성

- 공개 논문의 cpWER, tcpWER, 화자 무관 WER는 정의와 정규화가 다르고, 지연과 참조 화자 활동 사용 여부도 다르다. 표의 수치는 각 논문 내부의 비교에 쓴다.
- 오프라인 비교에는 [[source-nemotron-3-5-transcribe-tech-report|Nemotron 3.5 Transcribe]]와 SE-DiCoW를 쓸 수 있다. 과거 음성만 쓰는 참조 방식의 지연과 효과, 교차 언어 겹침에서의 효과는 VAPKT에서 실측이 필요하다.
