---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-11-10
created: 2026-10-27
updated: 2026-10-27
summary: AMI 는 공식 train 회의만 학습한다. 공식 dev 는 모델·임계·설정 선택에, eval 은 보고에만 쓴다(eval 화자는 train·dev 와 겹치지 않음 확인)
sources:
  - '[[output-diarization-integration-20261025]]'
related:
  - '[[decision-chime6-train-sessions-20261026]]'
  - '[[decision-eval-train-decontamination]]'
---

# AMI 는 train 회의만 학습한다 (2026-10-27)

## 맥락
- 사용자가 물었다: "AMI SDM, IHM dev set 은 학습에 사용해도 다른 논문들과 공평하게 비교 가능하지 않아?"
- 지금 학습 분할(phase2 `splits/v4.json`)은 AMI 공식 full-corpus-ASR 분할의 dev 18·eval 16 회의와 ES2014a–d 를 `ami` 에서 뺀다.
  - 원거리 버전 `ami-far` 는 학습 회의 129 개로만 만들었다(dev·eval 회의 없음, 2026-10-27 서버 확인).

## 확인한 사실 (2026-10-27, AMI 원 주석 `ami_public_manual_1.6.2.zip` 의 `corpusResources/meetings.xml` 전역 화자 ID)
| | 회의 | 화자 | eval 과 겹치는 화자 |
|---|---:|---:|---:|
| train(학습에 쓰는 133 회의) | 133 | 151 | 0 |
| dev | 18 | 21 | 0 |
| eval | 16 | 16 | – |

- train 과 dev 는 2 명(FIE038, MIO036)이 겹친다.
- 그래서 dev 를 학습에 넣어도 eval 회의나 eval 화자는 새지 않는다.
- CHiME-7 DASR 규칙은 CHiME-6·DiPCo·Mixer 6 에 대해 train·dev 분할을 재배치해도 된다고 허용했다(평가 분할은 금지). 하지만 AMI 는 그 대회 데이터가 아니다.

## 선택지
1. dev 를 학습에 넣는다 — 같은 도메인 18 회의(train 의 약 14 %)가 늘고, eval 만 보고한다.
2. train 만 학습한다 — dev 는 선택·조정용, eval 은 보고용(AMI 논문들의 관행).

## 결정
사용자 결정(2026-10-27): **2. 지금처럼 train 만 학습한다.**

이유:
- AMI 로 ASR·화자 귀속 ASR·화자 분리를 보고하는 논문들은 대개 train 으로 학습하고 dev 로 조정해 dev·eval 을 같이 낸다. dev 를 넣으면 같은 도메인 데이터 이점이 생겨 표에 "train+dev"라고 밝혀야 한다.
- dev 수치를 더는 보고할 수 없다.
- 지금 dev 를 선택에 쓰고 있다(헤드 임계 0.45, 체크포인트·평균, AOSC 설정 비교). dev 를 학습에 넣으면 그 선택을 할 다른 held-out 이 필요하고, ES2014 4 회의는 너무 작다.

## 결과
- `ami`·`ami-far`·화자 분리 학습 모두 지금 분할을 그대로 쓴다.
- 최종 모델을 train+dev 로 다시 학습한 줄을 덧붙이려면, train 만 쓴 실험에서 정한 설정을 고정하고 따로 밝힌다(지금 계획에는 없다).

## 함께 고친 것
- `ami-far` 의 conv_id 앞에 정렬 점검 기록(dict 문자열)이 붙어 있었다. 원인은 `experiments/p2_build_far.py` 에서 버린 변형을 출력하는 반복문이 출력 키 변수를 덮어쓴 것이다.
  - 학습 내용에는 영향이 없다.
    - 행의 발화·토큰·화자·`meta.mono_src` 는 맞았다.
    - mono 캐시는 같은 이름으로 만들고 읽었다. 캐시를 못 읽어도 `mix_dialogue` 가 `mono_src` 로 원거리 소리를 만든다.
    - 번역 타깃은 `meta.far_of` 로 찾는다.
    - 창 시작 난수만 conv_id 문자열에 따라 달랐다.
  - `experiments/p2_fix_far_ids.py` 로 conv_id 를 `ami-far:<회의>@<장치>` 로 바꿨다.
    - 옛 jsonl 은 `.bad-conv-id` 로 남겼다.
    - mono 캐시 1032 개는 새 이름 심볼릭 링크로 이었다. 저장소가 하드 링크를 막는다(Errno 524).
    - build.json 은 `ami-far.build.json` 으로 복사했다.
  - 아무것도 지우지 않았다. 옛 이름의 build.json 과 캐시 파일은 그대로 있다.

## 재검토
- 논문 표를 만들 때(검토일 2026-11-10) 비교 대상 논문들의 AMI 학습 데이터(train 만인지, 외부 데이터를 썼는지)를 표에 같이 적는다.
