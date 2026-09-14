---
type: output
status: active
created: 2026-09-13
updated: 2026-09-14
summary: AMI 실제 4화자 38.4초를 480개 블록으로 직렬화; 전사 복원 검증과 marketing…expert EOT 검수 후보 확인
sources:
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[output-phase2-training-db]]'
  - '[[task-secure-meeting-corpora]]'
---

# 실제 데이터로 만든 Phase 2 시퀀스

> **실험 기록, 규약 아님:** 아래는 2026-09-13의 임시 registry·P-mode 후보 생성 결과다. 현재 Q1 C-mode·registry·serializer 계약은 [[output-phase2-streaming-asr-diarization-plan]]만 따른다. 과거 결과를 소급 수정하지 않으며 학습 승인 데이터가 아니다.

**AMI의 실제 38.4초 음성과 전사로 80ms 블록 480개를 만들었다.** 무음 후보·단독 발화·두 화자 겹침·화자 교대·지연 전사·ONSET/EOT를 확인했다. 네 화자가 등장하지만 최대 동시 발화는 이 표본에서 2명이다.

이것은 **학습 타깃 구성 실험**이다. 모델이 예측한 결과나 학습 성능이 아니다. 오디오는 실제로 잘라 mono로 만들었고 lexical BPE ID도 생성했다. 다만 `[AUDIO_k]`는 encoder 벡터가 들어갈 자리이며, 이번에는 Nemotron 특징 추출·학습·추론을 실행하지 않았다.

## 1. 사용한 데이터와 규약

| 항목 | 이번 실험 |
|---|---|
| 원본 | AMI `ES2002a`, 원 녹음 **54.40–92.80초** |
| 음성 | `Headset-0..3.wav`, 16kHz·16-bit·각 mono; 동기화된 네 채널을 고정 1/4 평균 |
| 전사·시각 | `ami_public_manual_1.6.2.zip`의 words XML; 이번에는 ForcedAligner 재실행 없음 |
| 활동·ONSET 기준 | AMI transcriber segment, 같은 화자 gap <250ms 병합. **음향 VAD 실측이 아닌 대용 라벨** |
| lexical | E2 thinker MLX export의 실제 tokenizer; 소문자·구두점 제거·내부 아포스트로피 유지·단어 앞 공백 |
| 지연 | `δ_text=4`, `δ_onset=2`; EOT는 정본의 P 모드(미래 행동 예측) 시험 휴리스틱 |
| 화자 순서 | 독립 crop에서 처음 나타난 순서: **B→1, D→2, A→3, C→4** |

이 crop은 주석상 모든 화자가 쉬는 시점에서 시작한다. 슬롯 순서는 원 회의 전체 ID가 아니라 **crop 내 순서**이며, 온라인 화자 식별 성능을 측정한 것이 아니다. 사후 전사·화자 메타데이터를 정답 생성에만 사용했다.

정규화는 이 표본의 영어 단어에 적용한 제한된 함수다. 읽히지 않는 프로젝트 TN 파일을 우회했으므로 **동결 TN-v1과 완전 동일하다고 검증한 것은 아니다**. 특수 토큰 6개는 tokenizer 메모리에만 추가했으며 원본 tokenizer/체크포인트는 수정하지 않았다.

## 2. 블록으로 보면

아래 시각은 원 녹음 기준이다. 각 행의 입력은 실제 해당 80ms mono 구간이다. 출력은 그 블록에서 모델이 **맞히도록 학습할 정답**이다. `NEXT`는 표에서만 `<NEXT_AUDIO>`를 줄인 표기다.

활동 열은 슬롯 순서 `[1,2,3,4]`이며, 1은 해당 청크와 발화 어노테이션이 겹친다는 뜻이다. `0000`도 호흡·잡음까지 없는 디지털 무음을 뜻하지 않는다.

| 블록 k | 입력 구간(초) | 활동 | 정답 출력 | 의미 |
|---:|---|---|---|---|
| 0–11 | 54.40–55.36 | 0000 | 각 블록 `NEXT` | 발화 주석 없음, audio 블록은 계속 입력 |
| 12 | 55.36–55.44 | 1000 | `NEXT` | 말은 시작했지만 onset 목표 시각 전 |
| 14 | 55.52–55.60 | 1000 | `<SPK_1><ONSET> NEXT` | 첫 화자 시작 |
| 254 | 74.72–74.80 | 1100 | `manager NEXT` | 두 화자가 겹치지만 이번 lexical은 1 소유 |
| 255 | 74.80–74.88 | 1100 | `<SPK_2><ONSET> NEXT` | 2의 짧은 발화 시작 |
| 264 | 75.52–75.60 | 1000 | `<SPK_2> great NEXT` | 2는 이미 조용하지만 지연 전사는 지금 도착 |
| 289 | 77.52–77.60 | 0010 | `<SPK_3><ONSET> NEXT` | 3의 시작 |
| 290 | 77.60–77.68 | 0010 | `<SPK_1> again NEXT` | 1의 마지막 전사는 별도 보존 |
| 292 | 77.76–77.84 | 0010 | `<SPK_1><EOT> NEXT` | 3이 말하는 중에 1의 종료 예측 |
| 301 | 78.48–78.56 | 0010 | `i'm NEXT` | 두 BPE 토큰을 같은 화자 묶음으로 방출 |
| 334 | 81.12–81.20 | 1000 | `<SPK_3> designer NEXT` | 현재 발화자는 1, 전사 소유자는 3 |
| 335 | 81.20–81.28 | 1000 | `<SPK_3><EOT> NEXT` | 이벤트만 있는 화자 블록 |
| 337–338 | 81.36–81.52 | 0000 | 각 블록 `NEXT` | 짧은 무발화 구간 |
| 340 | 81.60–81.68 | 0100 | `<SPK_1><EOT> NEXT` | 2의 음성과 1의 종료 출력은 독립 |
| 341 | 81.68–81.76 | 0100 | `<SPK_2><ONSET> NEXT` | 이전 맞장구 화자 2의 재등장 |
| 379 | 84.72–84.80 | 0000 | `marketing NEXT` | 발화 후 쉬는 중 지연 lexical |
| 381 | 84.88–84.96 | 0000 | `<SPK_2><EOT> NEXT` | 휴리스틱 종료 예측 — 아래 한계 사례 |
| 382–387 | 84.96–85.44 | 0000 | 각 블록 `NEXT` | 추가 빈 화자 블록 없음 |
| 390 | 85.60–85.68 | 0001 | `<SPK_4><ONSET> NEXT` | 네 번째 화자 시작 |
| 397 | 86.16–86.24 | 0101 | `<SPK_2><ONSET> NEXT` | 4가 말하는 동안 2가 다시 발화 |
| 405 | 86.80–86.88 | 0101 | `expert NEXT` | 2의 앞선 “marketing”을 잇는 단어 |
| 413 | 87.44–87.52 | 0001 | `<SPK_4> um NEXT` | 4의 지연 전사 |
| 436 | 89.28–89.36 | 1000 | `<SPK_4><EOT> NEXT` | 1이 다시 말할 때 4 종료 예측 |

일부 행만 뽑았으므로 생략된 블록에도 오디오와 NEXT는 존재한다. 태그 생략 여부는 **전체 시퀀스의 마지막 selector**를 기준으로 한다. 전체 480행은 [blocks.tsv](../../raw/sources/experiments/2026-09-13-phase2-sequence-probe/blocks.tsv)에 있다.

### 실제 토큰 ID와 학습 라벨

블록 335의 타깃 ID는 `[151719, 151722, 151705]`다.

```text
입력/정답 시퀀스: [AUDIO_335] <SPK_3> <EOT> <NEXT_AUDIO> [AUDIO_336]
해당 위치의 예측:     SPK_3      EOT      NEXT      loss 없음
타깃 ID:             151719    151722    151705       -100
손실 가중:              1         2        0.3          0
```

`i'm`은 이 tokenizer에서 `[600,2776]`이다. lexical ID는 기존 모델의 실제 ID이고, `<SPK_1..4>`는 `151717..151720`, ONSET/EOT는 `151721/151722`라는 **시험용 추가 ID**다. 모델 embedding/lm_head 확장 전에는 이 파일을 그대로 학습기에 넣을 수 없다. 저장된 training-sequence는 청크 본문이며 chat prefix·DELAY prefix·실제 soft embedding은 학습기 연결 때 추가해야 한다.

## 3. 만들어 보니 확인된 것

| 검사 | 결과 |
|---|---|
| 블록 구성 | **480 audio 자리 + 480 NEXT**, NEXT-only 블록 376개 |
| lexical | 이 구간에서 방출한 단어 묶음 90개, BPE **104개** |
| 구조 출력 | 화자 태그 23개, ONSET **9개**, EOT **5개** |
| 총 시퀀스 위치 | **1,101개** — audio 자리 포함, prefix 제외 |
| 활동 청크 | 무발화 주석 27 / 단독 415 / 두 화자 겹침 38 |
| 출력 밀도 | NEXT 포함 최대 **3토큰/청크**, cap 8 초과 없음 |
| 복원·학습 라벨 | 별도 parser로 화자별 lexical ID 완전 복원, 모든 next-token label·이벤트 태그 귀속 검사 통과 |
| nominal lexical 지연 | 단어 끝 기준 **330–400ms**; 모델 실행 지연·실시간성 측정 아님 |

이 표본에는 3–4명 동시 발화나 매우 빠른 겹침이 없어 cap 8이 항상 충분하다는 근거는 아니다. 활동 청크 수도 경계가 겹친 80ms 청크를 센 값이지 정확한 겹침 초 수가 아니다.

crop 끝에는 `okay`, `um` 두 단어가 목표 지연 때문에 다음 블록으로 남았다(k=480,483). **녹음 EOF가 아닌 crop 종료이므로 가짜 EOT나 EMPTY_AUDIO flush로 없애지 않고 pending으로 저장**했다.

## 4. 가장 중요한 발견: 자동 EOT는 아직 위험하다

화자 2의 실제 전사는 다음과 같이 이어진다.

```text
82.08–84.46  and i'm andrew and i'm uh our marketing
86.07–86.50  expert
```

중간에 화자 4가 시작하므로 현재 시험 휴리스틱은 84.576초 경계에 SHIFT 후보를 만들고, **84.96초 블록 끝에 EOT 타깃**을 둔다. 이후 “expert” 구간은 다른 화자 발화 안의 1초 미만 구간이라 BC와 비슷한 타이밍으로 분류해 EOT를 생략한다.

그러나 전사 문맥상 “marketing expert”는 하나로 이어지는 표현이다. **짧은 겹침 = 의미적 맞장구가 아니며, 다른 화자가 먼저 시작함 = 이전 화자가 내용을 마쳤음도 아니다.** 이 규칙을 의미 EOT gold로 사용하면 잘못된 감독이 된다. P 모드 행동 예측 표본으로도 정의·품질 검수가 필요하다.

이 사례는 **확정 오답이 아닌 검수 후보**다. 문법적으로 이어져도 발언권을 넘겼다가 다시 말했을 수 있으므로 청취 전에는 EOT의 정오를 확정하지 않는다. 후속 [[output-phase2-eot-review-framework]]는 후보 추출·독립 청취·누락 감사·라벨 릴리스 절차를 설계한다.

이번 EOT는 발화 구간·재개·다른 화자 시작을 본 **약한 라벨**이다. 예컨대 위 EOT는 라벨 판정에 87.576초까지를 사용하지만 출력 타깃은 84.96초다. 미래는 정답 생성에만 썼으며 ‘84.96초에 종료가 이미 확인됐다’는 뜻이 아니다.

**판정:** 실제 데이터에서 시퀀스와 화자별 버퍼를 구성하는 것은 가능하다. 다음은 더 큰 학습보다 먼저 (1) 실제 VAD/정렬과 어노테이션 비교, (2) 이런 재개·짧은 겹침 EOT 후보 검수, (3) 동결 TN·token registry 연결이다. 모델 학습 가능성 자체는 이번 직렬화 실험만으로 판정하지 않는다.

## 5. 산출물과 재현

- [38.4초 mono 음성](../../raw/sources/experiments/2026-09-13-phase2-sequence-probe/mono-crop.wav): 실제 네 채널의 평균 혼합. 원본 파일은 변경하지 않았다.
- [전체 블록 표](../../raw/sources/experiments/2026-09-13-phase2-sequence-probe/blocks.tsv) · [블록 ID/사건 JSON](../../raw/sources/experiments/2026-09-13-phase2-sequence-probe/blocks.json)
- [학습 시퀀스 본문·next-token labels](../../raw/sources/experiments/2026-09-13-phase2-sequence-probe/training-sequence.json)
- [원 단어·시각](../../raw/sources/experiments/2026-09-13-phase2-sequence-probe/words.json) · [휴리스틱 사건 근거](../../raw/sources/experiments/2026-09-13-phase2-sequence-probe/events.json) · [검증·소스 해시](../../raw/sources/experiments/2026-09-13-phase2-sequence-probe/summary.json)
- 생성 코드: `experiments/p2_sequence_probe.py`. 같은 출력 폴더를 덮어쓰지 않으므로 재실행은 새 출력 경로를 지정한다.

```sh
/Users/taesookim/anaconda3/envs/vapasr-local/bin/python experiments/p2_sequence_probe.py \
  --audio /Users/taesookim/Downloads/vapkt-corpora/ami \
  --annotations /Users/taesookim/Downloads/vapkt-corpora/ami/annotations/ami_public_manual_1.6.2.zip \
  --tokenizer /Users/taesookim/Desktop/VAPKT-models/hf-E2-final-thinker-mlx/tokenizer.json \
  --output /tmp/vapkt-p2-sequence-reproduction
```

로컬 인덱스·일부 코드가 `dataless`라 읽기가 정체되어, 읽을 수 있는 정본·AMI 원본·MLX export tokenizer로 수행했다. AMI 원본 주석과 음성을 직접 사용했으며 새 문장이나 단어 시각을 지어내지 않았다. 이 표본은 Q0 진단용으로 노출됐으므로 향후 untouched 평가에서는 분리한다.
