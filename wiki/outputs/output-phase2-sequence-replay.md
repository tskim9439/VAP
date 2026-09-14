---
type: output
status: active
created: 2026-09-14
updated: 2026-09-14
summary: 보존된 AMI 실제 38.4초로 480블록 재직렬화; 실제 토큰 복원 검증과 미검수 EOT 5개를 분리해 표시
sources:
  - '[[output-phase2-real-sequence-probe]]'
  - '[[output-phase2-eot-review-framework]]'
---

# 실제 데이터 시퀀스 재구성 및 블록 표시

> **실험 기록, 규약 아님:** 이전 P-mode probe를 재직렬화한 결과다. 현재 Q1 C-mode와 정식 token registry는 [[output-phase2-streaming-asr-diarization-plan]]을 따른다. 아래 JSON은 현재 Dataset 입력·승인된 학습 라벨이 아니다.

## 무엇을 실행했나

AMI `ES2002a` **54.40–92.80초, 실제 mono 음성 38.4초**와 보존된 단어 시각·이벤트 레코드를 읽어 시퀀스를 다시 직렬화했다. 새로운 문장이나 시각을 만들지 않았다. 기존 결과와 토큰 ID를 비교하고 별도 parser로 화자별 전사를 복원했다.

원본 AMI WAV·주석 ZIP은 이번 확인 시 이전 Downloads 경로에 없었다. 따라서 **9월 13일 보존본의 재직렬화**이며 새 녹음 표본 추출·재정렬·새 EOT 유도 결과는 아니다. 원본이 없어진 이유나 서버 전송 완료 여부는 이번 작업에서 확인하지 않았다. 기존 raw 파일은 수정하지 않았다.

- 입력: 16kHz mono PCM, 80ms마다 1,280 sample → 480블록.
- 화자 슬롯: 원 화자 B→1, D→2, A→3, C→4. 이번 crop의 정답 매핑이며 모델의 화자 식별 결과는 아니다.
- lexical δ=4, ONSET δ=2. `[AUDIO_k]`는 실제 PCM에 대응하는 soft-token **자리**다. encoder 벡터 추출·모델 추론은 하지 않았다.
- 활동 표시는 AMI 발화 구간 주석 대용값이며 음향 VAD 실측이 아니다. `0000`은 호흡·잡음까지 없는 무음을 뜻하지 않는다.
- EOT는 자동 후보 5개, 사람 검수 0개다. 결과 manifest는 `event_complete=false`, `approved_for_joint_training=false`로 저장했다.

## 실제 블록 예시

`NEXT`는 `<NEXT_AUDIO>`의 약칭이다. 모든 행 앞에는 해당 구간의 `[AUDIO_k]`가 있다. 표는 일부 발췌이며 생략된 블록도 audio와 NEXT를 갖는다. 태그가 없으면 전체 시퀀스의 직전 selector가 유지된다.

| k | 원 녹음 입력 구간 | 활동 슬롯 | 출력 후보 | 읽는 법 |
|---:|---|---|---|---|
| 0–11 | 54.40–55.36 | 0000 | 각 블록 `NEXT` | 발화 주석 없음; 오디오 입력은 계속 |
| 12–13 | 55.36–55.52 | 1000 | 각 블록 `NEXT` | 1이 시작했지만 ONSET 목표 지연 전 |
| 14 | 55.52–55.60 | 1000 | `<SPK_1><ONSET> NEXT` | 1의 시작 출력 |
| 254 | 74.72–74.80 | 1100 | `manager NEXT` | 겹침 음성을 받으며 1의 전사 출력 |
| 255 | 74.80–74.88 | 1100 | `<SPK_2><ONSET> NEXT` | 2의 시작; audio를 화자별로 나누지 않음 |
| 264 | 75.52–75.60 | 1000 | `<SPK_2> great NEXT` | 이미 조용해진 2의 지연 전사 |
| 379 | 84.72–84.80 | 0000 | `marketing NEXT` | 2의 마지막 단어가 지연되어 출력 |
| 380 | 84.80–84.88 | 0000 | `NEXT` | 새 출력 없이 다음 오디오로 |
| 381 | 84.88–84.96 | 0000 | `<SPK_2><EOT> NEXT` | **미검수 종료 후보**, 확정 아님 |
| 382–387 | 84.96–85.44 | 0000 | 각 블록 `NEXT` | 무발화 주석 구간도 같은 블록 구조 |
| 390 | 85.60–85.68 | 0001 | `<SPK_4><ONSET> NEXT` | 4의 시작 |
| 397 | 86.16–86.24 | 0101 | `<SPK_2><ONSET> NEXT` | 4가 말하는 중 2 재개 |
| 405 | 86.80–86.88 | 0101 | `expert NEXT` | 2의 전사; 앞의 EOT가 버퍼를 닫지 않음 |

핵심 구간을 직렬화하면 다음과 같다. `…`는 실제 파일에는 없는 설명용 생략 표시다.

```text
[AUDIO_379] marketing <NEXT_AUDIO>
[AUDIO_380] <NEXT_AUDIO>
[AUDIO_381] <SPK_2><EOT><NEXT_AUDIO>    ← 자동 후보, 미검수
[AUDIO_382] <NEXT_AUDIO>
…
[AUDIO_390] <SPK_4><ONSET><NEXT_AUDIO>
…
[AUDIO_397] <SPK_2><ONSET><NEXT_AUDIO>
…
[AUDIO_405] expert <NEXT_AUDIO>
```

`marketing … expert`가 이어진다는 사실만으로 EOT를 오답으로 확정하지 않는다. 발언권을 넘겼다가 재개했을 수도 있어 청취가 필요하다. 이 후보는 기준 offset **84.576초**, 목표 방출 **84.96초**, 라벨 판단에 사용한 미래 끝 **87.576초**로 분리했다. 미래는 보존된 후보 생성의 근거이지 84.96초 모델 입력이 아니다. [[output-phase2-eot-review-framework]]

## 실제 ID·검사 결과

블록 381의 출력 ID는 `[151718, 151722, 151705]`다. 각각 `<SPK_2>`, `<EOT>`, `<NEXT_AUDIO>`이며 다음 토큰 예측 관계는 `AUDIO→SPK_2→EOT→NEXT`다. **진단용** next-token label을 함께 저장했지만 검수 승인된 학습 파일은 아니다.

| 항목 | 결과 |
|---|---:|
| 전체 위치 | 1,101 |
| audio / NEXT | 480 / 480 |
| lexical BPE / selector | 104 / 23 |
| ONSET / EOT 후보 | 9 / 5 |
| NEXT-only 블록 | 376 |
| 활동 주석 0명 / 1명 / 2명 블록 | 27 / 415 / 38 |
| 실제 PCM 길이·tokenizer 해시·단어 재토큰화 | 통과 |
| 480블록 ID 일치·화자별 전사 복원·명시적 이벤트 귀속 | 통과 |

끝의 `okay`, `um`은 지연 규약상 k=480,483에 남아 pending으로 보존했다. crop 끝을 녹음 EOF로 바꿔 강제 EOT/flush하지 않았다. 특수 토큰은 시험용 registry이며 체크포인트 embedding 확장은 하지 않았다. 제한된 영어 정규화와 동결 TN의 완전 일치도 아직 검증하지 않았다.

## 열어볼 파일과 재현

- [실제 mono 음성 38.4초](../../raw/sources/experiments/2026-09-13-phase2-sequence-probe/mono-crop.wav)
- [전체 480블록 표](../../raw/sources/experiments/2026-09-14-phase2-sequence-replay/blocks.md)
- [블록 ID JSON](../../raw/sources/experiments/2026-09-14-phase2-sequence-replay/blocks.json) · [전체 시퀀스](../../raw/sources/experiments/2026-09-14-phase2-sequence-replay/sequence.json)
- [미검수 EOT 후보 5개](../../raw/sources/experiments/2026-09-14-phase2-sequence-replay/eot-review-candidates.json) · [검증 manifest](../../raw/sources/experiments/2026-09-14-phase2-sequence-replay/manifest.json)
- 생성 코드: `experiments/p2_replay_sequence.py`. 새 출력 폴더만 허용하며 기존 보존본을 덮어쓰지 않는다.

```sh
/Users/taesookim/anaconda3/envs/vapasr-local/bin/python experiments/p2_replay_sequence.py \
  --source raw/sources/experiments/2026-09-13-phase2-sequence-probe \
  --tokenizer /Users/taesookim/Desktop/VAPKT-models/hf-E2-final-thinker-mlx/tokenizer.json \
  --output <새-출력-폴더>
```

이번 작업은 실제 음성에 대응하는 시퀀스를 보여주는 재현 단계다. 프레임워크의 독립 청취·누락 감사·라벨 승인 루프를 완료한 것으로 해석하지 않는다.
