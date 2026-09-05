---
type: source
status: active
created: 2026-09-05
updated: 2026-09-05
summary: Stage 1 mono overfit @600에서 KO 방출 타이밍이 통과하고 EN 증거 전 방출과 실시간 tick 병목이 남은 결과
raw_path: raw/sources/experiments/2026-09-05-stage1-mono-overfit-600-timing.md
observed: 2026-09-05
raw_authors:
  - tskim
---

# Stage 1 mono overfit @600 타이밍

## 무엇인가

단일 화자 mono 16개씩을 반복 학습한 `@600` 체크포인트에서, `delta=2`의 설계 지연
160 ms가 실제 토큰 방출에 학습됐는지 확인한 중간 결과다. 내용 정확도와 함께 토큰
방출 지연, evidence-time 위반, 토큰율, 이월, chunk 처리시간을 측정했다.

## 핵심 결과

| 지표 | KO 16 | EN 16 | 판정 |
|---|---:|---:|---|
| 지연 p50 / p90 / p99 | 189 / 232 / 312 ms | 201 / 227 / 494 ms | 둘 다 지연 중심·꼬리 통과 |
| `viol` / `viol80` | 0 / 0 | 9.4% / 7.3% | KO 통과, EN 미통과 |
| tok/chunk / 참조 | 0.267 / 0.267 | 0.211 / 0.218 | 둘 다 ±10% |
| M 강제 | 0 | 0 | 둘 다 통과 |
| 라벨 이월 backlog p99 | 2.85 chunk | 0 | 둘 다 4 이하 |
| tick p99 | 186 ms | 151 ms | 둘 다 80 ms deadline 미통과 |

KO는 매칭 토큰 418개 전부가 증거 뒤에 방출됐고, p50은 설계 지연에 정렬 오차를
더한 값과 일치한다. EN의 지연 중심도 같지만 WER 14% 상태에서 반복 토큰의 단조
매칭과 실제 조기 방출이 섞인 음수 지연 꼬리가 남아 있다.

## 해석상 주의

- 이 결과는 16개 발화를 암기한 overfit 검사다. 파이프라인과 방출 규약의 학습 가능성을
  확인하지만 dev 일반화 성능을 말하지 않는다.
- 고정 발화의 정답 chunk를 위치로 외웠을 가능성도 있으므로, KO 결과만으로 방출 시점의
  음향 의존성을 확정하지 않는다. 선행 무음을 추가해 같은 음성을 이동했을 때 방출도
  같은 chunk 수만큼 이동하는지 확인하는 shift test가 이를 구분한다.
- 현 `difflib.SequenceMatcher`는 반복되는 흔한 토큰의 대응 위치가 유일하지 않다.
  EN `viol80=7.3%`를 모델의 실제 선행 방출로 확정하기 전에 위반 사례의 문맥과 정렬을
  직접 감사해야 한다.
- 코드에서 `backlog_p99`는 실시간 처리 큐가 아니라 정답 토큰을 chunk별로 배치할 때의
  최대 이월량이다. tick deadline miss가 장시간 누적되는지는 서비스 시간 시계열로
  별도 계산해야 한다.
- 현 tick 타이머는 캐시된 feature를 받아 thinker를 순차 호출하는 구간만 잰다.
  `chunk_embed`는 타이머 밖에서 전체 스트림에 미리 계산되고 encoder·입력 전송·flush도
  제외되므로, 151–186 ms는 end-to-end tick이 아니라 decoder-only 하한이다.
- 창 배치는 여러 스트림의 처리량에는 도움을 주지만, 한 스트림 안의 autoregressive
  토큰 단계는 순차적이다. 단일 스트림 latency는 CUDA graph·fused runtime·forward 수
  축소의 효과를 따로 측정해야 한다.

## 이 볼트에 준 영향

- [[output-stage1-mono-pilot]]에 최초의 방출 타이밍 관측과 잠정 판정을 추가했다.
- Stage 1의 다음 판정은 @900–1500의 EN WER·`viol80` 추세, 새 텍스트 규약 overfit,
  위반 토큰 감사까지 보류한다.
- 고정 16개에서 확인한 KO 타이밍을 음향 의존성으로 해석하기 전에 leading-silence
  shift test로 절대 위치 암기를 배제한다.
- 실시간성은 tick p99 하나와 라벨 backlog가 아니라 deadline miss율과 누적 처리 지연을
  함께 보고하도록 보강한다.

## 출처

- 원본: `raw/sources/experiments/2026-09-05-stage1-mono-overfit-600-timing.md`
- 평가 코드: `experiments/s1_train_mono.py`, `vapasr/uslm/mono_model.py`
