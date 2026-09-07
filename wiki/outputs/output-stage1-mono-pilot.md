---
type: output
status: active
created: 2026-09-05
updated: 2026-09-07
summary: Stage 1 mono는 full FT로 개선됐으나 RNN-T 동급 정확도와 예측 가능한 지연에는 데이터·목표·런타임 개선이 필요
sources:
  - [[source-streaming-speech-llm-related-work]]
  - [[source-stage1-mono-overfit-600-timing]]
  - [[source-stage1-mono-overfit-1500]]
  - [[source-stage1-mono-pilot-6000-sentinel-partial]]
  - [[source-stage1-mono-run-ab]]
  - [[source-asr-output-style-probe]]
  - [[decision-mono-input]]
  - [[decision-asr-backbone]]
---

# Stage 1 mono 스트리밍 ASR 파일럿

## 목적과 구성

Nemotron `[56,0]`의 80 ms mono 특징을 adapter로 Qwen3-ASR thinker에 넣고,
`text 0..M → <NEXT_AUDIO>` 방출 규약이 내용과 시점을 함께 학습하는지 검증한다.
파일럿은 절대 WER 관문보다 인식 하락 추세, RNN-T 대비 격차, evidence-time 준수,
설계 지연 추종과 폭주 여부를 본다.

세부 실행 규약은 `plans/stage1-mono-pilot.md`에 있다. 데이터 타깃은
[[asr-text-normalization]]을 따르며, 기존 KsponSpeech 철자형 정렬과 새 발음형
`align2/`를 구분해 보존한다.

## Overfit 진행

| step | KO CER | EN WER | KO tok/chunk | EN tok/chunk | 해석 |
|---:|---:|---:|---:|---:|---|
| 300 | 0.364 | 0.917 | — | 0.141 | EN·KO 내용 미수렴, 특히 EN 방출률 부족 |
| 600 | 0.000 | 0.141 | 0.267 / 참조 0.267 | 0.211 / 참조 0.218 | 두 언어의 방출률과 중심 지연 학습 |
| 900 | 0.000 | 0.000 | 0.267 / 참조 0.267 | 0.218 / 참조 0.218 | 기능 관문 전 항목 통과 |
| 1200 | 0.000 | 0.000 | 0.267 / 참조 0.267 | 0.218 / 참조 0.218 | @900 결과 유지 |
| 1500 | 0.000 | 0.000 | 0.267 / 참조 0.267 | 0.218 / 참조 0.218 | @900 결과 유지 |

@600 중간 결과는 [[source-stage1-mono-overfit-600-timing]], 최종 결과는
[[source-stage1-mono-overfit-1500]]에 기록한다.

## @1500 기능 관문 판정

- **EN·KO 내용과 방출 타이밍 통과.** @900부터 WER/CER와 `viol80`이 모두 0이고
  tok/chunk가 참조율과 정확히 같으며 M 강제는 없다. @1500까지 변화 없이 유지됐다.
- **지연 설계 일치.** @1500 EN p50/p90/p99는 201/231/240 ms, KO는
  189/232/312 ms로 `delta=2` 설계값 160 ms와 정렬 오차 위에 모였다.
- **@600 EN 꼬리 해소.** WER 14.1%일 때의 `viol80=7.3%`는 내용 수렴과 함께
  @900에서 0이 됐고 이후 재발하지 않았다.
- **실시간성은 별도 미통과.** 경합 전 decoder-only tick p99 151–186 ms가 이미
  80 ms보다 크다. @1200 이후 679–755 ms는 GPU 경합으로 오염돼 속도 판정에서 제외한다.

KO 분포는 음향 증거를 따라 방출한 결과와 일치하지만 고정 16개 overfit만으로 이를
증명하지는 못한다. 같은 발화 앞에 0.4–1.6초 무음을 추가했을 때 방출 chunk도 정확히
그만큼 이동하는 leading-silence shift test로 절대 위치 암기를 배제한다.

## EN `viol80` 판정 결과

@900에서 EN WER과 `viol80`이 동시에 0이 됐고 @1500까지 유지됐다. 따라서 현재
overfit 관문을 막는 evidence-time 문제는 해소됐다. 반복 토큰의 `difflib` 모호성은
dev에서 WER이 남아 있을 때 다시 나타날 수 있으므로, dev 평가에서 `viol80>0`이면
위반 토큰의 발화 ID·문맥·참조 시각·방출 chunk를 덤프해 감사한다.

## 실시간성 보강 지표

현재 `backlog_p99`는 데이터셋의 **토큰 이월**이고 실행 큐 backlog가 아니다.
배포 판정에는 chunk별 service time `s_k`로 다음 누적 지연도 계산해야 한다.

```text
runtime_lag_0 = 0
runtime_lag_k = max(0, runtime_lag_(k-1) + s_k - 80 ms)
```

tick p50·p90·p99와 함께 평균 service time, 80 ms deadline miss율, 최대 연속 miss,
runtime lag p99/max, 장시간 RTF를 보고한다. 여러 창을 묶는 배치는 throughput과
다중 세션에는 유효하지만 단일 스트림 latency의 해결책인지 별도로 구분한다.

현 구현의 tick은 `stream_decode` 안의 thinker 호출만 포함한다. encoder는 캐시됐고,
adapter의 전체-stream `chunk_embed`, host/device 입력, stream-end flush는 타이머 밖이다.
따라서 현재 151–186 ms 자체가 end-to-end 지연이 아니라 **decoder-only 하한**이다.
배포용 측정은 실제 80 ms PCM 입력부터 encoder·adapter·thinker·출력까지 포함해야 한다.

또한 현재 한 chunk의 thinker 호출 수는 최대 `M+2`다: audio 1회, 방출 텍스트 최대
M회, `<NEXT_AUDIO>`를 KV에 넣는 1회다. 마지막 `<NEXT_AUDIO>` 반영을 다음 tick으로
미루고 `[<NEXT_AUDIO>, next_audio]` 두 위치를 한 forward로 처리하면 causal 의미를
유지하면서 정상 무방출 chunk의 호출을 2회에서 1회로 줄일 수 있다. 이 deferred-control
fusion을 먼저 구현·측정하고, 그 다음 CUDA graph·fused runtime을 비교한다.

## 완료된 파일럿 진행 이력

1. KsponSpeech 발음형 manifest와 LibriSpeech `align2/`를 생성했다.
2. 새 규약 `overfit-v2`가 KO 숫자 읽기와 EN 발화 경계를 포함한 표적 표본에서
   @900 WER/CER와 `viol80` 0을 재현했다.
3. random-init adapter·LoRA 6,000-step 파일럿으로 학습 가능성과 EN 개선 추세를
   확인했지만, 큰 RNN-T 격차와 KO 과소 방출을 발견했다.
4. 교사강제 및 자유실행 S/D/I 진단으로 EN 내용 학습 부족과 KO 삭제 중심의 방출
   붕괴를 분리했다.
5. 증류 adapter 초기화와 KO `next_weight=0.15`를 적용한 A/B 비교를 실행해
   B(full FT)를 다음 조건으로 선택했다.

leading-silence shift test는 overfit의 절대 위치 암기를 구분하는 유용한 보강 검사로
남아 있다. 이는 Stage 2 착수의 필수 차단 조건은 아니지만, 타이밍 주장을 강화하려면
최종 평가 전에 수행한다.

## 개선 run A/B 최종 결과

6,000-step 파일럿의 원인 진단 후 증류 adapter 초기화와 KO `next_weight=0.15`를
공통 적용하고, A(LoRA r16)와 B(thinker full FT)를 4,470 step 비교했다. 상세 원본과
수렴 곡선은 [[source-stage1-mono-run-ab]]에 기록한다.

| dev | A LoRA r16 @4000 | **B full FT @4470** | 파일럿 @6000 | RNN-T `[56,0]` |
|---|---:|---:|---:|---:|
| dev-clean WER | 0.212 | **0.169** | 0.228 | 0.044 |
| dev-other WER | 0.304 | **0.237** | 0.311 | 0.082 |
| kspon-dev CER | 0.474 | **0.438** | 0.623 | 0.202 |

**판정은 B(full FT) 채택이다.** B는 A보다 상대 오류를 EN 두 세트에서 20–22%,
KO에서 7.6% 줄였으며 세 세트의 방향이 일치한다. 다만 A는 rank·LR 전체를 탐색한
실험이 아니므로 “LoRA의 본질적 용량 한계”가 아니라 **현재 LoRA r16 설정 대비
full FT 우위**로 판정을 제한한다.

KO 방출률은 0.288로 참조 0.273에 도달해 파일럿의 과소 방출은 해소됐다. 그러나
KO `viol80=4.85%`, 지연 p99 797 ms가 남았고 EN도 p99가 387–459 ms다. tick p99는
105–145 ms로 80 ms 관문을 통과하지 못했다.

## RNN-T 동급 정확도와 예측 가능한 타이밍의 가능성

### 결론

**연구를 계속할 가능성은 충분하지만, 현재 recipe에 같은 데이터를 더 오래 학습하는
것만으로 RNN-T 동급에 도달할 가능성은 낮다.** Stage 2의 약 1,900 h 데이터 확장과
정렬·방출 목표 및 런타임을 함께 개선하면 RNN-T에 근접할 가능성은 열려 있다.

| 목표 | 현재 평가 |
|---|---|
| 오디오 조건부 텍스트와 약 200 ms 중심 방출 학습 | 가능성 높음 — overfit과 dev에서 확인 |
| Stage 2 후 RNN-T 1.5배 이내 | 가능성 중간 — 데이터 차이를 줄인 뒤 검증 필요 |
| RNN-T 상대 10% 이내와 지연 관문 동시 달성 | 불확실 — 현재 구조 그대로는 가능성 낮음 |
| 같은 200 h를 30 epoch 이상 반복 | 주력 실험으로 비권장 |
| 현재 디코더로 tick p99 80 ms | 구조·런타임 최적화 없이는 가능성 낮음 |

### 긍정적 근거

1. **구조의 기본 학습 가능성은 입증됐다.** overfit에서 내용·방출률·설계 지연을
   동시에 맞췄고, dev에서도 B가 세 언어 조건의 오류를 일관되게 낮췄다.
2. **같은 frozen encoder의 RNN-T 성능이 낮다.** 이는 입력 특징에 필요한 음향 정보가
   존재한다는 강한 증거이며, 현재 격차의 주원인을 thinker·adapter·목표 함수 쪽에서
   찾을 수 있게 한다.
3. **구조 계열 자체의 선례가 있다.** [[source-muse-voice-transcribe|Muse]]는 80 ms
   soft token과 가변 방출 구조를 사용한다. 2026년 decoder-only LLM 연구도 연속 음성
   구간과 텍스트를 교차 처리하면서 학습된 read/write 정책과 latency loss로 기존
   streaming baseline과 경쟁했다
   ([논문](https://arxiv.org/html/2601.22779)).

### 부정적 근거

1. **남은 정확도 격차가 크다.** B/RNN-T 오류비는 dev-clean 3.84배, dev-other
   2.89배, KO 2.17배다. RNN-T 상대 10% 이내에 들려면 B에서 오류를 각각 약
   71%, 62%, 49% 더 줄여야 한다.
2. **더 긴 동일 데이터 학습의 근거가 약하다.** B는 이미 EN 약 16.3 pass, KO 약
   13.9 pass를 보았고 train text top-1은 약 0.98이다. 큰 dev에서 @4000→@4470의
   EN은 소폭 반등하고 KO만 개선됐다. “마지막까지 계속 하락해 전체가 under-trained”라는
   판독은 정확하지 않다.
3. **현재 평가는 전체 dev가 아니다.** 선택 표본은 EN 50/50 streams와 KO 300
   utterances다. 방향 선택에는 충분하지만 full-dev 확정치와 신뢰구간 없이 RNN-T
   격차나 best checkpoint를 최종 주장할 수 없다.
4. **좋은 중심 지연이 전체 타이밍을 보장하지 않는다.** 현재 지연은 주로 정확히
   매칭된 토큰에서 측정된다. KO CER 43.8%에서는 삭제·치환된 상당수 토큰이 지연
   분포에서 빠지므로 p50 176 ms만으로 “타이밍 해결”을 선언하면 낙관적이다.
5. **단일 forced-alignment 시점의 CE 목표는 구조적으로 불리하다.** RNN-T는 여러
   정렬 경로를 주변화하지만 현재 규약은 정렬 오차와 허용 가능한 방출 시점의 다양성을
   한 점에 고정한다. Alignment-Restricted RNN-T도 한 점 대신 허용 구간을 사용해
   WER–latency 절충을 제어한다
   ([논문](https://arxiv.org/abs/2011.03072)).
6. **현재 tick은 end-to-end 지연의 하한이다.** encoder와 입출력을 제외한 측정도
   p99 80 ms를 넘는다. ragged KV batching은 동시 세션 throughput에는 도움이 되지만
   단일 스트림 deadline을 자동으로 해결하지 않는다.

## 권장 Stage 2 계획

### 1. 데이터와 대조군

- 1,930 h alignment 생성 전에 [[output-asr-tn-v1-spec|`asr-tn-v1.0.0`]]의 lexical
  구현·전체 transcript audit·fingerprint 동결 관문을 통과한다. display 계약은
  alignment와 독립이지만 full-FT 시작 전에는 동결한다.
- 영어는 `lexical_text`와 `display_text`를 분리한다. 기존 WER·정렬·주 streaming
  target은 lexical view로 고정하고, 출처가 검증된 대소문자·문장부호 label만
  display 보조 loss에 사용한다. full-FT가 Qwen의 기존 표기 능력을 잊지 않는지는
  고정 display subset으로 별도 감시한다.
- B를 주력 초기값으로 삼아 LibriSpeech 960 h와 KsponSpeech 전체로 확장한다.
- 같은 200 h의 30 epoch 연장은 짧은 통제 실험으로만 두고 주력 GPU 시간을 쓰지 않는다.
- 200/500/1,000/1,900 h 지점에서 동일한 full-dev와 정규화 규약으로 scaling curve를
  남긴다.
- 기존 RNN-T는 훨씬 많은 데이터로 학습됐으므로, 구조 비교용으로 같은 200 h 또는
  같은 Stage 2 데이터의 RNN-T 대조군을 추가한다.

### 2. 목표 함수

- 한 점의 `delta` 정답 대신 `[t_end + delta_min, t_end + delta_max]` 허용 window,
  soft target 또는 정렬 주변화를 비교한다.
- 조기 방출·지연·삭제를 별도 비용으로 두는 emission/read-write loss를 추가한다.
- streaming CE와 offline ASR CE를 공동 학습하고, 필요하면 학습 전용 CTC 또는 RNN-T
  보조 head를 ablation으로 둔다. 이는 추론 구조를 RNN-T로 되돌리는 것이 아니라
  음향 정렬 regularizer로만 사용한다.
- KO `next_weight`는 0.15/0.175/0.2/0.25의 짧은 Pareto sweep으로 삭제와 조기 방출을
  함께 비교한다. 최소 지연을 명시적으로 최적화하는 transducer 연구도 큰 latency
  감소를 작은 WER 비용으로 달성한 선례가 있다
  ([Minimum Latency Training](https://www.isca-archive.org/interspeech_2022/shinohara22_interspeech.html)).

### 3. 타이밍 평가

- 매칭 토큰 지연 외에 모든 참조 토큰을 대상으로 deadline 내 정확 방출률을 보고한다.
- 삭제는 `미방출`로 포함한 deadline recall, 전체 가설 기준 조기 방출률, `delta` 변경에
  따른 지연 분포 이동을 측정한다.
- 전용 GPU에서 PCM→Nemotron encoder→adapter→thinker→출력까지 end-to-end service
  time, deadline miss율과 누적 runtime lag를 측정한다.

### 4. 런타임

- `<NEXT_AUDIO>` 예측 직후 별도 forward하지 않고 다음 tick에서
  `[<NEXT_AUDIO>, next_audio]`를 한 forward로 처리하는 deferred-control fusion을
  먼저 검증한다.
- 이후 CUDA graph·compile·fused kernel을 단일 스트림 latency 기준으로 비교하고,
  ragged KV batching은 다중 스트림 throughput 항목으로 분리한다.

## 계속·재설계 관문

- 500 h에서 RNN-T 오류비가 2배 이하로 줄어드는지 본다.
- 데이터를 두 배로 늘려도 상대 오류가 10% 미만 감소하고 RNN-T 격차가 2배를 넘으면,
  데이터 추가보다 정렬·방출 목표를 먼저 재설계한다.
- 1,000–1,900 h에서도 RNN-T 오류비가 1.5배 아래로 내려오지 않으면 현재 hard-δ
  IS-SLM 규약을 주 구조로 유지할 근거가 약해진다.
- 최종 성공 조건은 정확도만이 아니라 bias 0 방출률 ±10%, 모든 참조 토큰 기반 조기
  방출 관문, end-to-end tick p99 80 ms를 동시에 만족하는 것이다.

현재 판정은 **조건부 계속**이다. B와 데이터 확장은 합리적이지만, 성공 경로는
`full FT + 더 많은 epoch`가 아니라 **full FT + 데이터 확장 + 정렬을 허용하는 목표 +
명시적 지연 최적화 + 디코더 호출 구조 개선**이다.
