---
type: output
status: active
created: 2026-09-05
updated: 2026-09-05
summary: Stage 1 단일 화자 mono 스트리밍 ASR 파일럿의 데이터 준비·overfit·타이밍 판정을 누적하는 실행 보고서
sources:
  - [[source-stage1-mono-overfit-600-timing]]
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

@600 결과는 [[source-stage1-mono-overfit-600-timing]]에 상세히 기록한다.

## @600 타이밍 잠정 판정

- **KO: 학습 가능성 관문 통과.** 418개 매칭 토큰 모두 증거 뒤에 방출됐고,
  p50 189 ms는 설계값 160 ms에 정렬 오차 약 30 ms를 더한 위치다.
- **EN: 중심 지연은 통과, evidence-time 꼬리는 보류.** p50/p90은 201/227 ms로
  KO와 같지만 `viol80=7.3%`다. @300의 76%보다 빠르게 감소 중이므로 @900–1500에서
  WER과 함께 0으로 수렴하는지 본다.
- **실시간 deadline은 미통과.** tick p99가 KO 186 ms, EN 151 ms로 80 ms보다 크다.
  이는 학습 가능성 판정과 분리해 기록하되 배포 전 해결해야 하는 구조적 병목이다.

KO 분포는 음향 증거를 따라 방출한 결과와 일치하지만 고정 16개 overfit만으로 이를
증명하지는 못한다. 같은 발화 앞에 0.4–1.6초 무음을 추가했을 때 방출 chunk도 정확히
그만큼 이동하는 leading-silence shift test로 절대 위치 암기를 배제한다.

## EN `viol80` 판정 절차

@900–1500에서 단순 비율만 보지 않고 다음 순서로 판단한다.

1. 위반 토큰별로 발화 ID, 토큰, 앞뒤 문맥, 참조 종료시각, 방출 chunk를 덤프한다.
2. 참조·가설에서 한 번만 나오는 토큰 또는 주변 n-gram이 고유한 토큰의 위반율을
   따로 낸다.
3. 반복 토큰의 모호한 대응과 ForcedAligner 오류를 분리한다.
4. WER이 0에 가까운 표본에서도 `viol80`이 남으면 실제 조기 방출로 판정한다.

이 절차는 매칭 방식을 지연값에 맞춰 고르는 것을 피한다. 시간에 유리한 참조 위치로
재매칭하면 실제 위반을 숨길 수 있으므로, 원 단조 매칭 수치와 고유 문맥 감사 수치를
둘 다 보존한다.

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

## 다음 판정 순서

1. 기존 overfit의 @900/@1200/@1500에서 EN WER·`viol80` 추세 확인.
2. KO·EN overfit 표본의 leading-silence shift test로 방출의 음향 의존성 확인.
3. deferred `<NEXT_AUDIO>`+다음 audio 2-token forward로 decoder 호출 수를 줄여 tick 재측정.
4. 새 KsponSpeech 발음형 manifest·`align2/`의 QC 완료.
5. 새 규약 데이터로 overfit을 다시 실행해 내용·타이밍 재현.
6. Qwen 오프라인과 Nemotron RNN-T `[56,0]` 대조군 측정.
7. 6,000-step 파일럿 진행.
