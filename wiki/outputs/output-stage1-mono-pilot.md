---
type: output
status: active
created: 2026-09-05
updated: 2026-09-06
summary: Stage 1 mono 스트리밍 ASR은 @900 overfit 기능 관문을 통과했고 새 정규화·정렬 규약 재검증을 대기 중
sources:
  - [[source-stage1-mono-overfit-600-timing]]
  - [[source-stage1-mono-overfit-1500]]
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

## 다음 판정 순서

1. 새 KsponSpeech 발음형 manifest·LibriSpeech `align2/` 완료를 대기한다.
2. `overfit-v2`가 선택한 ID를 기록하고 KO 구규약 대비 변경 표본 수와 EN 다중발화
   경계 검사 수가 각각 1 이상인지 확인한다. 0이면 표적 회귀 표본을 따로 만든다.
3. 새 규약 `overfit-v2`를 900 step 실행해 KO 숫자 읽기와 EN 발화 경계를 재검증한다.
4. 결과가 같으면 Qwen 오프라인과 Nemotron RNN-T `[56,0]` 대조군을 측정한다.
5. 6,000-step 파일럿을 진행한다.
6. 최종 체크포인트에서 한가한 GPU로 decoder-only 및 end-to-end tick을 단독 측정한다.
7. 필요하면 deferred `<NEXT_AUDIO>`+다음 audio forward, CUDA graph 순으로 최적화한다.

leading-silence shift test는 overfit의 절대 위치 암기를 구분하는 유용한 보강 검사지만,
새 규약 overfit과 dev 일반화 평가가 이어지므로 6,000-step 착수의 필수 차단 조건으로
두지는 않는다.
