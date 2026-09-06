---
author_id: tskim
created: 2026-09-06
source_type: source-drop
---

# Stage 1 mono 6,000-step 파일럿 sentinel 중간 결과

파일럿이 6,000-step 학습을 마쳤고 마지막 sentinel 평가는 진행 중이다. 현재
dev-clean은 끝났고 dev-other와 kspon-dev가 남았다. 아래 값은 bias 0, `delta=2`다.

| step | dev-clean WER | dev-other WER | kspon-dev CER (bias 0 / 최적) | KO tok/chunk (참조 0.273) | EN `viol80` · p50 | KO `viol80` · p50 |
|---:|---:|---:|---|---:|---|---|
| 1000 | 0.556 | 0.582 | 0.851 / 0.759 (`b=1`) | 0.074 | 3.6% · 265 ms | 7.1% · 311 ms |
| 2000 | 0.386 | 0.496 | 0.656 / 0.656 (`b=0`) | 0.230 | 1.8% · 213 ms | 8.0% · 212 ms |
| 3000 | 0.289 | 0.391 | 0.713 / 0.602 (`b=1`) | 0.110 | 1.1% · 216 ms | 5.1% · 283 ms |
| 4000 | 0.230 | 0.303 | 0.679 / 0.627 (`b=1`) | 0.137 | 0.5% · 209 ms | 5.2% · 248 ms |
| 5000 | 0.221 | 0.289 | 0.659 / 0.609 (`b=1`) | 0.140 | 0.5% · 206 ms | 5.6% · 245 ms |
| 6000 | 0.228 | 평가 중 | 평가 중 | — | 0.4% · 193 ms | — |

학습 text loss는 2.05에서 0.96으로, text top-1은 0.50에서 0.75로 변했다.

참고 대조군은 Nemotron RNN-T `[56,0]`이며 dev-clean 4.4%, dev-other 8.2%,
kspon-dev 20.2%다.

## 당시 해석과 실행 계획

- EN은 WER이 전체적으로 하락하고 방출률이 참조와 비슷하며 지연 p50이 약 200 ms다.
  6,000-step dev-clean 0.228은 작은 20-stream sentinel의 잡음 범위로 본다. 절대 WER은
  RNN-T보다 매우 높지만 파일럿의 학습 가능성 목적은 충족한 것으로 읽었다.
- KO는 CER이 전체적으로 개선됐지만 bias 0 tok/chunk가 참조의 약 절반이고 최적 bias가
  주로 1이다. overfit에서는 참조율에 맞았으므로 전체 데이터에서 `<NEXT_AUDIO>` 쪽으로
  기우는 목표 문제로 보고, 6,000-step 결과 뒤 KO `next_weight` 0.2/0.1 sweep을 후보로 뒀다.
- 마지막 평가가 끝나면 ckpt 4000/5000/6000을 bias 0/1, dev 표본
  100/100/500으로 GPU 2·6에서 병렬 선택한다.
- tick p99는 한가한 GPU에서 단독 측정한다.

이 기록은 마지막 dev-other·kspon-dev 평가와 큰 dev 선택 전의 중간 상태다.

