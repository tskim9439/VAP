---
type: source
status: active
created: 2026-09-07
updated: 2026-09-07
summary: Stage 1 mono 개선 run A/B에서 full FT가 LoRA보다 우수했으나 RNN-T 격차와 KO 조기 방출이 남은 결과
raw_path: raw/sources/experiments/2026-09-07-s1-runAB/
observed: 2026-09-07
raw_authors:
  - tskim
---

# Stage 1 mono 개선 run A/B

## 무엇인가

6,000-step 파일럿에서 확인한 EN 내용 오류와 KO 과소 방출을 개선하기 위해 실행한
두 학습 조건의 결과다. 두 조건 모두 증류 adapter 초기화, KO `next_weight=0.15`,
Nemotron encoder 동결, 유효 배치 EN 96 / KO 384, 4,470 step(15 epoch)을 사용했다.

- A: Qwen thinker LoRA r16, thinker LR `4e-4`
- B: Qwen thinker 0.6B full fine-tuning, thinker LR `4e-5`

## 선택 평가

`--select`는 dev-clean 50 streams, dev-other 50 streams, kspon-dev 300 utterances에서
bias 0, `delta=2`로 평가했다. 따라서 아래 값은 방향 선택에는 유효하지만 전체 dev
확정값이나 공개 보고값은 아니다.

| dev | A LoRA r16 @4000 | B full FT @4470 | 파일럿 @6000 | RNN-T `[56,0]` |
|---|---:|---:|---:|---:|
| dev-clean WER | 0.212 | **0.169** | 0.228 | 0.044 |
| dev-other WER | 0.304 | **0.237** | 0.311 | 0.082 |
| kspon-dev CER | 0.474 | **0.438** | 0.623 | 0.202 |

B는 A보다 상대 오류를 dev-clean 20.3%, dev-other 22.0%, kspon-dev 7.6% 줄였다.
세 평가 세트에서 방향이 일치하므로 다음 주력 조건은 B로 정한다. 다만 이 비교는
LoRA rank와 LR을 폭넓게 탐색한 것이 아니므로, 결과를 “LoRA의 본질적 용량 한계”로
일반화하지 않고 **현재 A 설정 대비 full FT 우위**로 한정한다.

## 수렴 판독

큰 select 표본에서 A는 @4000 이후 세 세트가 모두 정체 또는 반등했다. B는
@4000→@4470에서 EN WER가 0.167→0.169, 0.235→0.237로 소폭 반등했고 KO CER만
0.448→0.438로 개선됐다. 학습 text top-1은 약 0.98이고 마지막 LR은 사실상 0이다.

유효 노출은 EN 약 16.3 pass, KO 약 13.9 pass다. 따라서 B 전체를 단순히
“아직 under-trained”로 판정할 수 없다. KO는 같은 조건의 추가 학습 여지가 있지만,
EN은 현재 데이터·정렬 목표의 일반화 정체가 시작됐다는 해석이 더 안전하다.

## 방출과 지연

| 세트 | tok/chunk / 참조 | 지연 p50 / p90 / p99 | `viol80` | tick p99 |
|---|---:|---:|---:|---:|
| dev-clean | 0.222 / 0.219 | 221 / 294 / 387 ms | 0.27% | 105 ms |
| dev-other | 0.215 / 0.213 | 237 / 330 / 459 ms | 0.30% | 105 ms |
| kspon-dev | 0.288 / 0.273 | 176 / 322 / 797 ms | 4.85% | 145 ms |

파일럿의 KO 과소 방출은 해소됐지만 참조보다 5.3% 많이 방출하며 조기 방출 꼬리가
남았다. 또한 지연은 정답과 매칭된 토큰을 중심으로 계산하므로 CER 43.8%인 KO에서
이 통계만으로 전체 토큰의 예측 가능한 방출을 입증할 수 없다.

## 원본

- `raw/sources/experiments/2026-09-07-s1-runAB/results-A-select.json`
- `raw/sources/experiments/2026-09-07-s1-runAB/results-B-select.json`
- `raw/sources/experiments/2026-09-07-s1-runAB/select-A.log`
- `raw/sources/experiments/2026-09-07-s1-runAB/select-B.log`
- `raw/sources/experiments/2026-09-07-s1-runAB/slurm-A.out`
- `raw/sources/experiments/2026-09-07-s1-runAB/slurm-B.out`

## 연결

- 종합 판정과 후속 계획: [[output-stage1-mono-pilot]]
- 중간 파일럿: [[source-stage1-mono-pilot-6000-sentinel-partial]]

