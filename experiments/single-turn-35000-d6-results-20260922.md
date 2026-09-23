# checkpoint-35000 δ=6 추가 평가 — 2026-09-22

사용자 요청에 따라 δ=8 평가는 중지하고, 학습 설정에 포함된 δ=6으로
11,246개 발화의 Single Turn ASR 평가를 완료했다.
δ=8 중간 결과는 보존했으며 완료된 벤치마크로 취급하지 않는다.

| 평가 세트 | 발화 수 | 지표 | δ=2 | δ=4 | δ=6 |
|---|---:|---|---:|---:|---:|
| LibriSpeech test-clean | 2,620 | WER | 6.63% | 4.74% | **4.51%** |
| LibriSpeech test-other | 2,939 | WER | 14.19% | 11.34% | **10.88%** |
| KsponSpeech eval_clean | 3,000 | 공백 제외 CER | 13.18% | **12.11%** | 12.16% |
| KsponSpeech eval_other | 2,687 | 공백 제외 CER | 14.21% | **12.71%** | 12.79% |

영어에서는 δ=4보다 오류가 추가로 줄었다(test-clean 122단어, test-other 242단어 감소).
한국어는 eval_clean 24문자, eval_other 46문자 오류가 늘어, 추가 개선은 관측되지 않았다.
작은 한국어 차이에 대한 통계적 유의성 검정은 수행하지 않았다.

KsponSpeech eval_other는 서버에 없는 313개를 제외한 **2,687개 부분집합**이다.
다른 세 세트는 전량이다. δ=2·4·6 모두 동일한 가용 발화 집합을 사용했다.

## 비교 조건

- 체크포인트: `/soundai/Model/VAPASR/hf-approved-qwen-v1-e10-n2/checkpoint-35000`.
- 원본 발화 뒤 무음 1초, 앞 무음 추가 없음. 발화마다 KV cache 초기화.
- greedy, next_bias=0, FP32, TF32 비활성화, EMPTY_AUDIO flush 최대 8회.
- 원전사·모델 원출력을 보존하고 채점에서만 asr-tn-v1.3.0 적용.
- 전체 S+D+I / 전체 정답 단위 수로 집계.
- 이전 δ=2·4 실행과 config를 비교해 **deltas 이외의 설정 차이가 없음**을 확인했다.
  체크포인트 파일 메타데이터, 코드·manifest·TN 지문도 동일하다.

체크포인트 config의 학습 delays는 `[2,3,4,6]`이다. δ=6의 설계상 지연은
480 ms지만, 이 보고서는 ASR 정확도 결과이며 실제 방출 지연이나 실시간성 관문을
측정한 결과는 아니다. 따라서 인식 점수만으로 배포용 δ를 확정하지 않는다.

## δ=6 오류 분해

| 세트 | S | D | I | 정답 단위 수 |
|---|---:|---:|---:|---:|
| test-clean | 1,863 | 223 | 284 | 52,576단어 |
| test-other | 4,405 | 652 | 636 | 52,343단어 |
| eval_clean | 3,849 | 1,356 | 636 | 48,019문자 |
| eval_other | 5,228 | 1,870 | 807 | 61,829문자 |

한국어 보조지표는 eval_clean 공백 포함 CER 12.14% / WER 28.12%,
eval_other 공백 포함 CER 13.06% / WER 32.97%다.

## 실행 및 산출물

mxc GPU 0·2 두 장, GPU당 배치 128로 수행했다. 모델 적재 후 전량 추론·채점은
느린 worker 기준 974.47초(약 16분 15초) 걸렸다.
peak allocated GPU memory는 약 26.27 / 25.62 GB였다.
δ=8 worker 종료를 확인한 뒤 δ=6 worker를 시작했으며, δ=6 완료 후에도 종료를 확인했다.

단일/배치 디코딩 비교 표본 8건에서 토큰 ID·방출 청크·강제 NEXT 수·flush 횟수가
일치했다. 강제 NEXT는 KO eval_clean 2회, eval_other 1회였으며 해당 발화도 점수에 포함했다.
로컬 사본에서도 11,246개 ID가 이전 평가와 같고 누락·중복·추가가 모두 0임을 확인했다.
소스 코드 해시, 발화별 편집 오류 합, 세트별 집계도 검증했다.

- [최종 summary.json](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/delta6/summary.json)
- [전수 검증 기록](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/verification-delta6.json)
- [발화별 결과: worker 0](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/delta6/predictions-rank0.jsonl)
- [발화별 결과: worker 1](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/delta6/predictions-rank1.jsonl)
- [이전 δ=2·4 보고서](single-turn-35000-results-20260922.md)
- [공통 평가 도구·δ=6 재실행 명령](single-turn-asr-evaluation.md)
- [δ=8 취소 기록](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/delta8-cancelled.json)

서버 결과 경로: `/soundai/users/tskim/VAPKT-data/results/single-turn-35000-d6-v1/`.
