# checkpoint-35000 발화 단위 ASR 평가 — 2026-09-22

공통 평가 도구로 11,246개 원본 발화를 δ=2·4에서 평가했다.
총 22,492건을 처리했으며 예상 ID×δ 조합 대비 누락·중복·추가 행은 모두 0이다.
네 세트 모두 δ=4의 오류율이 낮았다.

| 세트 | 발화 수 | 주지표 | δ=2 | δ=4 |
|---|---:|---|---:|---:|
| LibriSpeech test-clean | 2,620 | WER | 6.63% | 4.74% |
| LibriSpeech test-other | 2,939 | WER | 14.19% | 11.34% |
| KsponSpeech eval_clean | 3,000 | CER, 공백 제외 | 13.18% | 12.11% |
| KsponSpeech eval_other | 2,687 | CER, 공백 제외 | 14.21% | 12.71% |

**KsponSpeech eval_other는 원래 3,000개 중 서버에 있는 2,687개 부분집합이다.**
누락 313개 목록은 [coverage.json](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/coverage.json)에 보존했다.
공개 전량 점수와 직접 같은 조건이라고 볼 수 없다.

## 평가 조건

- 체크포인트: `/soundai/Model/VAPASR/hf-approved-qwen-v1-e10-n2/checkpoint-35000`.
- `trainer_state.json`에서 global_step=35000 확인. 저장된 학습 encoder와 thinker/adapter 사용.
- 원본 오디오 뒤에 **디지털 무음 정확히 1초** 추가. 앞 무음 추가나 발화 연결 없음.
- 발화마다 모델의 KV cache 초기화, greedy, next_bias=0, δ=2/4.
- 음향 입력 처리 후 기존 EMPTY_AUDIO flush 최대 8회. 첫 빈 flush에서 종료.
- FP32, TF32 비활성화. 원출력과 원전사 보존, 채점 시에만 asr-tn-v1.3.0 적용.
- 전체 오류 수 / 전체 참조 단위 수로 micro 집계. 실패 사례를 제외한 평균이 아님.

이 실험은 발화 단위 인식 정확도 평가다. δ=2/4의 명목 지연은 160/320 ms지만,
이번 결과만으로 실제 증거 대비 지연이나 실시간 처리 관문을 통과했다고 판정하지 않는다.
토큰별 방출 청크는 후속 분석용으로 저장했다.

## 오류 분해

아래 수치는 EN은 단어, KO는 공백 제외 문자 기준이다.

| 세트 | δ | S | D | I | 정답 단위 수 |
|---|---:|---:|---:|---:|---:|
| test-clean | 2 | 2,481 | 331 | 676 | 52,576 |
| test-clean | 4 | 1,940 | 242 | 310 | 52,576 |
| test-other | 2 | 5,440 | 858 | 1,132 | 52,343 |
| test-other | 4 | 4,542 | 700 | 693 | 52,343 |
| eval_clean | 2 | 4,190 | 1,332 | 807 | 48,019 |
| eval_clean | 4 | 3,810 | 1,365 | 642 | 48,019 |
| eval_other | 2 | 5,836 | 1,879 | 1,071 | 61,829 |
| eval_other | 4 | 5,187 | 1,844 | 828 | 61,829 |

δ=4에서는 네 세트 모두 치환과 삽입이 감소했다. 삭제는 영어 두 세트와
KO eval_other에서 감소했지만 KO eval_clean에서는 1,332→1,365로 소폭 증가했다.
즉, 이번 차이를 단순히 끝부분 삭제가 줄어든 효과로만 해석할 수는 없다.

한국어 보조지표:

| 세트 | δ | 공백 포함 CER | WER |
|---|---:|---:|---:|
| eval_clean | 2 | 13.18% | 30.52% |
| eval_clean | 4 | 12.11% | 28.04% |
| eval_other | 2 | 14.59% | 36.25% |
| eval_other | 4 | 13.02% | 32.83% |

한국어 보조지표도 프로젝트 TN 규약에 따른 값이며, 외부 벤치마크의 동일한
라벨 해석·정규화 규약을 확인하기 전에는 공식 점수와 동등하다고 단정하지 않는다.

## 검증과 처리량

- 패딩·편집 오류 집계·청크 상태 전환 단위 테스트 5개 통과.
- 스모크 16건 + 본 실행 16건에서 기존 단일 디코딩과 배치 디코딩의
  토큰 ID·방출 청크·강제 NEXT 수·flush 횟수 일치.
- 초기 BF16 스모크에서 쉼표 한 토큰의 차이를 발견해 FP32로 재검증한 뒤 전량 실행.
- manifest의 `(ID, delta)` 전 조합과 결과 전수 대조: 22,492개 일치, 중복/누락/추가 0.
- 모든 결과의 tail_s=1.0 및 S+D+I, N_hyp=N_ref−D+I 일관성 확인.
- 강제 NEXT는 전체 5회(KO eval_clean δ=4: 1회, eval_other δ=2/4: 각 2회).
  해당 발화도 모두 점수에 포함했다.
- mxc GPU 0·2 두 장, GPU당 디코더 배치 128. encoder는 발화별 실행 후 두 δ가 특징 공유.
- 모델 적재 후 전량 추론·채점 시간: 느린 worker 기준 1,356초, 약 **22분 36초**.
  준비·스모크 시간은 이 수치에서 제외한다.
- 프로세스별 peak allocated GPU memory: 약 26.28 / 25.65 GB.
  처리량 측정이며 단일 스트림의 실시간 지연 측정값은 아니다.

## 재실행과 산출물

[평가 규약 및 실행 방법](single-turn-asr-evaluation.md)

- [최종 summary.json](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/full/summary.json)
- [worker 0 발화별 결과](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/full/predictions-rank0.jsonl)
- [worker 1 발화별 결과](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/full/predictions-rank1.jsonl)
- [검증 기록](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/verification.json)
- [실행 라이브러리 버전](../raw/sources/experiments/2026-09-22-single-turn-35000-evaluation/full/environment.json)

서버 결과: `/soundai/users/tskim/VAPKT-data/results/single-turn-35000-d2-d4-v1/`.
최종 집계의 `complete=true` 확인 후 두 GPU worker의 종료를 확인했다.
