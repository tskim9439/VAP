# 공통 Single Turn ASR 평가 v1

이 도구는 모델별 출력 형식과 무관하게 원본 발화 단위로 평가하기 위한 것이다.
현재 추론 어댑터는 `VapAsrForStreamingASR` mono 체크포인트를 지원한다.
데이터·채점 코드는 `vapasr/data/single_turn_eval.py`, 추론은
`vapasr/hf/batch_decode.py`, 실행은 `experiments/eval_single_turn_asr.py`에 분리했다.

## 고정 규약

| 항목 | 규약 |
|---|---|
| 영어 | LibriSpeech `test-clean`, `test-other` 원본 발화 전량 |
| 한국어 | KsponSpeech `eval_clean`, `eval_other` 원본 발화 전량 중 접근 가능한 파일 |
| 입력 | 원본 mono 16 kHz 오디오 + **끝에 디지털 무음 16,000 samples(1초)** |
| 선행 처리 | 추가 앞 무음·VAD 절단·발화 연결 없음 |
| 모델 상태 | 발화마다 prefix 및 KV cache 초기화; 발화 간 정보 공유 없음 |
| delta | 같은 발화 집합에서 2, 4 각각 평가(명목상 160 / 320 ms) |
| 디코딩 | greedy, next_bias=0, 체크포인트의 blocked IDs·runaway_cap 사용 |
| 연산 정밀도 | 기본 FP32, TF32 비활성화. BF16은 명시적 별도 실험으로만 사용 |
| 종료 | 1초 음향 무음 처리 **후** 기존 EMPTY_AUDIO flush 최대 8회; 첫 텍스트 없는 flush에서 종료 |
| 원문 보존 | 데이터셋 원전사 및 모델 원출력을 저장; 원출력을 TN으로 덮어쓰지 않음 |
| 채점 | 고정된 asr-tn 지문 기록; EN WER, KO 공백 제외 CER(주), 공백 포함 CER, WER |
| 집계 | 전체 S+D+I 합 / 전체 정답 단위 수; 발화별 오류율 단순 평균 아님 |
| 누락/실패 | 누락 ID 목록과 원본 라벨 수·가용 수 기록; 추론 실패는 중단, 성공 사례만 최종 결과로 보고하지 않음 |

KsponSpeech 이중표기·주석은 프로젝트의 고정된 Kspon 파서로 참조만 해석한다.
공백 포함 CER를 계산하지만 프로젝트의 숫자/이중표기 규약을 사용하므로
외부 논문의 공식 점수와 동일한 프로토콜이라고 단정하지 않는다.
한국어 원출력의 숫자/영문 표현 차이는 이 규약에 따라 오차로 남을 수 있다.

LibriSpeech 표준 발화 수는 test-clean 2,620, test-other 2,939,
KsponSpeech eval_clean/eval_other는 각각 3,000을 점검 기준으로 둔다.
`coverage.json`의 `full_standard_set=false`인 세트는 **가용 부분집합 평가**로 표시한다.
전사 자체가 없는 경로, 중복 ID, 예상치 못한 오디오 형식은 준비 단계에서 실패한다.
Kspon raw PCM의 홀수 길이 마지막 1바이트는 제외하고, 이 여부를 발화별로 기록한다.

이 점수는 인식 정확도 평가다. 발화별 토큰 방출 청크와 오디오 끝 이후 방출 수는
저장하지만, 단어 정렬 정답 없이 증거 대비 지연이나 조기 방출률을 추정하지 않는다.
배치 처리 시간은 처리량이며 실서비스의 단일 스트림 tick latency를 뜻하지 않는다.

## 처리량과 정확성

길이가 비슷한 발화를 묶어 thinker 디코딩을 배치화한다. 각 행이 자기 청크와
텍스트/NEXT 상태를 독립적으로 진행하므로 다른 행의 발화 길이에 맞춰
인공적인 NEXT 토큰이나 오디오를 끼워 넣지 않는다.
완료된 행은 결과 수집을 종료하고 나머지 행이 끝날 때까지 더미 입력만 소비한다.

encoder는 파일별 정규화·패딩의 수치 차이를 피하도록 발화별로 실행한다.
그 출력은 GPU 메모리에 두고 δ=2와 δ=4가 공유한다. CPU는 오디오를 병렬로
미리 읽어 RAM에 보관한다. GPU 메모리가 부족하면 디코딩 배치를 절반씩 나눠 재시도한다.

`--verify`는 각 언어·δ의 첫 배치 양끝 표본을 기존 `model.stream_decode`로도
디코딩해 토큰 ID, 방출 청크, 강제 NEXT 수, flush 횟수가 정확히 일치하는지 확인한다.
불일치하면 실행을 중단하고 parity JSON에 두 시퀀스를 기록한다.
짧은 발화·긴 발화가 포함된 `--limit` 스모크를 먼저 실행한 후 전량 평가한다.

초기 BF16 스모크에서 긴 KO 발화 δ=4의 쉼표 한 토큰이 배치/단일 간 달랐다.
평가 정규화로 제거되는 기호이지만 원출력 재현성 검증을 통과시키기 위해
기본 연산을 FP32로 올리고 TF32를 끈다. 이는 체크포인트 가중치를 다시 학습하거나
정밀도 높은 원본 가중치를 복원하는 것이 아니라, 같은 저장된 가중치의 연산 정밀도 변경이다.

## mxc 실행

환경 변수는 아래 셸 프로세스 안에서만 설정한다. wrapper는 GPU 1~2장만 허용하며
중복 GPU ID 및 같은 결과 디렉터리에서 동시 실행하는 것을 막는다.
기존 학습이나 다른 사용자의 프로세스에는 영향을 주지 않는다.

```bash
cd /soundai/users/tskim/VAPKT

# 원본에서 공통 manifest 작성(원본 파일 수정 없음)
/tmp/sa_tskim-vapasr-env-local/bin/python experiments/eval_single_turn_asr.py prepare \
  --libri-root /soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/LibriSpeech \
  --kspon-root /soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/KsponSpeech \
  --out /soundai/users/tskim/VAPKT-data/data/evaluation/single-turn-v1

# 체크포인트, 결과 폴더, GPU ID, 배치 크기, 세트당 제한 수(0=전량), delta 목록(기본 2,4)
bash experiments/run_single_turn_asr_mxc.sh \
  /soundai/Model/VAPASR/hf-approved-qwen-v1-e10-n2/checkpoint-35000 \
  /soundai/users/tskim/VAPKT-data/results/single-turn-35000-smoke-fp32-v1 \
  0,2 16 4

bash experiments/run_single_turn_asr_mxc.sh \
  /soundai/Model/VAPASR/hf-approved-qwen-v1-e10-n2/checkpoint-35000 \
  /soundai/users/tskim/VAPKT-data/results/single-turn-35000-d2-d4-v1 \
  0,2 128 0

# 같은 프로토콜로 학습 범위 안의 delta=6 추가 평가
bash experiments/run_single_turn_asr_mxc.sh \
  /soundai/Model/VAPASR/hf-approved-qwen-v1-e10-n2/checkpoint-35000 \
  /soundai/users/tskim/VAPKT-data/results/single-turn-35000-d6-v1 \
  0,2 128 0 6
```

같은 명령을 다시 실행하면 저장된 `(발화 ID, delta)`는 건너뛴다. 모델 파일 메타데이터,
config hash, 코드 hash, TN 지문, 평가 manifest, 패딩·디코더 설정이 바뀌면
기존 결과와 섞이지 않도록 재개를 거부한다. 변경 실험은 새 결과 디렉터리를 쓴다.

`checkpoint-35000`에는 `<DELAY_8>` 토큰이 있지만 config의 학습 delays는
`[2,3,4,6]`이다. δ=8 실행은 이 학습 설정 범위 밖의 조건 평가이며,
토큰을 입력할 수 있다는 사실만으로 실제 방출 지연이 640 ms라고 보장하지 않는다.
δ=8 전량 평가는 사용자 요청으로 중지했으며, 중간 결과는 `single-turn-35000-d8-v1/`에
보존했다. 후속 평가는 학습 설정에 포함된 δ=6으로 진행한다.

## 산출물

- `manifest.jsonl`: 원본 경로·오디오 형식·실제 길이·원전사·채점 참조.
- `coverage.json`: 세트별 가용/누락 수·시간·누락 ID·TN 및 manifest 지문.
- `config-rank*.json`: 모델·코드·데이터·프로토콜 지문 및 실행 설정.
- `predictions-rank*.jsonl`: 발화별 원출력·정규화 문자열·S/D/I·토큰 ID·방출 청크.
- `parity-rank*.json`: 기존 단일 발화 디코더와의 검증 결과.
- `rank*.log`: 처리량·GPU peak memory·처리 개수.
- `done-rank*.json`: 해당 shard의 종료 증거.
- `summary.json`: 세트·δ별 corpus micro 평균. 모든 shard와 예상 행 수가 일치해야 `complete=true`.

외부 ASR 모델은 같은 manifest의 원본+1초 무음을 입력으로 받고 원출력을
`score_pair(reference, hypothesis, lang)`에 전달하면 같은 방식으로 채점할 수 있다.
비교 시 모델 이름뿐 아니라 이 입력·정규화·세트 지문도 함께 기록한다.
