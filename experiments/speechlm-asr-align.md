# 새 SpeechLM Arrow: 원출력 ASR + forced alignment 준비

## 대상과 원칙

- 원본: `/soundai/users/tskim/VAPKT-data/data/speechlm-asr-single-speaker-v1-20260922`
- 26개 catalog, 47,662,558행, 약 108,513시간. 이는 품질 승인셋이 아니다.
- 원 Arrow·tar·원전사는 수정하지 않는다. Qwen·Whisper의 대소문자, 숫자, 구두점도 그대로 저장한다.
- TN은 합의 점수 계산에만 적용한다. Qwen3-ASR-0.6B와 Whisper-large-v2를 사용한다.
- 영어 단어 / 한국어 공백 제외 문자 단위로 `edit_distance(Q, W) / max(len(Q), len(W)) ≤ 0.05`를 적용한다.
- 공백/빈 출력, 추론 실패, 생성 길이 경고, 오디오 QC 보류는 합의율과 별개로 보류한다.
- 원전사와의 차이는 진단값이다. 원전사 일치를 새로운 필수 관문으로 추가하지 않는다.
- 통과한 **Qwen 원출력**을 Qwen3-ForcedAligner-0.6B로 정렬한다. 문자 offset으로 원문 BPE에 시각을 투영하며, 구두점은 음향 시각이 아닌 비어휘 anchor로 표시한다.

## tar 사전 인덱싱: CPU 전용

Arrow의 `audio_tar` 열에서 참조되는 tar만 중복 제거하여 수집한다. 오디오 디렉터리 전체 glob은 사용하지 않는다.
각 비압축 tar의 member header에서 `(offset_data, size)`를 추출한다. 원음 추출·재저장은 하지 않는다.

- 인덱스: `.../VAPKT-data/data/speechlm-asr-tar-index-v1/`
- 감사 결과: `.../VAPKT-data/data/speechlm-asr-index-audit-v1/`
- 인덱스 키: tar 절대경로·파일 크기·mtime_ns의 digest. 오디오 콘텐츠 해시의 대체물이 아니다.
- CPU worker 16개, 최대 설정 48개. 공유 스토리지 병목이 생기면 worker를 낮춘다.
- 임시 파일 후 atomic rename, tar별 flock. 다른 worker가 동일 tar 인덱스를 동시에 쓰지 않는다.
- 학습/추론 reader는 member offset으로 필요한 바이트만 읽는다. 프로세스별 메모리 인덱스 캐시는 최대 tar 32개다.
- 기존 인덱스는 식별자가 같으면 재사용한다. 크기/mtime 변경 시 새 인덱스를 만든다.
- 인덱스 완료는 오디오 디코딩 성공이나 모든 manifest member의 존재를 인증하지 않는다. 실제 읽기에서 별도로 검증한다.

```bash
cd /soundai/users/tskim/VAPKT
bash experiments/prepare_speechlm_asr_align.sh index
```

## CPU 입력 준비 → GPU 스모크

```bash
bash experiments/prepare_speechlm_asr_align.sh smoke
bash slurm/submit-speechlm-asr-align.sh smoke 1
```

스모크는 DB당 32행, 총 832행이다. Arrow shard 여러 곳의 앞부분을 분산해서 고르므로 경로·스키마·모델 호환성 확인용이며, 품질 통과율의 비편향 표본은 아니다.
CPU check는 shard당 2행을 읽는다. 제출기는 인덱스 완료 및 CPU check 통과를 확인한다.

GPU는 **hpc partition에서 Slurm 노드당 8개 전체**, task당 1개를 할당한다. 제출 쉘과 sbatch 파일 모두 hpc를 기본값으로 사용한다(2026-09-23 사용자 요청). worker 하나에 Qwen·Whisper·aligner를 상주시켜 모델 재적재와 오디오 재읽기를 줄인다. ASR 두 모델은 같은 디코딩 waveform을 순서대로 사용하며, 합의를 통과한 행만 aligner에 보낸다.

- task당 CPU 8개: I/O thread 7개 + torch thread 1개.
- 배치 최대 64행 및 오디오 합 480초. 언어를 분리하고 길이순으로 구성한다.
- 현재·다음 배치만 오디오를 미리 읽는다. 전체 데이터 waveform을 메모리에 올리지 않는다.
- CUDA 메모리 상한 90%. OOM/배치 오류는 프레임을 해제한 뒤 이분할 재시도하며, 단일 행 실패는 명시적으로 저장한다.
- 8 kHz 원음은 추론 시 16 kHz로 resampling하며 원본 sampling rate를 보존한다.
- **평가용 1초 무음 패딩은 적용하지 않는다.** 여기는 offline 교사 라벨 생성/원음 기준 정렬이다.
- 30초 초과·train 외 split·mono 미확인·지원 밖 언어는 삭제 없이 보류한다. 긴 오디오 분할은 별도 규약이 필요하다.

## 전량 준비와 재개

GPU 스모크 후 전사 원출력, pair 점수, 정렬 실패 원인, peak GPU memory, DB별 처리량을 확인하고 실행한다.

```bash
bash experiments/prepare_speechlm_asr_align.sh full
# 아래는 2노드 예시이며 자동으로 제출하지 않는다.
bash slurm/submit-speechlm-asr-align.sh full 2
```

전체 준비는 CPU에서 별도 JSONL shard를 만들므로 추가 디스크와 시간이 필요하다. 기본 shard는 4,096행이다. 완료한 shard는 내용 해시 검증 후 건너뛰며, 중단된 shard만 다시 계산한다. worker 수를 바꿔도 완료 결과를 재사용한다. `prepare`를 기존 출력 루트에 다시 실행하면 거부하고, GPU 재개는 제출 명령만 다시 실행한다.

config에는 입력 Arrow·JSONL 해시, 모델 파일 해시/메타데이터, 코드·패키지·TN 지문을 기록한다. worker는 코드·패키지 및 모델 크기/mtime를 검사한다. 모든 rank가 수십 GB 가중치를 재해싱하지는 않으므로, 실행 중 모델 경로는 불변으로 유지한다. 관련 Python 코드 변경 시 이전 지문으로 재개하지 않고 별도 출력 루트를 사용한다.

## 결과와 승인 경계

각 `results/part-*/`:

- `records.jsonl`: 원본 행, raw Qwen/Whisper 전사, 합의 점수, 오디오 hash/측정값, 보류 사유.
- `aligned.jsonl`: Qwen 원문, 단어 시각, BPE ID/원문 offset/종료 시각, 비어휘 mask.
- `failed.jsonl`: 합의는 통과했지만 정렬에 실패한 행.
- `summary.json`: 상태별 건수, 정렬 성공/실패, 실행시간, 메모리 peak, 출력 해시.

입력 행은 반드시 기록 하나로 회계되고, 선택 수 = 정렬 성공 + 정렬 실패로 검증한다. 전체 summary는 완료 여부와 DB별 건수를 제공한다. 기술적 완료와 품질 통과는 다르다.

**전량 ASR 실행 전 기존 train/held-out과 locator 중복 제거를 권장한다. 현재 준비 코드에는 전역 중복 제거가 구현되어 있지 않다.** 디코딩 waveform hash는 수집하지만, 서로 다른 8/16 kHz 사본이나 근접 중복까지 해결하지 않는다. 본 학습 전에는 별도 중복/평가 누출 감사, alignment QC, DB별 샘플링 비중과 8 kHz 비율 결정을 거쳐야 한다. 자동 `training_eligible=True` 승격은 없다.

ETA는 GPU 스모크의 DB별 audio-hours/GPU-hour를 실측한 뒤 계산한다. 108,513시간 전체에 이전 소규모 처리속도를 그대로 적용하지 않는다.
