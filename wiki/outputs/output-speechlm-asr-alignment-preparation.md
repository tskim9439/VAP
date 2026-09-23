---
type: output
status: active
created: 2026-09-22
updated: 2026-09-22
summary: 새 SpeechLM Arrow의 tar 4,790개를 CPU로 사전 인덱싱하고 원출력 ASR·5% 합의·정렬 GPU 스모크를 준비한다.
sources:
  - [[source-speechlm-v1-encoder-alignment-2607]]
  - [[output-speechlm-single-speaker-manifest]]
---

# 새 Arrow의 ASR·forced alignment 준비

## 결정한 실행 흐름

```text
새 Arrow (26 DB, 47,662,558행)
  ├─ CPU: 참조 tar 중복 제거 → 4,790개 offset 인덱스
  └─ CPU: 원본 보존한 bounded 입력 shard
        → Qwen3-ASR-0.6B 원출력
        → Whisper-large-v2 원출력
        → 비교할 때만 TN, Qwen–Whisper 거리 ≤5%
        → 통과한 Qwen 원문을 ForcedAligner로 정렬
        → 원문 BPE 시각 투영 → 별도 timing QC·중복 감사
```

기존 raw 저장 원칙을 유지한다. 원전사, Qwen 출력, Whisper 출력은 덮어쓰지 않는다. 원전사 일치는 필수 관문이 아니며 진단용이다. 오디오 오류·지원 밖 split·길이·채널, 빈 출력, 생성 한계 경고는 합의 점수와 별도로 보류한다.

현재 버전은 30초 이하 English/Korean train clip을 대상으로 한다. 긴 음성 분할과 전역 PCM/근접 중복 제거는 아직 구현되어 있지 않다. 모든 결과의 `training_eligible`은 false이며 정렬 완료를 본 학습 승인으로 해석하지 않는다.

## tar 인덱스

전체 새 Arrow를 실제 스캔해 47,662,558행과 참조 tar 4,790개를 확인했다. 오디오 디렉터리 glob으로 임의 tar를 추가하지 않는다.

- 저장: `/soundai/users/tskim/VAPKT-data/data/speechlm-asr-tar-index-v1`
- 감사: `/soundai/users/tskim/VAPKT-data/data/speechlm-asr-index-audit-v1`
- 로그: `/soundai/users/tskim/VAPKT/logs/speechlm-asr-index-v1.log`
- mxc CPU worker 16개로 시작했으며 GPU는 사용하지 않는다.
- 원 tar의 member offset/size만 추출한다. 공용 원본 옆에 sidecar를 만들지 않는다.
- 파일 경로·크기·mtime를 식별자로 쓰며 변경 시 새 인덱스 생성, 파일별 잠금과 atomic rename을 적용한다.
- worker는 최대 tar 32개의 인덱스를 메모리에 유지하고 필요한 byte range만 읽는다.

이는 콘텐츠 중복 제거 또는 모든 오디오의 디코딩 검증이 아니다. waveform hash와 실제 sampling rate/channel 검사는 추론 시 별도로 수행한다.

## 실행 상태와 다음 관문

2026-09-22 구현·검증 시점:

- ASR·정렬 준비 코드, CPU 인덱서, 노드당 8 GPU Slurm 실행기 작성.
- 로컬 Python 3.11 관련 테스트 26개 통과(모의 모델 worker의 원문 정렬·행 회계·재개 포함). 서버 새 tar/정책·모의 worker 테스트 8개 통과.
- 전체 tar 인덱싱 실행 중이며 관측 시점에 200개 성공·오류 0개 로그를 확인했다. 완료 여부는 `index-summary.json`의 `complete`와 `ready`로 판정한다.
- DB당 32행, 총 832행의 스모크 입력과 `config.json` 저장 완료. 실제 오디오 CPU 검사 실행 중이며 `cpu-check.json` 완료를 확인해야 한다.
- GPU 추론은 아직 미실행이며 전체 데이터 작업을 자동 제출하지 않았다.

GPU task당 세 모델을 상주시켜 반복 로딩을 줄이고 동일 waveform을 재사용한다. I/O thread 7개, batch 최대 64행/480초, OOM 이분할 재시도와 원자적 shard 재개를 사용한다. 환경 변수는 실행 쉘과 Slurm job 내부로 제한한다.

인덱스와 CPU 검증 완료 후 1노드 스모크:

```bash
cd /soundai/users/tskim/VAPKT
bash slurm/submit-speechlm-asr-align.sh smoke 1
```

GPU 스모크에서 DB별 성공률·정렬 실패 원인·peak memory·처리량을 측정한 뒤 전량 범위를 결정한다. 특히 기존 train/held-out 중복 제거가 끝나지 않았으므로 전량 실행 전 locator 감사를 권장하며 본 학습 전에는 필수다. MLS 비중 및 8 kHz 혼합 비율도 별도로 정해야 한다.

세부 명령·저장 스키마·재개 방법은 [실행 문서](../../experiments/speechlm-asr-align.md), 구현은 [파이프라인](../../experiments/speechlm_asr_align.py)과 [오디오 로더](../../vapasr/data/speechlm_inference.py)를 따른다.
