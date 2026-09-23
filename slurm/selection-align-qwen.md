# 선별된 Qwen 원출력 forced alignment 제출

이 작업은 **재전사나 학습이 아니라**, 선별된 Qwen 원출력의 단어 시각을 추출하는 작업이다.
기본 입력은 `pair-v0.6-full-20260921/results/part-*/keep-asr.jsonl`이다.
약 177만 건을 대상으로 하며, 실제 작업량은 제출 시 입력 summary에서 검증한다.

## 자원과 실행

- 기본은 노드 1개이며 `--nodes=N`으로 확장한다. **각 노드의 GPU 8개를 모두 할당**하고
  GPU당 독립 task 1개를 둔다. DDP 학습은 아니다.
- 노드당 CPU 32개: task마다 오디오 decode thread 3개 + torch thread 1개.
- 기본 partition `apex`, 시간 제한 24시간. 실제 partition 제한에 맞게 제출 시 변경한다.
- GPU 메모리 상한 프로세스당 90%, 최대 배치 64건·합계 480초. 이는 사용량 목표가 아니라 상한이다.
- OOM/배치 오류는 배치를 반분해 재시도하고, 단일 샘플 실패는 `failed.jsonl`에 남긴다.
- Slurm의 task별 GPU 바인딩을 사용한다. 전체 노드의 물리 GPU 번호를 임의로 선택하지 않는다.

다음 파일들이 서버 프로젝트의 동일한 상대 경로에 있어야 한다.

```text
slurm/selection_align_qwen.sbatch
experiments/selection_align_qwen.py
vapasr/data/selection_alignment.py
```

기존 `vapasr/data/selection.py`, `selection_audio.py`, `archive.py`와 `vapasr` conda 환경도 필요하다.
**`sbatch`/`srun`이 제공되는 SLURM 로그인 노드**에서 아래 명령을 실행한다.
현재 접속한 `mxc` 셸에서는 `sbatch`가 PATH에 없었으므로, 그 셸에서 제출 가능하다고 가정하지 않는다.

```bash
cd /soundai/users/tskim/VAPKT
mkdir -p logs/train

# 먼저 GPU 8개 바인딩과 실제 정렬을 검증: 최대 8 shards × 16건.
# 결과는 기본 출력 경로에 -smoke를 붙인 별도 폴더에 저장.
sbatch --partition=apex --time=00:30:00 --export=ALL,SMOKE=1 slurm/selection_align_qwen.sbatch

# smoke의 summary와 실패 항목을 확인한 뒤 전체 실행 (자동 연쇄 제출하지 않음).
sbatch --partition=apex --export=ALL,SMOKE=0 slurm/selection_align_qwen.sbatch

# 기존 실행을 중지하고 8개 노드 × GPU 8개(총 64 GPU)로 같은 결과를 재개:
scancel <기존-job-id>
sbatch --partition=apex --nodes=8 --ntasks-per-node=8 --gres=gpu:8 \
  --export=ALL,SMOKE=0,RESUME=1 slurm/selection_align_qwen.sbatch
```

`hpc`를 쓰려면 `--partition=hpc`로 바꾼다. 이미 사용할 수 있는 로컬 스테이징 Python이 있으면
`--export=ALL,PY=/tmp/sa_tskim/vapasr-env/bin/python`로 지정할 수 있다.
기본은 공유 conda 환경이므로 초기 import가 느릴 수 있다. 환경 스테이징을 자동 수행하지 않는다.

배치를 키우려면 smoke에서 먼저 검증한 뒤 새 출력 경로와 함께 지정한다.

```bash
sbatch --partition=apex \
  --export=ALL,SMOKE=1,BATCH=128,BATCH_SEC=960,OUT=/soundai/users/tskim/VAPKT-data/data/selection/align-qwen-verbatim-b128 \
  slurm/selection_align_qwen.sbatch
```

## 텍스트와 타이밍 계약

1. 정렬 입력은 `recommended_training_target.text`이며 `transcripts.qwen_raw`와 완전히 같은지 확인한다.
   예전 `row.text`는 사용하지 않는다. **TN·소문자화·구두점 삭제를 저장 타깃에 적용하지 않는다.**
2. teacher 선별 때 사용한 오디오와 decoded waveform SHA-256이 일치해야 한다.
   다르면 해당 항목을 실패로 기록한다. 새로운 임의 crop/채널 선택은 하지 않는다.
3. Qwen 정렬기 자체는 구두점을 제외한 단어 단위로 시각을 반환한다. 반환한 단어를 원문 문자 위치로
   정확히 대응시키고, Qwen tokenizer의 원문 BPE 토큰에 **단어 종료 시각을 투영**한다.
   숫자를 읽기형으로 바꾸거나 대응에 실패한 단어를 적당한 위치에 붙이지 않는다.
4. 구두점·공백만 포함한 토큰은 `nonlexical_anchor`, `lexical_timing_mask=false`이다.
   앞 단어 종료 시각(문두에서는 첫 단어 시작 시각)을 표시용으로 붙이지만 **음향 근거로 취급하면 안 된다.**
   단어 내부 BPE 시각도 독립적인 토큰 정렬 측정치가 아니라 단어 시각의 투영이다.
5. 시간축은 **디코딩된 클립의 0초 기준**이다. VAD 조각 연결 등은 원본 세션 시각으로 단순 환산할 수 없다.
6. `training_eligible=false`, `turn_supervision=false`를 유지한다. 정렬 완료는 정확성 검수 통과가 아니며,
   EOT·화자 라벨이나 Phase 1/2 학습 시퀀스를 자동 생성하지 않는다. 학습기 연결은 후속 작업이다.

## 출력과 재개

기본 출력:
`/soundai/users/tskim/VAPKT-data/data/selection/align-qwen-verbatim-v1-20260921/`

```text
config.json                    입력·모델·tokenizer·코드·패키지 SHA, 실행 설정
results/part-00000/
    aligned.jsonl              원출력, 단어/토큰 시각, provenance
    failed.jsonl               실패 원인, 확보된 원출력·정렬기 결과
    summary.json               완료 여부, 성공/실패 건수, 결과 파일 SHA
work/                          중단된 미완료 shard의 임시 결과 (자동 삭제 없음)
locks/                         중복 작업의 동일 shard 동시 쓰기 방지
summary.json                   전체 집계; smoke와 full을 명시적으로 구분
```

- 입력/모델/코드/배치 설정 지문이 같으면 같은 제출 명령으로 재개한다.
- 완료 shard는 입력·출력 SHA를 검증하고 건너뛴다. 중간에 끊긴 shard만 처음부터 다시 처리한다.
- `--requeue`는 클러스터의 선점 정책에 따르며, 모든 실패를 자동 재제출하는 루프는 없다.
- **설정·코드·모델 변경이나 실패 샘플 재시도는 새 `OUT`**으로 실행한다.
- worker 자체 오류는 job에 실패로 전파된다. 샘플별 오류는 계속 처리하며 완료 summary를
  `complete_with_failures`로 표시한다. 따라서 **SLURM COMPLETED ≠ 모든 항목 정렬 성공**이다.
- 전체 결과 집계가 중단됐다면 CPU로 `summarize --out <경로>`를 다시 실행할 수 있다.
  집계 시 출력 SHA를 재검증하므로 대규모 결과에서는 I/O 시간이 든다.

로그는 `logs/train/SA_SFT_FullDuplexStage5-qwen-align-<jobid>.out/.err`에 남는다.
`PROGRESS`, `SHARD_DONE`, `ALIGN_RETRY`와 마지막 `summary`를 확인한다.
작업 시간 추정은 smoke 처리량 확인 후 산정한다. 이 문서의 24시간은 예상 소요 시간이 아니라 제한값이다.
