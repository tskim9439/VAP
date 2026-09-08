---
type: source
status: active
created: 2026-09-08
updated: 2026-09-08
summary: HF PreTrainedModel 래퍼(vapasr.hf)와 Trainer 서브클래스로 학습 코드를 전환하고 패리티·Liger 벤치·2-GPU 스모크(학습·평가·저장·재개·선점)를 검증한 기록
raw: raw/sources/experiments/2026-09-08-hf-trainer-migration/
related:
  - [[output-huggingface-training-framework-choice]]
  - [[source-stage1-mono-run-ab]]
  - [[decision-asr-tn-v1-freeze]]
---

# HF Trainer 전환 검증 (2026-09-08)

## 무엇을 바꿨나
- **모델**: `vapasr/hf/modeling_vapasr.py` `VapAsrForStreamingASR(PreTrainedModel)` + `VapAsrConfig`. encoder(Nemotron [56,0], `attach_encoder`) → adapter → Qwen3-ASR thinker(오디오 타워 제거, lm_head tied). forward 는 기존과 같은 라벨 위치 전용 CE·`<NEXT_AUDIO>` 가중, `stream_decode` 동일. `from_qwen`(새 모델) · `from_legacy`(ckpt-last.pt 변환) · `save_pretrained/from_pretrained`(동결 인코더는 저장 제외, .nemo 에서 재구성).
- **Liger**: `vapasr/hf/liger.py` — RMSNorm·SwiGLU·RoPE 인스턴스 패치. fused linear CE 는 미사용(라벨 위치 전용 lm_head 가 이미 메모리를 줄임).
- **학습**: `vapasr/hf/trainer.py` `VapAsrTrainer(Trainer)` — 라운드로빈 언어별 버킷 배치(`vapasr/hf/data.py`, rank 분할을 직접 하므로 accelerate 가 다시 나누지 않음), 언어별 next_weight, 옵티마이저 그룹(adapter lr↑·임베딩 wd 0·encoder lr), **전 rank 분산 스트리밍 평가**(gloo all_gather), `PreemptCallback`(PREEMPT 파일·SIGUSR1/SIGTERM → gloo 합의 → 저장 후 종료). 체크포인트·재개는 Trainer 표준(`checkpoint-<step>/` = model.safetensors 2.4 GB + optimizer 4.8 GB + 스케줄러·RNG·trainer_state).
- **엔트리**: `experiments/s3_train_hf.py`, `slurm/s3_train_hf.sbatch`(산출물 `/soundai/Model/VAPASR/hf-<RUN>/`, DONE 은 스크립트가 씀, `INIT` 로 Qwen 디렉토리|HF 디렉토리|ckpt-last.pt 선택).
- DeepSpeed/FSDP 는 넣지 않았다(0.6B full FT 는 rank 당 21–55 GB 로 DDP 충분). 인코더 해동·1.7B 급에서 검토.

## 검증 결과
| 항목 | 결과 |
|---|---|
| 패리티(s1-C step-500) | 파라미터 Δ0 · 손실 0.523754 동일 · 저장/재로드 Δ0 · 디코드 토큰열 동일 |
| Liger | 969 → 734 ms/step (+24 %), peak 21.3 → 21.2 GB, 손실 Δ 6.8e-4(bf16) |
| 스모크 2 GPU 60 step | loss 11.9 → 2.6, 20 step 저장(6.8 GB, ~1 분), 30 step 분산 평가 130 s |
| 재개 60 → 90 | checkpoint-60 자동 감지, 손실 연속(2.61 → 2.44), final/·DONE 생성 |
| 선점 | 재개 중 PREEMPT 파일 → 다음 step 경계(107)에서 저장 후 종료 코드 0, checkpoint-107 생성 |

## 추가 (2026-09-08 16:20) — 다중 노드 평가 행의 원인 확정
SIGUSR2 스택 덤프(job 66260, hpc-101 의 8 rank)로 확인: 모든 rank 가 `_eval_set → stream_decode → encode → NemotronOnline.forward → nemo …/conformer_encoder.py update_max_seq_length` 에서 대기.
NeMo ConformerEncoder 는 forward 마다 `sync_max_audio_length=True` 이면 기본 NCCL 그룹에 `all_reduce(MAX)` 를 날린다. 학습은 모든 rank 가 step 마다 인코더를 한 번씩 불러 맞지만,
평가는 rank 별 스트림 수가 달라 collective 호출 수가 어긋나 교착한다(GPU 100 % 스핀). 단일 노드 스모크는 표본을 rank 수로 나눠떨어지게 준 덕에 통과했다.
같은 원인이 rank 0 단독 평가(66066)·NCCL barrier 시절의 IB 오류·조용한 종료(65963·65965·66007)까지 설명한다. 디코드 폭주·gather 장치 가설은 틀렸다(기록 보존).
수정: `vapasr/features/online.py` 에서 `enc.sync_max_audio_length = False`(커밋 48d6790). 회귀 검사: 2 rank 불균등 표본(3/3/5) 분산 평가.
부수 수정: 재개 시 `trainer_state.json` 의 eval_steps 가 복원되므로 평가를 끄려면 `eval_strategy="no"`(6865dbd); 오프라인 평가 루프 `experiments/s3_eval_loop.sh`.

## 남은 것
- 8-노드 SLURM 에서 `slurm/s3_train_hf.sbatch` 실전 검증(66103 C2 가 끝난 뒤 또는 별도 노드).
- `--select`/`--final` 평가 경로는 아직 s1_train_mono.py 에 남아 있다(HF 모델을 `from_pretrained` 로 읽는 평가 스크립트로 이관 예정).
- 로그의 learning_rate 는 첫 파라미터 그룹(adapter) 값이다. thinker lr 를 함께 찍도록 개선 여지.
- Liger 사용 시 학습 손실이 bf16 수준으로 달라지므로 비교 실험은 같은 설정끼리.
