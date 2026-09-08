# HF Trainer 전환 검증 기록 (2026-09-08, mxc 컨테이너 GPU 2·4·5)

코드: `vapasr/hf/` (configuration_vapasr · modeling_vapasr · liger · data · trainer), `experiments/s3_train_hf.py`, `slurm/s3_train_hf.sbatch`, `experiments/hf_parity.py`. 커밋 4e8dbd4 이후.
환경: transformers 4.57.6 · accelerate 1.12.0 · peft 0.20.0 · torch 2.9.0 · liger-kernel 0.8.2(신규 설치) · qwen-asr 0.0.6 · nemo 3.1.0 — conda `/soundai/users/tskim/VAPKT-data/conda/envs/vapasr`.

## 1. 패리티 (experiments/hf_parity.py, GPU 2) — 기존 MonoInterleavedASR vs VapAsrForStreamingASR, s1-C/ckpt-last.pt(step 500)
| 검사 | 결과 |
|---|---|
| 파라미터 | thinker·adapter max\|Δ\| = 0 |
| 같은 배치 손실 (B=12, L=1474, dev-clean 12 스트림) | legacy 0.523754 / hf 0.523754 (next 0.1427, text 1.0214, top1 0.6948 동일) |
| save_pretrained → from_pretrained | model.safetensors 2.4 GB(동결 인코더 제외), 34 s 저장 / 10 s 로드, 손실 Δ 0, lm_head tied 유지, tokenizer 특수 토큰 id 일치 |
| stream_decode (dev-clean 스트림 1) | 토큰열 동일(63/63), forced 0, tick p50 62.8 vs 62.3 ms |
| Liger(RMSNorm 113·SwiGLU 28·RoPE) | 손실 0.523070 (Δ 6.8e-4), 디코드 63 토큰 중 2 개 상이(bf16 반올림) |
| 처리량 벤치 (forward+backward, grad ckpt, GPU 1) | Liger 없음 969 ms/step · peak 21.3 GB → Liger 734 ms/step · 21.2 GB (**+24 %**) |

## 2. 스모크 학습 (torchrun 2 GPU, --train librispeech-dev,kspon-full, bs 4/8, 새 모델 = Qwen3-ASR + 증류 adapter)
- 60 step: loss 11.9 → 2.6, checkpoint-20/40/60(각 6.8 GB = model 2.4 + optimizer 4.8, ~1 분), step 30/60 분산 평가 130 s(4+4 스트림 + 8 발화, 2 rank 분할, gloo gather). 초기 모델이라 err 1.0.
- 재개(--max-steps 90): checkpoint-60 자동 감지 → step 65 loss 2.44 로 연속, 90 step 완주 → eval-90(err 0.996, 토큰 방출 시작), final/(HF 모델+tokenizer), results.json, DONE, checkpoint 2 개 유지(80·90).
- 선점(--max-steps 130, checkpoint-90 재개 중 12:44 PREEMPT 파일 touch): 다음 step 경계에서 `선점/종료 신호 → step 107 저장 후 종료`, checkpoint-107 저장, 종료 코드 0, DONE 미기록(재시작 시 107 부터).

## 3. 알게 된 것
- qwen_asr 의 thinker 는 transformers 내장이 아니라 패키지 자체 클래스(`Qwen3ASRThinkerForConditionalGeneration`, 디코더 층은 `Qwen3ASRText*`) → Liger 는 인스턴스 패치로 적용.
- 컨테이너 기본 python(/usr/local, transformers 5.4) 로는 qwen_asr 가 import 되지 않는다(`check_model_inputs()` 시그니처). `scripts/activate-env.sh` 가 mxc 프로필에서 MXC_CONDA_DIR 를 활성화하도록 수정.
- Trainer 의 `eval_dataset` 에 dict 를 주면 셋마다 evaluate 를 호출하므로 자리표시자 하나만 넘기고 dev 셋은 트레이너 속성으로 보관.
- Trainer 기본 콜백(PrinterCallback 등)이 뒤에 붙으므로 선점 콜백은 객체 참조로 들고 있어야 한다.
