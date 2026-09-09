---
type: output
status: active
created: 2026-09-08
updated: 2026-09-08
summary: VAPASR는 HF-native 모델을 코어로 삼고 Trainer·Accelerate를 1차 runner로, DeepSpeed·Liger와 MS-SWIFT는 검증 후 선택 적용한다
sources:
  - [[source-hf-trainer-migration]]
  - [[output-stage1-mono-pilot]]
  - [[output-interleaved-streaming-slm-architecture]]
  - [[source-stage1-mono-run-ab]]
  - [[decision-asr-backbone]]
  - [[decision-compute-environment]]
---

# VAPASR Hugging Face 전환과 학습 프레임워크 선택

## 결론

**1번을 수정한 형태가 유리하다.** 모델 코어를 Hugging Face native로 만들고,
학습 runner는 먼저 `Trainer` 또는 `Accelerate` 기반으로 옮긴다. DeepSpeed와 Liger는
기본 전제가 아니라 **각각 켜고 끌 수 있는 최적화 옵션**으로 둔다.

```text
HF-native VAPASR model/config/processor
          │
          ├─ 현재 custom loop (회귀 기준, 전환기에 유지)
          ├─ HF Trainer/Accelerate runner (주 경로)
          │      ├─ DDP/FSDP2
          │      ├─ DeepSpeed ZeRO-2 (벤치마크 후)
          │      └─ Liger 일부 kernel (동등성 확인 후)
          └─ MS-SWIFT plugin (추후 편의 계층)

NeMo encoder backend는 유지
NeMo/Megatron 전체 training stack 전환은 현재 보류
```

핵심은 **Hugging Face 친화성은 Trainer 사용 여부가 아니라 모델 계약**이라는 점이다.
`PreTrainedConfig`, `PreTrainedModel`, `ProcessorMixin`, `save_pretrained()` 및
`from_pretrained()`를 먼저 구현하면 runner는 나중에 교체할 수 있다. Hugging Face도
커스텀 config/model 상속을 통해 저장·로드와 AutoClass를 제공한다
([공식 문서](https://huggingface.co/docs/transformers/en/custom_models)).

질문의 `DeepSpeech`는 분산 학습 엔진 **DeepSpeed**를 뜻하는 것으로 해석했다.
Mozilla DeepSpeech를 뜻했다면 이는 이 선택지의 학습 인프라가 아니라 별도 ASR 모델
계열이므로 후보에서 제외한다.

## 현재 코드가 일반 SFT와 다른 지점

현재 `experiments/s1_train_mono.py`, `vapasr/uslm/mono_model.py`,
`vapasr/uslm/mono_data.py`에는 다음 의미론이 들어 있다.

- EN/KO별 배치 크기와 길이 bucket, 두 corpus의 교대 공급
- 언어별 `<NEXT_AUDIO>` loss weight와 텍스트 위치 통계
- `(B,L,152k)` 전체 logits를 만들지 않는 **label 위치 전용 LM head**
- 연속 Nemotron feature를 토큰 embedding 위치에 삽입하는 `inputs_embeds` 경로
- audio clock에 맞춘 custom KV-cache streaming decode와 flush round
- WER/CER 외 `viol80`, 방출률, 지연 분포, backlog, tick latency를 재는 sentinel
- 느린 rank-0 평가 동안 NCCL timeout을 피하기 위한 별도 Gloo 동기화
- SLURM `USR1` 선점 감지, 안전 저장, requeue 및 NFS checkpoint 복구
- adapter·thinker·embedding·encoder별 optimizer parameter group

따라서 stock `Trainer`, stock MS-SWIFT SFT, stock NeMo recipe 중 어느 것도 그대로
대체하지 못한다. 프레임워크 선택 기준은 이 규약을 얼마나 적은 비공개 override로
보존할 수 있는가다.

## 후보 비교

| 기준 | HF native + Trainer/Accelerate | MS-SWIFT | NeMo/Megatron Bridge |
|---|---|---|---|
| HF Hub·`save_pretrained` 호환 | **가장 직접적** | 높음 | HF 변환 bridge 제공 |
| 현재 custom batch/loss 이식 | 공식 `get_train_dataloader`, `compute_loss` 확장점으로 가능 | model·template·multimodal collator plugin을 별도 작성 | custom model/recipe·checkpoint bridge 작업 필요 |
| 0.6B thinker, 8–64 GPU 적합성 | **충분함** | 충분함 | 기능 대비 복잡도가 큼 |
| tensor/pipeline parallel 대규모 확장 | 보통 | Megatron-SWIFT 선택 가능 | **가장 강함** |
| 현재 NeMo encoder 재사용 | backend module로 유지 가능 | custom loader가 필요 | 자연스럽지만 전체 모델은 stock recipe가 아님 |
| 디버깅·수치 회귀 난이도 | **가장 낮음** | 중간 | 높음 |
| 현재 권장 역할 | **주 경로** | 추후 CLI/recipe 편의 계층 | 수십억~수백억 규모에서 재검토 |

Hugging Face는 `Trainer`의 데이터로더와 loss를 공식적으로 subclass할 수 있게 하고,
동작 시점만 바꾸는 경우 callback을 권장한다
([공식 문서](https://huggingface.co/docs/transformers/trainer_customize)). 반면
MS-SWIFT도 custom model을 쓰려면 model type과 loader를 등록하고, 멀티모달 모델은
template의 encode·collator를 재정의해야 한다
([공식 문서](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-model.md)).
즉 MS-SWIFT는 현재의 커스텀 구현을 없애기보다 다른 plugin API로 다시 쓰게 할
가능성이 크다.

NeMo Megatron Bridge는 HF↔Megatron checkpoint 변환, tensor/pipeline parallel과
PyTorch-native custom model을 지원한다
([공식 문서](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html)). 이는 모델이
커지면 매력적이지만, 현재 trainable thinker는 0.6B이고 frozen Nemotron encoder를
포함해도 H200 메모리에 들어간다. VAPASR는 Qwen3-ASR의 AuT를 그대로 학습하는 모델이
아니라 Nemotron feature injection, 특수 제어 토큰, 희소 weighted CE를 추가했으므로
공식 Qwen recipe가 곧바로 적용되지 않는다.

## DeepSpeed 판정

**처음에는 기존 DDP 또는 FSDP2를 기준으로 두고, DeepSpeed는 ZeRO-2만 비교한다.**
Hugging Face 문서도 ZeRO-2가 optimizer state와 gradient를 나누면서 ZeRO-3보다
통신 부담이 낮고, ZeRO-3는 ZeRO-2로 모델이 들어가지 않을 때 쓰라고 권한다
([공식 문서](https://huggingface.co/docs/transformers/deepspeed)).

현재 0.6B full-FT thinker는 H200에서 메모리 부족이 일차 병목이 아니다. 특히 64 GPU
환경에서는 ZeRO 통신과 checkpoint shard 복구가 오히려 복잡도를 늘릴 수 있다.
다음 세 조건을 동일 batch와 동일 200–500 step으로 비교해 optimizer step/s,
audio-hour/s, peak HBM, 통신 시간, 저장·재개 시간을 보고 선택한다.

1. DDP 기준선
2. FSDP2 또는 DeepSpeed ZeRO-2
3. ZeRO-2 + Liger 일부 kernel

ZeRO-3와 CPU/NVMe offload는 encoder를 열거나 thinker를 수십억 규모로 키워 ZeRO-2로
들어가지 않을 때까지 보류한다.

## Liger 판정

Liger는 **사용 가치가 있지만 drop-in 전체 적용은 금지**한다. 공식 프로젝트는 Qwen3의
RoPE, RMSNorm, SwiGLU, CE와 fused linear CE를 지원하고 Trainer, DeepSpeed, SWIFT와의
호환을 표방한다
([공식 저장소](https://github.com/linkedin/Liger-Kernel)). 다만 공식 처리량·메모리 수치는
Llama 3 8B와 긴 context 조건의 결과이므로 VAPASR에 그대로 기대할 수 없다.

VAPASR의 `lm_head`는 거대한 전체 logits를 피하기 위해 **라벨 위치에만** 적용되고,
`<NEXT_AUDIO>`와 텍스트에 서로 다른 가중치를 준다. 그러므로 Liger fused linear CE를
바로 켜면 현재 loss 의미론 또는 메모리 절약 경로가 달라질 수 있다.

적용 순서는 다음과 같다.

1. RMSNorm·RoPE·SwiGLU만 적용해 forward/backward 및 학습 곡선 동등성을 확인한다.
2. CE는 기존 희소 weighted CE를 유지한다.
3. 별도 custom kernel로 같은 가중 semantics를 정확히 재현할 때만 fused linear CE를
   비교한다.
4. 실제 VAPASR sequence length와 EN/KO batch에서 10% 이상의 안정적 개선이 없으면
   의존성을 추가하지 않는다.

## 권장 Hugging Face 모델 계약

### `VAPASRConfig(PreTrainedConfig)`

다음을 checkpoint와 함께 저장한다.

- Qwen thinker 및 Nemotron encoder의 model ID, revision, layer/context `[56,0]`
- audio sample rate, 12.5 Hz clock, feature dimension과 adapter 구조
- `<NEXT_AUDIO>`, `<EMPTY_AUDIO>`, `<DELAY_n>` ID와 최대 방출·flush 규약
- lexical TN version과 fingerprint, tokenizer revision/hash
- 기본 delay·stream decode 설정
- 코드 commit과 모델 format version

`next_weight`, learning rate처럼 학습 run에 속한 값은 config의 모델 의미론과 분리해
`TrainingArguments` 또는 run manifest에도 남긴다.

### `VAPASRForConditionalGeneration(PreTrainedModel)`

- 표준 이름의 `input_ids`, `attention_mask`, `labels`와 함께 `audio_values` 또는
  `audio_features`, `audio_positions`, `audio_lengths`를 받는다.
- `ModelOutput`에 `loss`, label 위치 logits 또는 선택적 logits, `loss_text`,
  `loss_next`, `top1_text`, `past_key_values`를 반환한다.
- encoder backend와 cached-feature 경로를 분리해 NeMo가 설치되지 않은 환경에서도
  cached feature 학습·thinker 로드가 가능하게 한다.
- 일반 `generate()`에 억지로 맞추기보다 audio clock과 flush를 보존하는
  `stream_generate()`를 공개 API로 먼저 유지한다.

### Processor와 collator

- `VAPASRProcessor`가 tokenizer, audio resampling, TN version 검사를 묶는다.
- `VAPASRDataCollator`가 audio/sequence padding과 `audio_positions`를 만든다.
- EN/KO 교대와 길이 bucket은 `VAPASRTrainer.get_train_dataloader()` 또는
  Accelerate loop의 공개 sampler로 유지한다.

## Trainer와 Accelerate의 역할

최종 사용자 편의에는 Trainer가 좋지만, **전환 첫 runner는 Accelerate도 동등 후보**다.
현재처럼 corpus 교대, 분산 streaming decode, SLURM 선점 저장을 세밀하게 제어하려면
Accelerate가 기존 loop를 보존하기 쉽다. HF 친화성을 Trainer와 동일시할 필요가 없다.

권장 구조는 다음과 같다.

- `train_vapasr.py`: Trainer 기반 표준 실행·실험 관리
- `VAPASRTrainer`: 공개 메서드인 dataloader, loss, optimizer group만 override
- `evaluate_streaming.py`: 느린 sentinel/final 평가는 우선 독립 evaluator로 유지
- `train_vapasr_accelerate.py` 또는 현 loop: 회귀 기준 및 복잡한 분산 평가 fallback

Trainer의 private checkpoint/evaluate 메서드를 깊게 override하면 버전 변경에 취약하므로
피한다. sentinel은 checkpoint 저장 뒤 별도 job으로 실행하거나, 모든 rank가 참여하는
독립 evaluator로 분리하는 편이 안전하다.

## 단계별 전환 계획

### 0. 현재 1,930 h 학습 보호

진행 중인 30-epoch run의 코드·환경·checkpoint 형식을 바꾸지 않는다. 현재 branch에서
완주시키고, 새 HF 경로는 별도 entry point로 만든다. 현 custom loop는 최소 한 번의
HF 학습 곡선 동등성 확인 전까지 삭제하지 않는다.

### 1. 모델만 HF-native로 전환

config/model/output/processor를 추가하고 기존 `MonoInterleavedASR` 가중치를
`save_pretrained(..., safe_serialization=True)` 형식으로 export/import한다. 이 단계는
학습 루프를 바꾸지 않는다.

### 2. 회귀 시험

고정 EN/KO batch에 대해 다음 관문을 통과한다.

- sequence, label, audio position이 기존 코드와 완전히 동일
- 선택 label logits, `loss_text`, `loss_next`, weighted total loss가 허용 오차 내 동일
- 한 optimizer step의 parameter delta가 허용 오차 내 동일
- save/load round trip 뒤 greedy token과 방출 chunk가 동일
- 중단·재개 run이 batch 순서, LR, optimizer, RNG와 sampler epoch를 복원

마지막 항목은 현 코드의 `step`만 복원하는 방식보다 강화해야 한다. 1,930 h 학습에서
재개 후 정확한 data position을 보존하는 것은 프레임워크 전환의 실질적 이득이다.

### 3. Trainer/Accelerate runner

custom sampler, optimizer parameter group, weighted sparse loss를 이식하고 300-step
overfit과 Stage 1 고정 sentinel을 재현한다. WER/CER뿐 아니라 `viol80`, tok/chunk,
지연 분포까지 같아야 한다.

### 4. 분산 backend 비교

8 GPU와 실제 64 GPU 환경에서 DDP, FSDP2, DeepSpeed ZeRO-2를 비교한다. 가장 복잡한
backend가 아니라 **audio-hour당 비용과 장애 후 복구 시간을 포함해 가장 나은 것**을
채택한다.

### 5. Liger와 MS-SWIFT

Liger를 kernel별로 켜며 동등성·성능을 잰다. HF checkpoint와 processor가 안정된 뒤
MS-SWIFT external plugin을 얇게 추가해 CLI recipe, LoRA sweep, 배포 도구를 활용한다.
MS-SWIFT가 core model이나 정본 checkpoint 형식을 소유하게 하지는 않는다.

## 재검토 관문

다음 중 하나가 생기면 NeMo/Megatron Bridge 또는 Megatron-SWIFT를 다시 검토한다.

- thinker가 수십억 파라미터 이상으로 커져 tensor/pipeline parallel이 필요하다.
- Nemotron encoder 전체를 열고 end-to-end 학습하여 DDP/FSDP2 처리량이 병목이 된다.
- sequence length나 model size 때문에 ZeRO-3도 충분하지 않다.
- 다중 노드에서 측정한 Megatron recipe의 처리량 이득이 변환·운영 비용을 명확히
  상쇄한다.

현재 규모에서는 그 조건이 아니다. 따라서 **HF-native core → Trainer/Accelerate →
필요한 최적화만 선택 적용**이 정확도 실험을 방해하지 않으면서 생태계 호환성과
장기 유지보수성을 얻는 가장 안전한 경로다.

## 구현 검토 (2026-09-08)

[[source-hf-trainer-migration]]에 기록된 `vapasr.hf` 구현은 주 경로 선택을 실제
코드로 옮겼다. legacy `s1-C step-500`과 파라미터·loss·저장 왕복·stream decode가
동일했고, Liger는 weighted sparse CE를 건드리지 않고 RMSNorm·SwiGLU·RoPE만
적용했다. 2-GPU에서 학습·분산 평가·저장·재개·선점 종료도 실행됐으므로
**HF 전환의 기능적 feasibility는 통과**로 본다.

다만 8-node 주 학습 전에 다음을 보완해야 한다.

### P0 — SLURM 기본 adapter 인자 누락

`slurm/s3_train_hf.sbatch`는 `COMMON`을 확장한 뒤에 `INIT_ADAPTER` 기본값을
설정한다. 사용자가 환경변수 `INIT_ADAPTER`를 명시하지 않으면 기본 증류 adapter
경로가 명령행에 들어가지 않아, 문서와 달리 random adapter로 시작할 수 있다.
`INIT`, `INIT_ADAPTER`, `TRAIN_ENCODER`의 기본값과 검증을 모두 `COMMON` 생성보다
앞으로 옮기고, 실행 직전 최종 `ARGS`와 초기화 출처를 검증해야 한다.

### P1 — 재개는 정확한 data position을 복원하지 않음

`TrainingArguments(ignore_data_skip=True)`이며 `RoundRobinLoader`는 자체 iterator
position을 checkpoint에 저장하지 않는다. 따라서 optimizer·scheduler·RNG·global
step은 복원되지만, 중단된 epoch 안의 정확한 EN/KO batch 위치는 복원되지 않는다.
60→90 smoke의 연속 loss는 상태 복원을 확인하지만 **무중단 run과 가중치가 같음을
입증하지 않는다**.

다음 중 하나가 필요하다.

- 우선 `ignore_data_skip=False`로 중단 run과 무중단 run의 최종 parameter hash를
  비교한다.
- 장기적으로 `RoundRobinLoader`가 corpus별 sampler epoch·batch offset·다음 corpus를
  `state_dict()`로 저장하고 O(1)에 복원하게 한다.

### P1 — HF checkpoint에서 recipe override가 무시됨

`--init`이 VAPASR HF 디렉토리이면 config의 기존 `next_weight`,
`next_weight_ko`, `delays`를 그대로 읽고 CLI 값을 다시 쓰지 않는다. 따라서 HF
checkpoint에서 시작하는 KO weight sweep이 명령행 표시와 달리 이전 weight로 돌 수
있다. 로드 직후 CLI override를 config에 명시적으로 반영하고 `run.json`에는 최종
effective config를 저장해야 한다.

### P1 — 평가 이력은 재개 시 이어지지 않음

`VapAsrTrainer.eval_hist`는 매 프로세스 시작 때 빈 리스트이고 checkpoint state에
포함되지 않는다. 재개 후 `eval/hist.json`과 최종 `results.json`이 이전 sentinel
곡선을 잃을 수 있다. 기존 `hist.json`을 읽어 이어 붙이거나
`trainer_state.json.log_history`에서 재구성해야 한다.

### P1 — 부분 checkpoint 검출

2-GPU에서 6.8 GB checkpoint 저장에 약 1분이 걸렸다. 8-node/NFS에서 강제 종료가
저장 중 발생하면 `get_last_checkpoint()`가 가장 번호가 큰 불완전 디렉토리를 고를
수 있다. 각 checkpoint 저장 완료 뒤 `.complete` marker를 원자적으로 기록하고,
재개 시 필수 파일과 marker가 있는 최신 checkpoint만 선택한다. 직전 complete
checkpoint 한 개는 rotation에서 항상 보존한다.

### P2 — “HF-native” 공개 계약 마무리

현재는 custom class의 `from_pretrained()`가 되는 단계다. 일반 HF 사용자가 별도
프로젝트 import 없이 쓰는 수준에는 다음이 남았다.

- `input_ids`, `attention_mask` alias와 `audio_values`, `audio_features` 등 표준화된
  forward 인자
- `AutoConfig`/`AutoModel` 등록 또는 Hub `auto_map`
- tokenizer와 audio/TN 검사를 묶는 `VapAsrProcessor`
- config에 TN fingerprint, tokenizer·encoder revision/hash, model format version 기록
- `--select`와 `--final`의 HF evaluator 이관

또한 `RoundRobinLoader`의 한 epoch는 각 corpus 1회가 아니라 **모든 corpus의 batch
수 합만큼 1:1 교대**다. 데이터 크기가 다르면 작은 corpus가 반복되고 큰 corpus는
일부만 보므로, `30 epoch`와 함께 corpus별 실제 batch 수·audio-hour·effective pass를
반드시 기록해야 한다. 이는 기존 학습 의미론을 보존한 것이지만 일반적인 epoch
해석과 다르다.

### Liger 판정

측정된 969→734 ms/step은 약 24% 개선으로 채택 근거가 있다. 그러나 같은 checkpoint의
bf16 loss가 `6.8e-4` 달라지고 63개 decode token 중 2개가 달랐으므로, Liger on/off를
같은 실험 곡선으로 섞지 않는다. package version과 patch 개수를 run metadata에
고정하고, 300-step overfit 및 고정 sentinel에서 수렴·방출 지표가 같은 범위인지 한 번
더 확인한 뒤 주력 recipe의 기본값으로 승격한다.

현재 판정은 **구조 전환 통과, 8-node 실행은 P0 수정 후 조건부 진행**이다.
