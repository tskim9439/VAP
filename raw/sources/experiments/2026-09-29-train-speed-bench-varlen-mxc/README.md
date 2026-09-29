---
author_id: tskim
created: 2026-09-29
source_type: source-drop
---

# thinker varlen attention 학습 처리량 벤치 (mxc 호스트 GPU 6, H200, 2026-09-29)

- 대상: `--attn-impl varlen` = torch 2.9 내장 FlashAttention-2 varlen 커널(`torch.ops.aten._flash_attention_forward`)을
  transformers `AttentionInterface` 에 `vapasr_varlen` 으로 등록(`vapasr/hf/varlen_attention.py`). flash-attn 패키지는 쓰지 않는다.
  - packing(`pack:N`): position_ids 로 cu_seqlens 를 만들어 샘플마다 따로 attention(Σnᵢ²).
  - 패딩 배치(`budget:N`): 모델 안에서 유효 토큰만 한 줄로 unpad 해 같은 커널로 돌린다(pad 토큰은 어느 층에서도 계산하지 않음).
- 스크립트: `experiments/train_speed_bench.py`(설정 이름 뒤 `@attn`). 09-27 벤치와 같은 모델·데이터:
  `/soundai/Model/VAPASR/hf-approved-qwen-v1-e10-n2/checkpoint-35000` + add_semantic_tokens, 인코더 동결, grad ckpt, bf16 autocast,
  Liger + fused AdamW, speechlm-all19-v035 part 앞 40 개 main 풀(KO 22,739 스트림), δ 2/3/4/6, `--pack-max-bs 256`, workers 16.
- 코드는 벤치 복사본(`/soundai/users/tskim/VAPKT-data/bench/speedup-20260927/code`)에만 올렸다(로컬 작업본과 md5 일치 확인,
  이전 파일은 `*.pre-varlen-20260929` 로 보존). 배포 복사본·학습 출력은 건드리지 않았다.
- 배치를 먼저 메모리로 읽고 그 배치로 GPU 계산만 잰다. 1차는 steps 12(warmup 3), 2차는 순서를 뒤집어 steps 24(warmup 4).
  같은 `budget:N` 설정끼리는 샘플러가 결정적이라 배치 구성이 같다(attention 만 다름).

## 결과 (오디오 시간 / GPU 시간)

| 설정 | attention | 샘플/step | s/step (1차 · 2차) | 오디오 h / GPU h (1차 · 2차) | 최대 메모리 |
|---|---|---|---|---|---|
| `budget:16384` | sdpa | 53 | 0.416 · 0.392 | 1,252 · 1,331 | 25 GB |
| `budget:16384@varlen` | varlen (모델 안 unpad) | 53 | 0.313 · 0.321 | 1,665 · 1,621 | 25 GB |
| `pack:16384@varlen` | varlen (cu_seqlens) | 53 | 0.309 · 0.305 | 1,688 · 1,711 | 25 GB |
| `budget:32768` | sdpa | 106 | – · 0.750 | – · 1,400 | 41 GB |
| `budget:32768@varlen` | varlen | 113 | 0.613 · – | 1,695 · – | 41 GB |
| `pack:32768@varlen` | varlen | 106–113 | 0.542 · 0.554 | 1,921 · 1,901 | 41 GB |
| `pack:16384` | sdpa | 53 | – · 2.361 | – · 221 | 25 GB |

- 같은 배치에서 sdpa → varlen: `budget:16384` 는 오디오 h/GPU h +33 %(1차) · +22 %(2차), s/step −25 % · −18 %.
  `pack:16384@varlen` 은 `budget:16384`(sdpa) 대비 +35 % · +29 %. 32 k 예산에서는 `pack:32768@varlen` 이 `budget:32768`(sdpa) 대비 +36 %.
- 설계 단계 예상(−10~15 %)보다 이득이 크다. KO main 은 실토큰 비율 0.996 이라 패딩 제거 몫은 작고, sdpa 가 패딩 마스크 때문에
  flash 대신 mask 경로(GQA repeat_kv, causal skip 없음)로 떨어지던 비용이 대부분인 것으로 보인다(thinker 단독 측정과 방향 일치).
- `pack:16384`(sdpa) 2.36 s/step 은 09-27 값(2.363)을 재현한다 — 그때 packing 이 느렸던 원인은 thinker attention 의 N² 마스크.
- 최대 메모리는 attention 구현과 무관하게 예산이 정한다(16 k 25 GB, 32 k 41 GB).
- 같은 설정의 1·2차 차이는 최대 약 8 %(짧은 측정, 순서·GPU 상태). 09-27 의 `budget:16384` 0.495 s/step 과도 다른 날 측정이라 직접 비교하지 않는다.
- 여기 데이터 처리량(1,190–3,313 오디오 h/h)은 페이지 캐시에 올라간 뒤의 값이다. 실제 학습의 콜드 오디오 I/O(09-27: 23–103 h/h)가
  여전히 병목이면 이 GPU 이득은 end-to-end 에 드러나지 않는다.
- EN 긴 스트림 end-to-end 는 돌리지 못했다: 벤치 디렉터리의 `en-trial*` 에는 words 만 있고 semcommit 라벨이 없다.
  긴 시퀀스 이득은 설계 단계의 thinker 단독 측정(혼합 8 × 200–4000: 1,327 → 256 ms, 긴 8 × 3500–4000: 1,484 → 488 ms)만 있다.

## GPU 정합성 테스트 (`tests/test_packing.py`, CUDA_VISIBLE_DEVICES=5, 13 passed)

- 커널(bf16) 대 fp32 구간별 참조(lens 300/17/1200/5/800, GQA 16/8, D128): out 0.19 %, dq 0.25 %, dk 0.31 %, dv 0.29 %. q 3·1 / k 10 에서 bottom-right causal 일치.
- tiny 모델 bf16 autocast, 기준 = 패딩 sdpa: packed varlen · 패딩 varlen 손실 차이 0, grad(adapter LN, 임베딩, k_proj) 0.42 / 0.03 / 0.45 %.
  같은 조건의 packed sdpa(정답 경로)는 0.21 / 0.03 / 0.31 %.
- KV 캐시 decode(prefill 9 → 1 → 4, 층 가중치 std 0.2 로 키워 attention 이 잔차를 지배하게 함): varlen 과 sdpa 가 비트 단위로 같고
  (H200 에서 sdpa 도 같은 FA2 커널로 간다) 둘 다 fp32 eager 대비 0.85–1.14 %. 일부러 top-left 로 정렬한 음성 대조는 100 % 어긋난다.
  → 설계 단계에서 의심했던 "decode logits 차이 정확히 0.0" 은 정상(기본 0.02 초기화에서는 bf16 logits 반올림이 attention 차이를 가린다).
- 별도 진단(stdin 실행, std 0.2 가중치, fp32 eager 대비 학습 grad): 패딩 sdpa 1.3–2.0 %, packed sdpa 1.2–2.0 %, packed·패딩 varlen 1.3–2.0 % — 같은 크기.

원자료: `bench-varlen.json`(1차), `bench-varlen-rev.json`(2차), `gpu-tests.txt`(pytest 출력 요약 두 번: 두 번째가 decode 음성 대조를 넣은 최종 테스트).

## 리뷰 반영 보정 (2026-09-29 14:00–14:30 KST)

리뷰 두 건의 지적을 확인하고 원자료를 보탰다. 위 본문은 고치지 않았다. 새로 올리거나 만든 벤치 복사본 파일은 모두 새 이름이다.
덮어쓴 파일은 원본을 `*.pre-review-20260929` 로 보존했다. lr 1e-8 대조에 쓴 중간판은 `train_speed_bench.py.review-lr1e-8-20260929` 로 남겼다.
삭제한 파일은 없다.

### 1. thinker 단독 마이크로벤치 재측정 (`thinker-microbench.py` · `thinker-microbench.json`)

- 설계 단계의 thinker 단독 수치(2.33 s, 251/215/175 ms, 1,327 → 256 ms, 1,484 → 488 ms)는 프로토타입을 stdin 으로 돌린 출력이다.
  파일로 남지 않아 출처를 확인할 수 없었다. 그래서 최종 구현(벤치 복사본 코드)으로 같은 입력 길이 난수열을 다시 쟀다.
  - 조건: mxc 호스트 GPU 1, H200, torch 2.9.0+cu128, transformers 4.57.6, Qwen3-ASR-0.6B text 구조(random init),
    Liger(RMSNorm·SwiGLU·RoPE), grad ckpt, bf16 autocast, fwd+bwd, 8회 평균(packed sdpa 는 3회).
  - `.py` 는 실행한 스크립트를 그대로 둔 것이고, `.json` 은 그 출력이다.

| 배치 (실토큰) | 패딩 sdpa | varlen, 모델 수준 unpad (학습 경로) | packed varlen | 층별 unpad (분기 2) | packed sdpa |
|---|---|---|---|---|---|
| ko54: 54 × 250–320 (15,491, 패딩 10 %) | 251.0 ms | 187.3 ms | 184.2 ms | 252.8 ms | 2,337 ms |
| mix8: 8 × 200–4,000 (17,291) | 1,324.6 ms | 258.0 ms | 252.7 ms | 386.8 ms | 2,873 ms |
| long8: 8 × 3,500–4,000 (30,275) | 1,476.4 ms | 513.4 ms | 481.3 ms | – | – |

- 재현된 값: 패딩 sdpa, packed sdpa, mix8, long8 은 설계 단계 수치와 ±1 % 안에서 같다.
- 달라진 값은 두 가지다.
  - packed varlen: 175 → 184 ms.
  - 층별 unpad: 215 → 253 ms. 설계 단계는 프로토타입 구현이었고 Liger 도 RMSNorm·RoPE 만 켰으므로, 그 차이로 추정한다.
- 층별 unpad 는 attention 밖의 연산을 패딩 행째로 계산한다. 그래서 패딩 10 % 배치에서는 sdpa 와 거의 같다.
  반면 모델의 패딩 경로(`_thinker_unpadded`)는 packed 와 같은 계산을 한다(187 대 184 ms).
- docstring 의 "0.25 s with varlen" 은 사실 패딩 sdpa 값이었다. `speedup.py`·`packing.py`·`varlen_attention.py` 의 수치와 라벨을
  위 표 기준으로 고쳤다.
- ko54 는 합성 배치라 패딩이 10 % 다. 실제 KO main 배치의 패딩은 0.4 % 이므로, 실제로는 패딩 제거 몫이 이보다 작다.

### 2. 09-27 sdpa 기준선의 신뢰도

- 09-27 `bench-budget.json`(GPU 7)은 한 run 안에서도 단조롭지 않다.
  - 처리량: budget:16384 1,043 → budget:32768 914 → budget:65536 1,345 h/h.
  - 16 k → 32 k 에서 토큰은 2 배인데 s/step 은 0.495 → 1.13 으로 2.3 배다.
  - 09-29 는 같은 구간에서 0.39–0.42 → 0.75 로 1.9 배라 일관된다.
  - 따라서 09-27 의 budget sdpa 수치는 신뢰도가 낮다고 본다.
- 같은 sdpa 설정이 날짜에 따라 크게 달라졌다. 이 폭은 varlen 이득(+22–33 %)과 비슷하거나 더 크다.
  - budget:16384: 1,043 → 1,252–1,331 (+20–28 %).
  - budget:32768: 914 → 1,400 (+53 %).
  - 반면 pack:16384 sdpa 는 2.363 → 2.361 s/step 으로 재현된다.
- 현재 sdpa 기준선은 09-29 값인 1,250–1,400 h/h(budget:16384·32768)로 둔다.
  09-27 의 1,043 을 기준으로 인용하면 varlen 이득이 +55–64 % 로 부풀려진다. 채택 판단은 같은 run 안에서 같은 배치로 한 A/B 로만 한다.

### 3. 호스트 공유와 측정 잡음

- 14:05 KST 에 관찰한 mxc 호스트 GPU 사용 상황:
  - GPU 3·4: root 의 CosyVoice 학습(deepspeed, 약 28 시간째, 사용률 60–100 %).
  - GPU 5·6: vLLM TP 워커(13:46 경 시작).
  - GPU 7: vLLM(약 15 시간째).
- 1·2차 벤치(GPU 6, 13:03–13:30)는 GPU 5·6 의 vLLM 이 뜨기 전이었다. 그래도 CosyVoice 학습·GPU 7 vLLM 과 같은 호스트의
  CPU·PCIe·저장소를 나눠 썼다.
- 아래 4 의 lr 0 run(GPU 0, 14:18–14:26)은 모든 설정이 1·2차보다 30–50 % 느렸다(pack:16384 sdpa 3.52 s/step).
  - 우리 run 이 끝난 직후(14:26) GPU 0–2 에 우리 것이 아닌 프로세스가 8–26 GB 씩 올라와 있었고, 호스트 load average 는 73–80(96 코어)이었다.
  - 그래서 이 run 의 처리량은 쓰지 않고 loss 대조에만 쓴다.
- 1·2차는 steps 12/24 로 짧다. 또 1차는 sdpa 를 첫 설정으로 돌렸다(2차에서 순서를 뒤집어 방향이 같음을 확인했다).
- 권고: 기본값을 varlen 으로 바꾸기 전에 두 가지를 다시 확인한다.
  - 유휴 GPU 에서 sdpa/varlen 을 번갈아(ABAB) 40 step 이상 반복 측정한다.
  - 콜드 오디오 I/O 가 포함된 실제 학습 end-to-end 로 +5 % 게이트를 확인한다.

### 4. Liger + varlen 실모델 loss 대조 (`bench-varlen-parity-lr1e-8.json`, `bench-varlen-parity-lr0.json`)

`train_speed_bench.py` 에 두 가지를 더했다.

- 배치를 읽기 전에 시드를 고정한다. 그래서 워커의 δ 추첨까지 포함해, 같은 설정이면 배치가 비트 단위로 같다.
- 첫 warmup step 의 loss 와 배치 지문(ids·labels·pos_weight·K 의 md5)을 기록한다.

비교는 모델·데이터가 위와 같고, steps 4 · warmup 2 로 돌렸다.

- 첫 시도(lr 1e-8, GPU 1)에서는 설정 사이 비교가 불가능했다.
  - 배치 지문은 설정 간에 같았다.
  - 그런데 loss 가 설정 순서대로 1.0617(sdpa) → 1.0591(varlen) → 1.0574(sdpa 반복)로 내려갔다. sdpa 반복끼리도 0.40 % 차이가 났다.
  - 원인: AdamW 는 부호형 갱신이다. 그래서 lr 1e-8 이라도 수억 개 파라미터가 모두 loss 를 줄이는 방향으로 움직이고,
    6 step(설정 하나)마다 loss 가 0.16–0.24 % 줄었다. 이 변화가 설정 사이 비교를 가렸다.
  - 확인: 두 run 의 첫 설정(budget:16384 sdpa, 앞선 step 없음)은 lr 과 무관하게 1.0616534 로 같다.
- 그래서 optimizer 를 lr 0 으로 바꿨다. step 계산량은 같고 가중치만 고정된다.
- lr 0 결과(GPU 0): sdpa 반복끼리는 loss 가 비트 단위로 같았다(forward 결정적). varlen 과 sdpa 의 차이는 다음과 같다.

| 설정 (같은 배치) | sdpa 첫 loss | varlen 첫 loss | 상대 차이 |
|---|---|---|---|
| budget:16384 | 1.0616534 (반복도 동일) | 1.0622249 | 5.4e-4 |
| pack:16384 | 1.0140048 | 1.0139465 | 5.7e-5 |
| budget:32768 | 1.1347315 | 1.1345253 | 1.8e-4 |

- 결론: Liger 를 켠 실제 모델에서 varlen 과 sdpa 의 loss 차이는 0.06 % 이하다. bf16 attention 오차 범위다.

### 5. GPU 정합성 테스트 재실행

- 리뷰 반영 후 `tests/test_packing.py` 를 GPU 1 에서 다시 돌렸다: 16 passed(3 분 12 초). 추가된 테스트는 3 개다.
  - gradient checkpointing 재계산 때도 cu kwargs 가 유지되는지(non-reentrant·reentrant).
  - B > 1 행별 구간.
  - cu kwargs 와 패딩 마스크를 함께 넘기면 거부하는지.
- 기존 CUDA 테스트 수치는 앞의 실행과 같다(`gpu-tests.txt` 끝에 추가).

원자료(추가분):
- `thinker-microbench.py` · `thinker-microbench.json`: thinker 단독 재측정.
- `bench-varlen-parity-lr1e-8.json`: lr 1e-8 첫 시도. 가중치 변화로 loss 대조가 가려진 run 이다.
- `bench-varlen-parity-lr0.json`: lr 0 loss 대조. 처리량은 호스트 경합으로 쓰지 않는다.
- `gpu-tests.txt`: 리뷰 후 재실행 요약.
