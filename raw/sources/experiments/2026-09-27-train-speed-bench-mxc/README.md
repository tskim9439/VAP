---
author_id: tskim
created: 2026-09-27
source_type: source-drop
---

# 학습 처리량 벤치 (mxc 호스트 GPU 7, H200, 2026-09-27)

- 스크립트: `experiments/train_speed_bench.py` (단일 GPU, semcommit 학습 경로 그대로: SemCommitDataset + semcommit_round_robin)
- 모델: `/soundai/Model/VAPASR/hf-approved-qwen-v1-e10-n2/checkpoint-35000` + add_semantic_tokens, 인코더 동결, grad ckpt, bf16 autocast, sdpa
- 데이터: speechlm-all19-v035 part 앞 40 개의 main 풀(KO 22,747 스트림), δ 2/3/4/6
- 배치를 먼저 메모리로 읽고(데이터 처리량) 그 배치로 GPU 계산만 잰다. steps 12–24 로 짧아 ±15 % 정도 흔들린다.

## 결과 (오디오 시간 / GPU 시간)

| 설정 | 샘플/step | s/step | 오디오 h / GPU h | 최대 메모리 |
|---|---|---|---|---|
| base: bs 16 버킷 패딩, Liger 끔, adamw_torch | 16 | 0.386 | 393 | 15 GB |
| + Liger + fused AdamW | 16 | 0.268 | 567 | 15 GB |
| + `--batch-max-tokens 16384` (샘플 수 × 최장 ≤ 예산) | 54 | 0.495 | 1,043 | 25 GB |
| + `--batch-max-tokens 32768` | 112 | 1.13 | 914 | 41 GB |
| + `--batch-max-tokens 65536` (`--pack-max-bs 256`) | 208 | 1.55 | 1,345 | 72 GB |
| + packing 16384 (sdpa) | 54 | 2.36 | 219 | 25 GB |
| + packing 32768 (sdpa) | 64 | 3.28 | 187 | 35 GB |

- 길이 버킷 배치는 이미 실토큰 비율 0.95–0.97 이라 packing 이 없앨 패딩이 거의 없다. sdpa 에서 packed 행은 행 전체(N²)에 대해 마스크된
  attention 을 계산해 오히려 2.6 배 느리다. packing 은 varlen 커널(flash_attention_2) 이 있을 때만 의미가 있다.
- flex_attention + packing: 손실은 sdpa 와 같지만 gradient 가 틀리다(k_proj 상대 오차 1.0, torch 2.9 / transformers 4.57). 코드에서 거부한다.
- 오디오 I/O 가 실제 병목: 처음 읽는(콜드) 배치는 workers 16 으로 23–103 오디오 h/h — GPU 계산(1,000+)의 5–25 분의 1.
  페이지 캐시에 올라간 뒤에는 540–1,180. 원본 오디오는 `/soundai/databricks_build_managed/...` 에서 읽고, 일부 파일은 없다(이웃 항목으로 대체, 2 건).

원자료: `bench-budget.json`, `bench-packing.json`.
