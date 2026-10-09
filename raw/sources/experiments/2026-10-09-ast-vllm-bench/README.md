---
author_id: tskim
created: 2026-10-09
source_type: source-drop
---

# AST 단조 번역 생성 속도: HF vs HF+fla vs vLLM (mxc, 2026-10-09)

- 스크립트: `experiments/ast_mono_pilot.py run`(프롬프트 v2, 커밋 2eb900a 와 같음), en-ko. 모두 Qwen3.8-27B 이고 greedy 로 디코딩했으며 생각 모드는 껐다. GPU 는 mxc 호스트 H200 한 장이다.
- 설정:
  - hf: `semcommit-teacher` env, transformers 5.8.0, torch 구현(빠른 커널 없음), 배치 32. 2026-10-08 v2 결과를 그대로 썼다.
  - hf_fla: `teacher-fla` env(`scripts/setup-teacher-fla-env.sh`), 위 구성에 flash-linear-attention 0.5.2·causal-conv1d 1.7.0 을 더했다. 빠른 경로가 켜진 것을 확인했고, 배치 32 다.
  - vllm: vLLM 0.30.0 env(`scripts/setup-vllm-env.sh`, torch 2.13.0+cu130). `--engine vllm`, max_num_seqs 256, `VLLM_USE_FLASHINFER_SAMPLER=0`.
  - vllm_400: 같은 vLLM 에 표본을 400 문장으로 늘렸다(`sample --n 400 --seed 1`, 영어 1.219 h). max_num_seqs 는 512 다.
- 결과(`summaries.json`, 처리량 = 출력 토큰 / 생성 시간):

| 설정 | 문장 | 로드 s | 생성 s | 출력 토큰 | 토큰/s | 관문 통과 일반/단조 | AL 일반/단조 s |
|---|---:|---:|---:|---:|---:|---|---|
| hf | 50 | 78 | 307 | 46,816 | 152 | 39 / 42 | 4.69 / 4.11 |
| hf_fla | 50 | 94 | 405 | 46,730 | 115 | 41 / 41 | 4.62 / 3.96 |
| vllm | 50 | 126 | 44 | 46,860 | 1,072 | 40 / 42 | 4.56 / 4.13 |
| vllm_400 | 400 | 109 | 338 | 387,269 | 1,145 | 341 / 345 | 4.95 / 4.26 |

- 첫 vLLM 실행은 엔진 준비 중에 멈췄다. FlashInfer 샘플링 커널을 그 자리에서 빌드하려다 `ninja` 를 PATH 에서 찾지 못했다. greedy 라 이 샘플러가 필요 없어 `VLLM_USE_FLASHINFER_SAMPLER=0` 으로 껐고, venv bin 도 PATH 에 넣었다.
- 단조(b)만 따로 센 출력 토큰은 vllm_400 에서 약 18 만 4 천 개다. 판정과 앞부분 번역을 포함하고, Qwen tokenizer 로 다시 셌다. 영어 1.219 h 분량이다.
- mxc 결과 폴더: `VAPKT-data/results/ast-mono-pilot-v2-fla-20261009/`, `ast-mono-pilot-v2-vllm-20261009/`, `ast-mono-bench-vllm-20261009/`
