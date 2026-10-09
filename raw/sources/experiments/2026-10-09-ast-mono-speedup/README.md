---
author_id: tskim
created: 2026-10-09
source_type: source-drop
---

# AST 단조 번역 생성 가속 시험: prefix cache(v3 프롬프트)·FP8·메모리 설정 (mxc 호스트 H200, 2026-10-09)

- 배경: 노드 2 개(GPU 16 장)만 쓸 수 있어 v2 bf16 으로는 약 42 시간이 걸린다. 그래서 메모리·연산 효율을 검토했다.
- 진단:
  - en-de 샤드 하나의 프롬프트는 692 만 토큰이고 출력은 62 만 토큰이다. 그래서 prefill 연산이 병목이다(bf16 최대 성능의 약 53 %).
  - 호출마다 공통인 지시문(약 230 토큰)도 다시 계산된다. Qwen3.8 은 Gated DeltaNet 혼합 구조라 vLLM 이 attention 블록을 784 토큰으로 잡고(로그 "Setting attention block size to 784 tokens"), 캐시는 이 단위로만 재사용된다. 그래서 적중이 0 이었다.
- v3 프롬프트(`experiments/ast_mono_pilot.py` `p_system_v3`·`p_translate_v3`):
  - v2 단조 번역 지시를 방향별 고정 시스템 메시지로 옮겼다.
  - 예시를 늘렸다. 완결 발화 9–10 개, 미완결 발화 3–4 개다. 구어 전사(간투사·반복)의 예시도 넣었다.
  - 그 결과 공통 앞부분이 784–786 토큰(완결 번역 호출), 811–813 토큰(앞부분 번역 호출)이 됐다. Qwen tokenizer 로 셌다.
  - 사용자 메시지는 v2 상태 문장과 원문뿐이다. 판정 프롬프트는 v2 그대로다.
- `comparison.txt`:
  - 같은 en-de u-00000(1,024 단위)에서 다섯 설정을 비교했다(시간·관문·AL·캐시 적중·v2 bf16 과 같은 번역 수).
  - 파일럿 세트(en-ko 400, en-de·de-en·ko-en 50)로 단조 번역 품질을 비교했다.
  - 다른 사용자의 가벼운 작업(4–5 GB, 사용률 약 14 %)이 GPU 0–2 에 함께 있었다. fp8v2 시간은 GPU 0 에서 쟀다.
- `shard-summaries.txt`: 각 설정의 summary.json
- 결과 폴더(mxc):
  - `VAPKT-data/data/ast-mono-v1/gen-t-{fp8v2,bf16v3,bf16v3m,fp8v3}/`
  - `VAPKT-data/results/ast-mono-{bench,pilot}-{fp8v3,bf16v3}-20261009/`
