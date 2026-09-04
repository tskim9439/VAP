---
type: output
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: Stage 1용 frozen encoder 7종 특징 캐시 설계·검증 — RTF, 저장 용량, 세그먼트 이어붙이기 정확성, AuT 13Hz 발견
sources:
  - [[output-encoder-causality-audit]]
  - [[output-vap-target-pipeline]]
---

# 특징 캐시와 컴퓨트 예산 (Stage 1)

코드: `vapasr/features/encoders.py`(인코더 레지스트리), `experiments/extract_features.py`, `experiments/run_feature_cache.sh`,
진단: `experiments/diag_length_invariance.py`, `diag_stitching.py`, `diag_qwen_determinism.py`.
캐시 위치: `/data3/tskim/features/<encoder>/<manifest>/<id>.npy` (2, T′, D) fp16 + `index.jsonl` + `stats.json`.

## 인코더 레지스트리 (공통 `encode(wav (2,T)@16k) → (2, T′, D)`)

| 이름 | Hz | D | lookahead | causal | RTF(stereo, A100) | 세그먼트 |
|---|---:|---:|---|---|---:|---|
| cpc | 50 | 256 | 0 | ✓ | **0.0018** | 300 s + 좌측 60 s |
| nemotron-c0 | 12.5 | 1024 | ≤80 ms | ✓ | **0.0052** | 270 s + 좌측 120 s |
| nemotron-c1 | 12.5 | 1024 | ≤160 ms | ✓ | ≈0.005 | 〃 |
| qwen-aut-cc1s | **13.0** | 1024 | 0–800 ms | 블록 | 0.018 | 240 s + 좌측 150 s, 프론트엔드 파일 단위 |
| qwen-aut-causal | **13.0** | 1024 | ≤80 ms | ✓ | 0.018 | 〃 |
| wavlm-base | 50 | 768 | ∞ (20 s 창) | ✗ | 0.017 | 20 s 독립 창 |
| wavlm-large | 50 | 1024 | ∞ (20 s 창) | ✗ | 0.019 | 20 s 독립 창 |

Stage 1 셋 **214 h** (otoSpeech 104.9 + AI Hub TS_01_5 50 h 서브셋 + VS_02 51.7 + TurnBench dev 7.3):
추출 시간 ≈ cpc 0.4 h · nemotron 1.1 h · qwen 3.9 h ×2 · wavlm 3.7 h + 4.1 h ≈ **17 GPU 시간** (GPU 1, 순차 2 작업).
저장 ≈ 39 + 39 + 39 ×2 + 118 + 158 = **430 GB** (/data3 여유 2.2 TB). peak GPU 1.4–2.4 GB.

## 세그먼트 이어붙이기 — 실측으로 잡은 함정 4개

긴 파일(최대 78 min)을 세그먼트로 나눠 인코딩하되 **무분할과 fp32 에서 일치**해야 한다. 진단 순서대로:

1. **유효 수용장 = 층당 좌측 context × 층수.** Nemotron 56 frame × 24 층 ≈ **107 s**. 20 s 겹침에서는 세그먼트 시작 후
   ~100 s 가 달랐다(rel 3.7). → 겹침 120 s. Qwen(8 s 창 × 18 층 = 144 s) → 150 s.
2. **TF32 가 켜져 있으면 배치 길이에 따라 값이 흔들린다** (CPC rel 7e-3). → 추출 시 TF32 off.
3. **출력 프레임 수는 `길이 × Hz` 가 아니다.** Nemotron 은 패딩으로 +1~2, **Qwen AuT 는 1 s chunk 당 13 프레임(=13 Hz, 12.5 아님)**.
   세그먼트별 출력을 기대 길이로 잘라 붙이지 않으면 경계마다 밀린다. → `expect = round(구간 길이 × Hz)` 로 트림, Qwen 은 13.0 Hz 로 선언.
4. **Whisper 프론트엔드의 utterance-max 정규화** → 세그먼트별로 fbank 를 뽑으면 스케일이 다르다. → 파일 단위 1회 계산 후 특징 축에서 분할.
   + 세그먼트 경계는 80 ms(=1280 샘플) 격자에 정렬 (40 ms 어긋나면 전 프레임이 바뀜).

검증 결과 (304 s 파일, 100 s 세그먼트 vs 무분할, fp32, 차원별 std 정규화): cpc 0 프레임 / nemotron-c0 0 / qwen-causal 5 (max 8.7e-3) /
qwen-cc1s 1 — 모두 수치 드리프트 수준. fp16 저장 시 플립 비율 1–2 % (정밀도 한계, 무해).

## 결정 사항

- **WavLM 은 비인과 참조** — 20 s 창 내 양방향. lookahead ∞ 로 표기하고 Stage 1 표에서 별도 행으로 둔다.
- Qwen 두 모드는 마스크를 학습 블록(8 s)에 맞춘 창으로 제한 — 원 모델의 분포에 가장 가깝고 세그먼트 exact 가 성립.
- AI Hub 학습 서브셋은 id 정렬 후 누적 50 h 컷(결정적). 플래그 파일 제외.
- 로더 병목(otoSpeech 48 k 리샘플 300 ms/item)은 16 k 사본(`otoSpeech16k`)으로 해소.

## 최종 (2026-09-04 완료)

| 인코더 | 대화 | 시간 | 용량 | RTF(공유 GPU) | peak GPU |
|---|---:|---:|---:|---:|---:|
| fbank (floor) | 816 | 213 h | 12 GB | 0.0002 | 1 GB |
| cpc | 816 | 205 h | 39 GB | 0.0025 | 9 GB |
| nemotron-c0 | 816 | 205 h | 39 GB | 0.010 | 13 GB |
| qwen-aut-cc1s | 816 | 205 h | 41 GB | 0.005 | 8 GB |
| qwen-aut-causal | 816 | 205 h | 41 GB | 0.014 | 8 GB |
| wavlm-base | 816 | 205 h | 118 GB | 0.013 | 10 GB |
| wavlm-large | 816 | 205 h | 157 GB | 0.018 | 18 GB |
| **합계** | | | **447 GB** | | |

세 작업을 GPU 1 에서 병행해 약 26 h 소요 (단독이면 ≈17 h). peak 는 78 min 짜리 AI Hub 파일의 390 s 세그먼트에서 발생 —
공유 GPU 에서는 대화별 `empty_cache()` 가 필요했다. TurnBench dev 는 scorer 가 38 대화 전부를 요구하므로 플래그 대화(tb-172)도 포함.
`experiments/show_cache_stats.py` 로 재확인.
