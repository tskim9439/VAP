---
type: output
status: active
created: 2026-09-29
updated: 2026-09-29
summary: WER/CER 개선 — 오류는 대부분 실제 인식 오류, 2-pass 로 최종 전사 EN −31~63 %·KO −11~25 %, 학습에 증강·인코더 학습·SEM 중립 4.3k h 적용
sources:
  - '[[output-semcommit-v035-d8-eval]]'
  - '[[output-semcommit-two-pass-plan]]'
  - '[[output-stage2-e2-final-eval]]'
  - '[[source-nemotron-3-5-asr-streaming]]'
  - '[[source-qwen3-asr]]'
  - raw/sources/experiments/2026-09-29-semcommit-v035-d8-eval-mxc/
related:
  - '[[output-semcommit-gold-v1]]'
  - '[[decision-asr-backbone]]'
raw_authors:
  - tskim
---

# ASR 정확도 개선: 진단과 적용 (2026-09-29)

사용자 요청은 "WER/CER 수치가 너무 안 좋다, 개선 방안을 고민해 적용해 달라"였다. 먼저 오류를 진단하고 상한을 쟀다. 그 뒤 네 가지 레버를 구현해 적용했다.

## 0. 요약

- **오류는 대부분 실제 인식 오류다.** 새 모델([[output-semcommit-v035-d8-eval]])과 E2 의 오류 구성은 거의 같다.
  - 정규화 영향은 작다. EN 숫자·합성어는 2–3 %, KO 구어/표준 표기 변이는 8–13 % 다.
  - 파국 실패(빈 출력·잘림·환각 반복)도 1 % 미만이다.
- **오프라인 상한과의 격차는 영어에서 크다.** 같은 규약 기준 δ6 대 Qwen3-ASR 오프라인 결과다.
  - 영어: 4.57 대 1.90(1.7B), 11.06 대 3.68(test-other). 2–3 배 차이다.
  - 한국어: 12.0 대 8.9. 26 % 차이다.
- **2-pass(SEM 트리거 재디코드, 커밋 0d2b584)가 최종 전사를 크게 줄인다.** 1.7B iso, δ4 기준 결과다.
  - ls-test 4.80 → 1.80, gs-test 15.0 → 10.4, ks-eval 11.9 → 10.7, ks-long 11.1 → 8.3.
  - 스트리밍 표시(0.3–0.4 s)는 그대로다. 최종 텍스트는 문장 경계에서 나온다(p50 1.5–4.2 s).
- **스트리밍 1차 자체를 위한 학습 옵션을 구현하고 배포했다.** SpecAugment, 속도 변형, 인코더 학습, SEM 중립 ASR 데이터다.
  - SEM 중립 데이터는 E2 한국어 코퍼스 3.5k h 와 라벨 전 speechlm 0.8k h 다.
  - 다음 학습 제출은 사용자가 한다(§6).
- **영어 평가를 VoxPopuli-Cleaned-AA 로 바꿨다**(커밋 ac23dee). 데이터는 로컬 T5 에만 받았다. 서버 업로드는 사용자가 한다.

## 1. 진단 — 오류 유형 (single-turn v1 전체 셋, 새 모델·E2)

분석 스크립트는 mxc `results/err_analysis.py` · `ko_variant.py` 다. 채점 문자열을 rapidfuzz editops 로 정렬했다.

**영어**(LibriSpeech test, δ4)

| 유형 | 비중 |
|---|---:|
| 내용어 치환 | 27–30 % |
| 기능어 치환 | 21–26 % |
| 철자 근사(고유명사) | 18–26 % |
| 기능어 삽입·삭제 | 11–14 % |
| 숫자 | ≤ 1 % |
| 합성어 분리·병합 | ≤ 2 % |

- 기능어 치환 예: in↔and 약 90 회/셋, a↔the, this→the
- 철자 근사 예: rachel→rachael, soames→solmes, murdoch→murdock

**한국어**(Kspon eval, δ4)

- 구어/표준 표기 변이(자모 거리 ≤ 2)는 CER 의 8–13 % 다. 예: 그니까→그러니까, 맞어→맞아, 아냐→아니야. 그 안에도 실제 오인식(만약에→마약에)이 섞여 있다.
- 필러·숫자·로마자 영향은 거의 없다.

**발화 단위**

- 빈 출력, 잘림(가설 < 0.6 × 참조), 환각(> 1.5 × 참조)은 셋마다 0–21 건이다.
- 오류의 18–28 % 가 최악 5 % 발화에 몰려 있다. 특정 파국 모드가 아니라 전반적 정확도 문제다.

## 2. 상한과 기준선 (single-turn v1, EN WER / KO CER 공백 제외, %)

| 시스템 | LS clean | LS other | Kspon clean | Kspon other |
|---|---:|---:|---:|---:|
| E2 δ4 / δ6 | 5.59 / 5.19 | 12.89 / 12.34 | 12.32 / 11.76 | 12.86 / 12.67 |
| 새 모델 δ4 / δ6 | 5.05 / 4.57 | 11.88 / 11.06 | 12.26 / 12.04 | 12.26 / 12.19 |
| Qwen3-ASR-0.6B 오프라인 | 2.30 | 4.65 | 10.82 | 10.68 |
| Qwen3-ASR-1.7B 오프라인 | **1.90** | **3.68** | **8.92** | **8.71** |
| Nemotron 자체 RNN-T [56,0] | 3.80 | 8.47 | 19.46 | 16.80 |
| Nemotron 자체 RNN-T [56,1] / [56,3](+80 / +240 ms) | 3.57 / 3.44 | 8.06 / 7.70 | 18.29 / 17.28 | 16.50 / 15.29 |

- 오프라인 기준선은 같은 manifest(원 발화 + 1 s 무음)와 같은 score_pair 로 쟀다.
- Nemotron RNN-T 는 우리와 같은 인과 인코더에 원래 디코더를 붙인 것이다. 룩어헤드(오른쪽 문맥)를 늘리면 얼마나 좋아지는지를 학습 없이 잰다.
- **영어에서는 같은 인코더 [56,0] 의 RNN-T(지연 거의 0)가 우리 δ6 보다 낫다**(3.80 대 4.57, 8.47 대 11.06). 정보는 인코더 출력에 있는데 adapter·thinker 가 영어에서 다 쓰지 못한다는 뜻이다. 가장 그럴듯한 원인은 영어 학습 데이터 양(라벨 약 1.3k h)과 인코더 동결이다 → §4 의 영어 SEM 중립 데이터·인코더 학습이 이 격차를 겨냥한다. 한국어는 우리 쪽이 훨씬 낫다(12 대 19).
- 룩어헤드 [56,3](+240 ms)의 이득은 RNN-T 에서 EN −9 %·KO −9~11 % 다. δ 를 늘리는 것과 비슷한 교환이라 우선순위 레버가 아니다([56,1] 은 NeMo 가 지원 목록 밖이라고 경고한다).

## 3. 2-pass 최종 재디코드 (vapasr/hf/twopass.py, experiments/semcommit_twopass_eval.py, 커밋 0d2b584)

- 1차 스트리밍 스트림에서 유효 `<SEM_END>`(단어 중간 아님) 사이를 세그먼트로 자른다. 절단점은 (k_last − δ + 1)·0.08 s 로 인과적이다.
- 0.8 s 미만이거나 1차 단어가 2 개 미만인 세그먼트는 다음 세그먼트와 합친다. 끝 세그먼트는 오디오 끝에서 확정한다.
- 각 세그먼트를 Qwen3-ASR 로 다시 디코드한다. iso = 세그먼트만, text = 앞 확정 텍스트를 context 로 넣는다.

δ4, 골드 v1 스트림:

| 세트 | 1차 | 1.7B iso | 1.7B text | 1.7B 오프라인(s5) | 0.6B iso |
|---|---:|---:|---:|---:|---:|
| ls-test (EN 낭독) | 4.80 | **1.80** | 1.72 | 1.63 | 2.07 |
| gs-test (EN 대화체) | 15.03 | **10.36** | 10.10 | 9.69 | 9.54 |
| ks-eval (KO 짧은) | 11.93 | **10.67** | 11.27 | 9.59 | 12.35 |
| ks-long (KO 긴) | 11.05 | **8.25** | 10.26 | 7.55 | 10.49 |

- 최종 텍스트 지연(알고리즘, 계산 제외) p50 / p90: ls 4.2 / 14.9 s, gs 2.8 / 7.4 s, ks-eval 1.5 / 4.8 s, ks-long 2.7 / 8.3 s. 1차 표시 지연은 0.32–0.39 s 로 그대로다.
- 2차 계산은 오디오 길이의 약 2–3 % 다(H200, bf16).
- 2-pass 를 쓰면 δ2 와 δ4 의 최종 품질이 거의 같다. δ 는 실시간 표시에만 영향을 준다.
- 주의할 점:
  - KO text 모드는 짧은 꼬리 세그먼트에서 앞 문장을 통째로 되풀이한다(셋마다 3–6 건). KO 는 iso 를 쓴다.
  - 0.6B 는 Kspon 에서 1차보다 나쁘다(도메인 적응된 1차가 이긴다).
  - 영어 commit 재현율이 낮아 ls-test 세그먼트가 길다(p90 약 17 s). 최대 세그먼트 강제 절단이 필요하다.
  - Qwen 의 아라비아 숫자 출력이 KO 채점에서 불리하다. 이 영향은 수치로 따로 재지 않았다.
- 계획서의 VAD·고정창·오라클 대조, 군집 bootstrap, 전체 셋·VoxPopuli-AA 평가는 남았다([[output-semcommit-two-pass-plan]]).

## 4. 스트리밍 1차 모델을 위한 학습 레버 (구현·배포 완료)

| 레버 | 구현 | 근거 |
|---|---|---|
| SpecAugment | vapasr/features/online.py(커밋 1ee510b) · `--spec-augment light\|nemo\|Fm,Fw,Tm,Tw` | 학습에 증강이 전혀 없었다. 마스크 0 = Nemotron 사전학습(2×27·10×5 %)과 같은 입력 분포 |
| 속도 변형 | semcommit_dataset(리샘플 + 이벤트 시각 1/배율, K·시퀀스 재구성) · `--speed-perturb 0.9,1.0,1.1` | 표준 ASR 증강, 시각 라벨을 함께 변형 |
| 인코더 학습 | `--train-encoder --lr-encoder 1e-5`(기존 옵션) | 76601 은 인코더를 동결했다. E2 에서 인코더 해동이 −21~−42 % 로 가장 컸다 |
| SEM 중립 ASR 데이터 | 모델 sem_mask(커밋 1ee510b: 그 행 softmax 에서 `<SEM_END>` 제외) + 데이터셋 sem_free · `--asr-words @목록 --asr-max-ratio R` | 라벨 없는 데이터도 텍스트·NEXT 는 배우고, 'SEM 아님' 신호는 주지 않는다 |

SEM 중립 데이터(목록 `semcommit-work/asr-words-v1/lists/asr-words-20260929.list`):

| 원천 | 파트 | 행 | 시간 |
|---|---:|---:|---:|
| 라벨링 전 파트(영어 541 = LibriSpeech·VoxPopuli·YODAS·Switchboard 나머지, 한국어 speechlm 217) | 758 | 223,899 | 영어 806 h + 한국어 |
| E2 Kspon-full | 238 | 619,034 | 1,186 h |
| E2 NIKL-1000 | 279 | 1,205,360 | 1,393 h |
| E2 AIHub 71631 / bc | 48 / 142 | 228,264 / 431,447 | 238 / 707 h |

- E2 한국어 코퍼스는 speechlm 선정에서 빠졌던 것이다(Kspon·AIHub 134 제외). 그래서 새 모델은 Kspon 을 다시 보지 못했다(Kspon δ4·δ6 이 E2 대비 개선 없음). 이 데이터를 다시 넣으면 망각을 막고 도메인도 보강할 수 있다.
- 남은 speechlm 파트 3,585 개의 words 는 `slurm/semcommit-words-only.sbatch` 로 만든다. CPU 작업이고 파트당 약 46 s 다(1 파트 시험). 32 task 면 약 1.5 h 다.
- 정합성:
  - 로컬 테스트 전체 통과(알려진 rapidfuzz 2 건 제외).
  - SEM 중립 행·풀 라우팅·속도 변형 청크 불변식·sem_mask 기울기 0 을 단위 테스트로 확인했다.
  - mxc 스모크(§5)에서 셋마다 시퀀스 불변식 불일치 0 이었다.

## 5. GPU 스모크 (mxc GPU 2, 30 step, 부분 목록: EN 12 + KO 12 파트 + short 8 + SEM 중립 3)

- 설정: init v035-snap0929-d8, 인코더 학습(lr 1e-5), SpecAugment light, 속도 0.9/1.0/1.1, SEM 중립 비율 1.0, varlen, δ 2–8, 예산 8,192 토큰.
- 데이터 검사: 셋마다 64 항목 × δ 5 개 시퀀스 불변식 불일치 0. KO 라벨 1,891 + SEM 중립 1,889(상한으로 6,057 뺌), short 풀은 speechlm 밖 행 7,960 을 건너뜀.
- 30 step 완주(EXIT 0), 최대 GPU 메모리 **37.6 GB**(예산 8,192). s/step 은 공유 호스트·단일 GPU 라 참고만(2.8 s, 30 step 째).
- 저장·재로드 parity 통과: 누락·추가·값 차이 0, 인코더 638 키 일치·저장, tokenizer 재로드 정상.

첫 step 의 top1_text 는 0.84 였다(76601 끝 0.985). SpecAugment light 의 시간 마스크(5 × 최대 5 %, 평균 약 12 % 프레임)가 그만큼 증거를 지운 결과로 본다. 스트리밍 모델에서는 증거 없는 방출을 배우게 할 수 있다. 그래서 첫 run 은 더 약한 시간 마스크를 권한다.

## 6. 다음 학습 제안 (제출은 사용자)

같은 스모크를 예산 16,384 로도 돌렸다. EXIT 0, 저장 검사 955/955 키 일치, 최대 **60.6 GB** 로 H200(141 GB) 에 여유가 있다. 그래서 76601 과 같은 예산을 권한다.

```bash
# 스케줄러 셸에서(/soundai/users/tskim/VAPKT)
bash slurm/submit-semcommit-train-apex.sh --run v036-asr-aug \
  --snap /soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637 \
  --init /soundai/Model/VAPASR/semcommit-v035-snap0929-d8/final --nodes 2 --time 24:00:00 \
  --delays 2,3,4,6,8 --batch-tokens 16384 --epochs 3 --lr 3e-5 --warmup 200 \
  --train-encoder --spec-augment 2,27,2,0.03 --speed-perturb 0.9,1.0,1.1 \
  --asr-list /soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list --asr-max-ratio 1.0 --varlen
```

- init 은 현재 최고 모델(v035-d8)에서 이어서다. LR 은 76601 의 절반인 3e-5 다. 인코더 LR 은 1e-5(E2 레시피)다.
- SpecAugment 는 약한 시간 마스크(2 × 최대 3 %, 평균 약 3 % 프레임)를 쓴다. light(평균 약 12 %)는 스모크에서 첫 step top1_text 를 0.985 → 0.84 로 떨어뜨렸다. 스트리밍 모델이 증거 없이 방출하는 법을 배울 위험이 있어 첫 run 은 약하게 간다.
- SEM 중립 비율 1.0: 언어별로 SEM 중립 항목이 라벨 항목을 넘지 않는다. 한국어는 E2 코퍼스 2.5 M 행 중 표본만 쓰고, 영어 806 h 는 대부분 쓴다.
- 새 옵션을 켜면 지문이 달라지므로 새 `--run` 이름이 필요하다.
- 평가는 끝난 뒤 두 가지로 한다. single-turn(Kspon + VoxPopuli-AA, E2·v035 와 같은 규약)과 골드 v1 commit 이다. 2-pass(1.7B iso)도 함께 본다.
- words-only 빌드(선택, 한국어 speechlm 3,585 파트): `sbatch slurm/semcommit-words-only.sbatch`. 한국어 SEM 중립은 이미 상한을 넘어, 다음 학습의 선행 조건은 아니다. 라벨링 재개와 한국어 다양성에 쓰인다.
- 보류: MNSC(싱가포르 영어, 7.4 만 스트림)는 mxc 정렬 캐시가 v1(`_items-online.json.gz`, src_offset_s 보존 수정 전)뿐이고 E2 에서도 쓰지 않아 넣지 않았다. v2 캐시를 만든 뒤 검토한다.

## 7. 평가 교체 — 영어 = VoxPopuli-Cleaned-AA (커밋 ac23dee)

- 데이터: HF `ArtificialAnalysis/VoxPopuli-Cleaned-AA` @07adf4a2. 628 발화, 1.98 h, 16 kHz float32, 292 세션. 로컬 `/Volumes/Samsung_T5/vapkt-corpora/voxpopuli-cleaned-aa` 에만 있다(서버 업로드는 사용자).
- 평가 도구 변경:
  - `eval_single_turn_asr.py prepare --voxpopuli-aa-root`
  - AA 비교용 점수(Whisper 정규화 + 숫자 분리, 발화 길이 가중 평균). AA 의 비공개 추가 규칙은 없어서 근사치다.
  - 오프라인 Qwen 기준선 스크립트를 더했다.
- 누수 점검(로컬):
  - 같은 발화·문단·정규화 문장 중복은 0 이다.
  - 세션은 292 개 중 280 개, 화자는 207 명 중 80 명이 학습 VoxPopuli 와 겹친다(학습 13,818 발화). 공식 분할이 화자 분리가 아니라서 공개 모델도 같은 조건이다.
  - 화자 분리 평가가 필요하면 그 80 명의 학습 발화를 빼는 목록을 만들 수 있다(미적용).

## 8. 결정이 필요한 것

1. **Codex 세션의 미커밋 short 풀 변경을 커밋할지.** 학습 76601 과 배포된 학습 코드가 이 변경에 의존한다. 그런데 저장소 HEAD 에는 없어서 재현성이 깨져 있다. 데이터셋·학습 스크립트 쪽 이번 변경(sem_free·속도 변형·인자)과 출력 형식 플래그도 같은 헝크라 커밋하지 못했다(mxc 에는 배포).
2. VoxPopuli-AA 업로드와 화자 겹침 처리(§7).
3. words-only 빌드 제출(§4)과 다음 학습 제출(§6).

## 9. 재현

- 오프라인 기준선: `results/qwen_offline_ref.py`, `results/nemotron_ref.py` → `single-turn-{qwen3-asr-*-offline,nemotron-rnnt}/`
- 비교 표: `results/ref_compare.py`, `results/st_bootstrap.py`
- 2-pass: `runs/semcommit/eval/v035-snap0929-d8/twopass/`
- 스모크: `runs/semcommit/smoke/asr-aug-20260929c/`
- SEM 중립 words: `data/semcommit-work/asr-words-v1/`(logs·lists)
