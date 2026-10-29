---
type: output
status: active
created: 2026-10-29
updated: 2026-10-29
summary: E11 — Nemotron 3 Diarization 고정 가지 구현·서버 검증(SDPA 로 컴파일 문제 해결), 진단·학습 제출 준비, 통합 설계와 G0 결과(이식 m = 8, 선형 DER 26.9 대 교사 22.1)
sources:
  - https://huggingface.co/nvidia/Nemotron-3-Diarization
  - '[[source-nemotron-3-5-transcribe-tech-report]]'
  - '[[output-nemotron35-transcribe-comparison-20261029]]'
  - '[[output-diarization-integration-20261025]]'
raw_authors:
  - tskim
---

# E11 — Nemotron 3 Diarization 고정 가지와 하나로 합치는 방법 (2026-10-29)

사용자 요청: "E9b 는 굳이 할 필요 없을거 같고. "Nemotron 3 Diarization 을 고정 가지"로 넣되 하나로 통합할 방법도 고안해놔."
- **E9b 는 제출하지 않는다**(사용자 결정). 코드와 `slurm/submit-q17-e9b.sh` 는 그대로 둔다.
- 배경: [[output-nemotron35-transcribe-comparison-20261029]] 2.1 절. 화자 분리 품질이 화자 귀속의 주 지렛대이고, NVIDIA 는 고정 화자 분리기를 한 체크포인트 안의 가지로 두어 한 모델이라고 부른다.

## 1. 자료 (HF 모델 카드 커밋 f667ed7, 2026-10-29 확인)
- **구조**: mel 128 → 특징 쌓기 ×8(80 ms) → Transformer 31 층(512, RoPE) → 머리(192) → 10 ms 출력이다. 화자 8 명을 도착 순서로 낸다. 파라미터는 99.2M 이다.
- **스트리밍**: AOSC(화자 캐시 264 프레임)와 FIFO 를 쓴다. 지연 설정은 0.32 s(청크 3 + 오른쪽 1)·0.64·1.04·30.4 s 다.
- **AMI SDM DER**(강제 정렬 참조, collar 0): 30.4 s 11.14 · 1.04 s 12.80 · 0.64 s 13.06 · 0.32 s 12.95 다. AMI MHM 은 9.25–9.62 다.
- **학습 데이터**: 실제 약 1 만 h + 모의 8.3 만 h 다.
  - **AMI train·dev 가 들어 있다.** 우리 AMI dev 평가는 그 모델에게 학습 데이터다. 비교는 eval(test)로 한다.
- 라이선스는 OpenMDW-1.1 이고, 런타임은 NeMo v3.0 이다.
  - HF transformers 경로는 5.18 dev 가 필요하다. 서버는 4.57 이라 쓰지 않는다.
- **ASR 통합 가이드**: NVIDIA 도 우리 인코더(nemotron-3.5-asr-streaming-0.6b)와 짝지어 쓰는 예를 둔다.
  - 거기서는 화자마다 따로 디코드하고, 화자 분리 청크를 ASR 청크에 맞춘다. 우리는 한 스트림이다.

## 2. 고정 가지 구현 (E11, 커밋 92dffd9)
### 가지 — `vapasr/hf/n3d.py`
- **로드**: 서버와 맥의 NeMo 는 같은 3.1.0(6989ae9fc)이다.
  - 저장 config 의 손실 대상(`bce_loss.BCEWithLogitsLoss`)과 보조 손실(activity·phantom)이 3.1.0 에 없다. BCELoss 로 바꾸고 뺀다. 추론에는 쓰지 않는다.
  - 가중치는 보조 `activity_head`(무음·한 명·겹침 3 분류)만 빼고 엄격히 올린다.
    - NeMo main 소스에서도 그 머리는 학습 손실용이고 화자 확률 계산과 무관하다.
    - 캐시 갱신·머리 계산을 main 과 비교했다. 차이는 CUDA graph·장치 처리뿐이다.
  - 생성자의 `random.seed(42)` 가 학습 난수를 건드리지 않게 상태를 되돌린다.
- **스트리밍**: NeMo `forward_streaming` 과 같은 고리다. 다른 점은 둘이다.
  - 분산 all_reduce 를 하지 않는다. 평가 rank 마다 길이가 달라도 collective 가 어긋나지 않게.
  - `process_signal` 대신 전처리기를 바로 부른다. 그 함수는 부를 때마다 `torch.cuda.empty_cache` 를 한다.
  - 10 ms 출력을 80 ms(우리 인코더 프레임)로 평균한다.
- **청크**:
  - 추론은 청크 4·오른쪽 0 이다. 우리 인코더 블록([56,3], 320 ms)과 같은 경계라, 블록 오디오가 다 오면 인코더 출력과 같은 때에 활동이 나온다. 지연을 더하지 않는다.
  - 학습은 청크 16 이다. 고리 수가 1/4 이다. 카드의 지연별 DER 차이가 작아(AMI SDM 0.32 s 12.95 · 1.04 s 12.80) 학습·추론의 활동 품질 차이도 작다.
- **활동**: 앞 4 채널(도착 순서 = 우리 슬롯 순서)을 임계 0.5 로 0/1 로 만든다. N3.5T 와 같은 임계다.
- **체크포인트 밖**: `N3DExpert` 는 nn.Module 이 아니라 state_dict·parameters·DDP 밖이다.
  - config `n3d_cfg` 에 경로·설정만 남고, 처음 쓸 때 그 장치로 읽는다. 환경 변수 `VAPASR_N3D_PATH` 가 경로를 덮는다.

### 모델 연결 — `modeling_vapasr.py`, `long_session.py`
- `add_n3d(path, chunk, rc, train_chunk, threshold)`: 슬롯 활동 입력(E5 계열)이 있는 모델에만 붙는다. E2 주입·두 스트림·E8 과는 함께 쓰지 않는다.
- `encode`: 스트림 1(헤드 예측 활동)을 전문가 활동으로 바꾼다. 스트림 0(ASR)·2(E10 화자 단서)는 그대로다.
  - 그래서 디코드 경로(배치·추측 prefill·회의 통째)가 바뀌지 않는다.
- `encode_long`(회의 통째):
  - 조각은 `n3d=False` 로 인코딩한다.
  - 전문가는 회의 전체 오디오로 한 번 돌린다. 화자 캐시가 회의 내내 이어진다.
  - AOSC·E7a 헤드와는 함께 쓰지 않는다.
- **학습 forward**:
  - 표본마다 확률 ρ 로 전문가 활동을 쓴다. 아니면 흔든 정답이다(E6 와 같은 ρ 경사).
  - 전문가 채널 순서는 늘 정답 슬롯 자리로 맞춘다(`align_slots`). 순열이 어긋난 입력이 라벨과 부딪히지 않게 하기 위해서다. 추론은 채널 k = 슬롯 k 이고, cpWER 은 순열에 불변이다.
  - 헤드(E1c)와 그 손실은 그대로 둔다. E10 의 탭 임베딩 출처다.

### 학습·평가·도구
- **학습**: `stage3_train.py --n3d <.nemo> [--n3d-train-chunk 16 --n3d-chunk 4 --n3d-threshold 0.5]`, 지문 키 `n3d`.
- **평가**: `cst_stage3_eval.py --n3d <.nemo> [--n3d-chunk]` 다.
  - E9·E10 체크포인트에 붙이면 학습 없는 진단이다.
  - E11 체크포인트는 config 의 경로를 쓴다.
  - `v3_eval_task.sh` 에서는 `N3D=<.nemo>` 로 주고, 산출 이름에 `-n3d` 가 붙는다.
- **DER**: `diar_eval.py run --n3d <.nemo> [--session]` 이다.
  - 같은 창·회의·참조·마스크·행 꼴이라 `summarize` 를 그대로 쓴다.
  - 행에 `extra`(채널 4–7 에만 있는 활성 프레임 — 우리 슬롯 밖으로 간 말)와 `n_hyp_spk8` 를 더한다.
- **제출**:
  - `slurm/submit-q17-e11.sh`: E9 final 에서 시작한다. `--from-e10` 이면 E10 final 에서 시작하고 이름은 q17-e11b 다.
  - 평가 묶음: `q17-e9-n3d`(의존 없음), `q17-e10-n3d`, `q17-e11`·`q17-e11b`.

### 검증
- **맥 CPU**(AMI ES2002a Array1 120 s, 실제 .nemo):
  - `stream_probs`(청크 4) 와 NeMo 자체 forward 의 최대 차는 0 이다.
  - 길이가 다른 배치 두 줄이 단독 실행과 같다(최대 차 0 · 1e-16, 짧은 줄 뒤는 0).
  - 청크 4 대 공식 0.32 s(3,1) 설정의 이진 일치는 99.2 % 다.
  - CPU fp32 실시간 배율: 청크 4 0.81 배, 청크 16 0.21 배.
- **맥 시험**:
  - `tests/test_n3d.py` 5 개가 통과했다. 가짜 전문가 4 개 + 실제 .nemo 1 개(`VAPASR_N3D_PATH`)다.
  - 관련 144 개(E9·E10·회의 통째·화자 분리)가 통과했다.
  - 전체로는 814 개가 통과했다. 실패 5 개와 수집 오류 2 개는 이번 변경과 무관하다(로컬에 rapidfuzz 없음, 단일 발화 평가의 기존 시험·코드 불일치, tests 패키지 import).
- **서버**: 코드를 반영했고 md5 를 전수 대조했다(실행 파일 같음). 서버 시험(n3d·E9·E10·회의 통째·화자 분리) 48 개가 통과했다.
- **업로드 경로**: 사용자 첫 업로드(`VAPKT-data/models/…`)는 403 AuthorizationPermissionMismatch 였다.
  - `models` 는 775(sa_tskim)이고, azcopy 주체는 NFS 에서 uid 65533(other)이다.
  - xcomet 과 같은 `VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/` 로 올렸다. 서버 대조: 크기·sha256 이 맥 원본·HF 와 같다.
- **서버 GPU 스모크**(호스트 GPU 5, E9 final + 전문가, AMI 학습 창 3 개, `tmp/e11-check/n3d_smoke*.log`)
  - 첫 판은 NeMo 그대로였다. 전문가 첫 forward 가 617 s, 26 s 창 활동이 55 s 걸렸다.
    - 원인: NeMo TransformerEncoder 가 CUDA 에서 `torch.compile(flex_attention, dynamic=True)` 를 쓰는데, 스트리밍처럼 길이가 바뀌면 블록 수마다 다시 컴파일한다(다시 컴파일 한도 8 을 넘었다).
  - 고친 것(커밋 93c18c7·85e546d):
    - `use_sdpa`: N3D 는 attn_mode full + RoPE 라 마스크가 키 패딩(kv < 길이)뿐이다. SDPA 키 패딩 마스크와 같은 계산이다(맥 시험: 스트리밍 확률·오프라인 층 출력이 1e-4 안).
    - `encoder.sync_max_audio_length` 를 끈다. 켜 두면 forward 마다 분산 all_reduce 를 해서, rank 마다 고리 수가 다른 학습·평가에서 collective 가 어긋난다.
    - 창 평가는 배치의 창을 한 번에 돌린다. 줄마다 단독과 같은 값이다.
  - 고친 뒤 결과:

| 항목 | 값 |
|---|---|
| 학습 fwd+bwd(창 3 개, 학습 청크 16) | 전문가 끔 0.86 s · 켬 1.81–1.91 s(약 +1 s), 메모리 39.0 GB 그대로, 손실 정상 |
| 스트리밍 = NeMo forward | 최대 차 0(fp32) |
| 청크 4 대 16 활동 일치 | 0.9954 |
| 추론 속도(청크 4, bf16, B = 1) | 300 s 오디오 41.7 s(실시간 0.14 배) — 1 시간 회의 약 8 분 |
| 디코드 | 스트림 1 = 전문가 활동, 슬롯 블록 디코드 정상(94 토큰, forced 0) |

  - 학습 비용: step 이 약 +1 s 다. E9 레시피(창 약 5 개/GPU)면 5,000 step 에 약 1.5 h 가 는다.
- **서버 평가 스모크**(호스트 GPU 6, `tmp/e11-check/eval-e9-n3d`·`der-n3d`)
  - E9 final + `--n3d`, AMI SDM 창 16 개 δ10: 직렬(추측 prefill) 대 배치 디코드 대조 `PARITY PASS`, 16 행.
  - `diar_eval run --n3d`, 창 32 개: 행·요약이 나온다.
  - 표본이 작아 수치는 쓰지 않는다.
- **제출 dry-run**(mxc): `submit-q17-e11.sh` 는 apex 2 노드·5,000 step·저장 500·평가 q17-e11 이다. `submit-evals.sh q17-e9-n3d` 는 32 작업(GPU 마다 창 DER → 창 cpWER → 회의 DER → 회의 cpWER)이다.

## 3. 실험 계획
| 이름 | 무엇 | 비교 | 판단 |
|---|---|---|---|
| 진단 `q17-e9-n3d` | 전문가 DER(창·회의, 우리 참조) + E9 final 의 활동만 전문가로(학습 없음, 창·회의 통째 δ10) | E9 헤드: 창 DER SDM eval 27, 회의 40 · E9 회의 통째 cp 53.47 / 66.35 | 활동 출처만 바꿔 귀속이 얼마나 주나 |
| E11 | E9 final → 전문가 활동으로 5,000 step(E9 레시피) | E10(같은 시작 — 주입 꼴) | 2×2: E9 → E10(꼴), E9 → E11(출처), E11b(둘 다) |
| E11b | E10 final → 전문가 활동 | E10·E11 | E10 결과를 본 뒤 |

- 외부 기준(회의 통째, IHM-mix / SDM): N3.5T 오프라인 10.29 / 17.34, Parakeet(SSA-v2) + N3 Diarization 13.38 / 20.33, VibeVoice-Streaming 7B 27.48 / 39.01 · 1.5B 45.15 / 56.83.
- DER 비교는 eval 로 한다. 전문가에게 AMI dev 는 학습 데이터다.

## 4. 하나로 합치는 방법 (설계)
**목표**: 배포 모델의 음향 인코더를 우리 ASR 인코더 하나로 줄이면서, 전문가의 화자 분리 품질(AMI SDM DER 13 근처)을 지킨다.
- 지금 E11 은 체크포인트 하나에 경로로 묶인 고정 가지다. N3.5T 의 Parallel Expert Encoder 와 같은 꼴이고, 인코더가 둘이다(평행 — 캐스케이드는 아니다).

### 4.1 길 셋
| 길 | 무엇 | 장점 | 위험 |
|---|---|---|---|
| U1 증류 | 우리 인코더 탭 위 헤드(E1c 꼴 12 층)를 전문가 출력(80 ms 확률)으로 학습 | 단순하다. diar_train·AOSC 코드를 그대로 쓴다 | 헤드 용량. 전문가는 9.3 만 h 로 배웠다 |
| **U2 이식 + 증류(권장)** | 전문가의 m 층 위 + 머리를 우리 인코더 위로 옮기고, 탭 → h_m 어댑터만 학습(교사 = 전문가) | 전문가가 배운 화자 모델링(위층)을 그대로 쓴다. 출력 꼴·AOSC 도 그대로다 | 우리 탭에 화자 정보가 모자라면 m 을 키울 수 없다 |
| U3 공동 미세조정 | 인코더를 풀어 ASR + 화자 분리 증류를 함께 | 상한이 가장 높다 | ASR 회귀, 비용 |

- 두 인코더의 프레임 격자가 같다(80 ms, mel 프레임 8 개). 그래서 우리 프레임 k 와 전문가 프레임 k 가 바로 대응한다.

### 4.2 U2 단계
- **G0 — 이식 층 고르기**(구현 끝: `experiments/n3d_graft_probe.py`)
  - dev 창에서 우리 탭(E9 의 diar_taps + 마지막 층)으로 전문가 층 입력 h_m(m = 0 … 31)을 능형 회귀한다.
  - eval 창에서 층마다 R² 를 잰다. `--graft-layers` 의 m 마다 ĥ_m 으로 전문가 m 층 위·머리를 돌린 DER(우리 참조)과 교사 일치를 잰다. 교사 DER 도 함께 잰다.
  - 판단:
    - 선형 이식 DER 이 교사 + 3 안인 가장 큰 m 을 고른다. 학습한 어댑터는 선형보다 낫다.
    - 모든 m 에서 크게 나쁘면 U1 으로 간다.
  - **결과(2026-10-29, 호스트 GPU 6, AMI SDM 20–40 s 창 — dev 300 창 맞춤 · eval 300 창 잼, 오프라인, 우리 참조, 166 s)**
    - 산출: `VAPKT-data/results/n3d-graft-probe-sdm/probe.json`
    - 탭은 E9 diar_taps(1, 3 … 21) + 마지막 층, 12,288 차원이다.

| 이식 m | R²(h_m) | DER | 놓침 | 헛 | 혼동 | 교사와 프레임 일치 |
|---:|---:|---:|---:|---:|---:|---:|
| 교사(전문가 그대로) | – | **22.10** | 18.66 | 2.38 | **1.06** | 1.0 |
| 0 | 0.16 | 58.06 | 51.31 | 0.71 | 6.04 | 0.60 |
| 4 | 0.28 | 29.68 | 22.22 | 2.32 | 5.14 | 0.83 |
| **8** | 0.33 | **26.88** | 20.03 | 2.74 | **4.12** | **0.84** |
| 12 | 0.35 | 28.40 | 20.74 | 2.43 | 5.22 | 0.84 |
| 16 | 0.38 | 32.21 | 21.13 | 2.18 | 8.90 | 0.79 |
| 20 | 0.36 | 35.08 | 24.03 | 1.89 | 9.15 | 0.77 |
| 24 | 0.30 | 34.65 | 23.90 | 1.86 | 8.88 | 0.77 |
| 28 | 0.37 | 42.78 | 31.70 | 1.78 | 9.30 | 0.71 |

    - **교사의 이점은 거의 혼동이다**(1.06 %).
      - 우리 참조에서 교사 DER 22.1 의 대부분은 놓침(18.7)이다. 우리 참조(전사 구간)는 구간 안 쉼도 말로 센다. 카드가 강제 정렬 참조를 쓰는 까닭과 같다(카드 오프라인 SDM 11.14).
      - 우리 헤드도 놓침이 크다. E6 헤드는 SDM eval 창(20–90 s)에서 DER 27.15 중 놓침 17.74 다. 차이는 헛·혼동(헤드 9.4 대 교사 3.4)이다. 창 범위가 달라 어림이다.
    - **선형 이식은 m = 8 이 가장 낫다**(DER 26.88, 교사 + 4.8, 혼동 4.1).
      - 아래 층(m ≤ 4)은 mel 수준 표현이라 우리 ASR 탭으로 선형 복원이 안 된다(R² 0.16–0.28).
      - 위 층(m ≥ 16)은 화자 정체 표현이다. ASR 인코더가 화자 차이를 지운 만큼 혼동이 9 % 로 오른다.
    - 판단: 교사 + 3 안의 m 은 없다. 다만 선형(R² 0.33)으로도 + 4.8 이고 혼동은 우리 헤드보다 작다. → **G1 은 m = 8 로 간다**(전문가 8–30 층 23 개 약 72M 이 화자 분리 머리가 된다).
      - 학습한 비선형 어댑터(시간 문맥 포함)로 교사 + 1.5 를 노린다.
      - 혼동이 4 % 대에 머물면 U1·U3(인코더 일부를 풀기)를 다시 본다.
- **G1 — 어댑터 학습**
  - 학생 = 우리 탭 → 어댑터(LN·Linear + 인과 conv 블록 2–4 개) → 전문가 m 층 위(고정) → 머리(고정).
  - 손실 = 교사 확률 BCE(80 ms) + h_m MSE(작은 가중).
  - 데이터: 학습 대화 전부 + 모의 회의. 교사가 라벨을 주므로 전사 없는 오디오도 쓴다.
  - 오프라인 창(30 s)에서 먼저 학습한다. 그다음 AOSC 스트리밍으로 미세조정한다. 캐시가 ĥ_m 을 담는다. 전문가가 오프라인 → 스트리밍 두 단계로 학습한 것과 같은 순서다.
- **G2 — 선택**: 위층을 LoRA 로 풀고 정답 라벨(Sort·PIT)을 섞는다.
- **G3 — 바꿔 끼우기**
  - E11 의 전문가를 학생으로 바꿔 평가한다(학습 없음). 활동 꼴이 같아 그대로 끼울 수 있다.
  - 차이가 크면 E11 을 학생 활동으로 짧게 미세조정한다.
- **판단 기준**:
  - 학생 DER 이 AMI SDM eval(우리 참조)에서 교사 + 1.5 안이어야 한다.
  - E11 cpWER 차이가 회의 bootstrap 으로 유의하지 않아야 한다.
- **계산**:
  - mel·특징 쌓기·아래 m 층이 빠진다. 대신 우리 인코더 출력을 쓴다.
  - 위 (31 − m) 층은 청크마다 [캐시 264 + FIFO 264 + 청크 4] 위를 돈다. 전문가와 같은 꼴이다.
- **방침**: 통합 뒤 꼴은 인코더 하나 + 전문가에서 옮긴 화자 분리 머리다. 사용자 방침("외부 모듈 없이 하나의 E2E 모델")에 맞는다.

## 5. 사용자 할 일
1. **.nemo 업로드**: 맥 T5 `/Volumes/Samsung_T5/vapkt-models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo`(198,676,480 B, sha256 `867c53f5…` — HF LFS 와 같다)
   → 서버 `/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/`(xcomet 과 같은 곳).
   - 2026-10-29 첫 시도(`VAPKT-data/models/…`)는 403 AuthorizationPermissionMismatch 였다. `models` 폴더는 775(sa_tskim)이고, azcopy 주체는 NFS 에서 uid 65533(other)이라 쓸 수 없다.
   - 777 폴더(`VAPKT-data`·`data`·`data/audio`)나 azcopy 가 만든 폴더(`vapkt-baselines`, 소유 65533) 아래는 된다.
2. **진단**: `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-evals.sh q17-e9-n3d)`(선점 파티션이면 `--partition low_p_hpc`)
3. **E11**: `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-e11.sh)`(`--partition low_p_hpc` 가능)
4. **G0 이식 층 고르기**: 업로드 뒤 호스트 GPU 한 장으로 돌린다(약 10–20 분, 읽기만).
