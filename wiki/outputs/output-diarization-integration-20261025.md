---
type: output
status: active
created: 2026-10-25
updated: 2026-10-28
summary: 화자 분리 통합 — 설계, E0–E8 결과(회의 통째 귀속 붕괴·태그 이끌기·E7b·헤드 지연), E9 슬롯 블록 형식(청크마다 모든 슬롯 태그 입력·<BEGIN>) 구현
sources:
  - https://arxiv.org/abs/2507.18446
  - https://huggingface.co/nvidia/Nemotron-3-Diarization
  - https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1
  - https://arxiv.org/abs/2506.22646
  - https://arxiv.org/abs/2605.15442
  - https://arxiv.org/html/2510.03723v2
---

# 화자 분리(diarization) 통합 검토 (2026-10-25)

사용자 요청 흐름:
- "ami sdm 의 수치를 20 미만으로 내리고 싶은데" → [[2026-10-25-ami-sdm-analysis]](raw: `raw/sources/experiments/2026-10-25-ami-sdm-analysis/README.md`)
- "우리도 Diarization 기능을 학습해 추가하는 건 어떨까?"
- "2605.15442 · 2507.18446 · nemotron 3 diarization 논문을 참고 해봐"

전제는 사용자 방침인 외부 모듈 없는 단일 E2E 모델이다. 2026-10-25 에 원문·모델 카드를 읽고 정리했다.

## 1. 읽은 자료 요약
- **Streaming Sortformer**(arXiv:2507.18446, Interspeech 2025, NVIDIA)
  - 구조: NEST(Fast-Conformer) pre-encoder(80 ms 프레임) 뒤에 Transformer 를 쌓고, 화자 4 명을 sigmoid 로 출력한다(117M).
  - 출력 순서는 도착 순서다. 손실은 Sort loss + PIT 다.
  - 스트리밍은 도착 순서 화자 캐시(AOSC)와 FIFO 로 한다.
    - 캐시는 화자별 프레임 임베딩 188 프레임(15 s)이다.
    - 점수는 S_i = log P_i + Σ_{j≠i} log(1 − P_j) 로, 혼자 말하는 프레임을 남긴다.
    - 무음 임베딩, 최근 가산, 화자별 최소 몫 가산이 있다.
  - 입력은 [캐시, 현재 청크]다. 오프라인 체크포인트를 캐시와 함께 미세조정해야 한다(안 하면 DIHARD 27.6).
  - DER: DIHARD III 19.0–20.3(지연 0.32–10 s), CALLHOME 10.1–13.7. 학습 데이터는 모의 5,150 h + 실제 2,030 h(AMI SDM 포함)다.
- **Nemotron 3 Diarization**(HF 모델 카드, 2026-09-23 공개, OpenMDW)
  - 100M, RoPE Transformer 31 층, 80 ms, 화자 8 명, AOSC(264 프레임)다.
  - 지연은 30.4 · 1.04 · 0.64 · 0.32 s 를 고를 수 있다.
  - 학습: 실제 약 10k h + FastMSS 모의 82.6k h(단일 화자 원천 약 28k h). 오프라인(모의) → 스트리밍 미세조정(실제 + 모의) 두 단계다.
  - **AMI SDM DER 11.1(30.4 s) · 12.8(1.04 s) · 13.0(0.32 s)**(강제 정렬 참조)다. 앞 버전 streaming sortformer v2.1 은 21.4–23.9 였다.
- **Multitalker Parakeet Streaming 0.6B**(HF 모델 카드, Nemotron-Speech-Streaming 기반 — 우리 인코더 계열)
  - 화자 분리 출력만 받는다(등록 음성 없음).
  - 화자 활동으로 만든 커널을 Fast-Conformer pre-encode 층에 넣고, **화자마다 인스턴스 하나**(배치)로 그 화자의 전사를 낸다.
  - cpWER(Streaming Sortformer v2, 1.12 s): **AMI IHM 21.3 · AMI SDM 37.4** · CH109 15.8 · Mixer 6 23.8.
  - 단일 화자 WER 은 7.44(기본 모델 7.16)다.
- **SSA — Self-Speaker Adaptation**(arXiv:2506.22646, Interspeech 2025)
  - 주입: X_inj = f_inj(X ⊙ y) + X. y 는 대상 화자 활동(0–1), f_inj 는 선형 2 층이고, 위치는 pre-encode 층이다(1.1M).
  - 학습: 대상 화자를 무작위로 골라 그 화자 전사만 정답으로 한다. 화자 분리기는 고정한다.
  - LibriSpeechMix 오프라인 cpWER 2.2 / 2.8 / 5.0(1·2·3-mix), 스트리밍 1.12 s 4.0 / 5.4 / 9.8, CH109 26.2 다.
- **Mind the Gap — FastMSS**(arXiv:2605.15442, Interspeech 2026 제출)
  - 모의 대화 생성기다: N 화자 HMM 교대(유지·교대·끼어듦·맞장구), RIR·잡음·이득.
  - 겹침을 늘리면 다화자 ASR 은 좋아지고 화자 분리는 나빠진다. 작업마다 따로 맞춰야 한다.
  - 원천 다양성이 도메인 일치보다 중요하다.
  - 잡음 + 잔향이 화자 분리에 크다(AliMeeting 원거리 DER 36.8 → 25.7).
  - 모의 → 실제 미세조정이 가장 좋다(DiCoW tcpWER 8.7, Sortformer DER 15.5).
- 참고 상한: DiCoW 는 AMI-SDM cpWER 16.3 이다(arXiv:2510.03723 — 정답 화자 구간 입력, 오프라인, Whisper 계열).

## 2. 우리 모델과의 대응
- 우리 슬롯 규칙(창 안에서 먼저 말한 사람 = A)이 Sortformer 의 도착 순서와 같다. 출력 형식을 바꾸지 않고 화자 분리 출력을 슬롯에 그대로 대응시킬 수 있다.
- 우리 AMI SDM 오류(1.7B 회의 + 4 슬롯 avg3, eval δ10 창 cpWER 53.5)와 대응:
  - 귀속 24.9 pt: 홀로 구간 긴 턴 통째 오배정 11.6 pt, 4 인 창 단어 75 % 를 2 슬롯으로. → "누가 누구인지" 문제. 화자 분리(캐시로 정체 유지)가 바로 겨누는 곳이다.
  - 받아쓰기 28.6: 겹침 구간 단어 21 % 의 S+D 57 %(오류의 47 %). → 한 스트림에 번갈아 쓰는 형식의 약점이다. SSA 처럼 화자마다 따로 디코드하면 겹침을 각자 받아쓴다.
  - 회의 통째 붕괴(V3 README 28 절): LLM 문맥 창 밖으로 나간 화자를 다시 알아보지 못한다. AOSC 가 정체를 들고 다니면 LLM 문맥은 최근 몇십 초로 둬도 된다.
- 기존 자산:
  - Stage 3 화자 슬롯 활동 헤드(`spk_head`, LLM 오디오 위치, 보조 손실 0.5)는 창 안 활동만 맞춘다. 캐시도 정체 손실도 없다.
  - 활동 라벨(`slot_activity`)은 대화·회의·합성 셋에 이미 있다.

## 3. 통합 설계 (한 모델 그래프)
1. **화자 분리 가지**:
   - 우리 Nemotron 인코더의 pre-encode 출력(80 ms)에 Transformer 를 쌓아 4–8 슬롯 활동을 낸다(Sort loss + PIT). 스트리밍은 AOSC + FIFO 다.
   - 같은 그래프 안의 가지라서 외부 모듈이 아니다.
   - 공개 가중치(Nemotron 3 Diarization, OpenMDW)로 초기화할 수 있다. 다만 입력 특징이 달라서 가지를 mel 에서 따로 시작해야 한다. 이건 사용자가 정할 일이다.
2. **주입(SSA)**: 인코더 pre-encode 출력에 f_inj(X ⊙ y_s) + X 를 넣는다(대상 슬롯 s 의 활동). 이 경로만 슬롯마다 다르다.
3. **디코드 형식**:
   - (a) 슬롯마다 따로(대상 화자 받아쓰기·번역 — Stage 2 단일 화자 형식을 그대로 쓴다): 겹침에 강하다. 귀속은 화자 분리가 정한다. 비용은 슬롯 수 배(배치)다.
     - 실시간 경로가 한 스트림 RTF 0.15 라 4 슬롯이면 약 0.6 으로 어림한다(재지 않음).
   - (b) 한 스트림 + 모든 슬롯 활동 주입: 싸지만 겹침은 여전히 번갈아 쓴다.
   - 분석상 겹침이 받아쓰기 오류의 절반이라 (a)가 맞다. 다만 (a)는 대화 단계 학습 형식이 바뀐다.
4. **긴 회의**: AOSC 로 정체를 유지하고, LLM 은 슬롯마다 최근 문맥만 본다. 회의 통째 평가 `--window full --context-s` 와 맞물린다.
5. **데이터**: FastMSS 식 모의를 쓴다.
   - 화자 분리용은 자연 교대 + 잔향·잡음(원거리)이고, 받아쓰기용은 겹침을 늘린다.
   - 원천은 우리 단일 화자 코퍼스(한·영·독)와 AMI 근접 마이크 등으로 다양하게 잡는다.
   - 그 뒤 실제 회의(AMI·ICSI·NOTSOFAR 학습 분할)로 미세조정한다.

## 4. 단계별 시험 (싼 것부터, 판가름 기준 포함)
- E0 분석(학습 없음): 지금 가설 단어를 정답 화자 구간으로 시각 기준 다시 귀속해 cp 를 잰다. 한 스트림 형식에서 화자 분리가 완벽하면 남는 귀속 오류(겹침 모호성)를 어림한다.
- E1 화자 분리 가지: 우리 인코더 위 Sortformer 가지를 0.6B 로 학습해 AMI SDM DER 을 잰다. 기준은 Nemotron 3 Diarization 의 1.04 s 12.8 이다.
- E2 상한: 정답 활동 주입 + 슬롯별 디코드(형식 (a))를 0.6B 로 학습해 AMI SDM cp 를 잰다. DiCoW 오라클 16.3 과 같은 조건이다.
- E3 실제: E1 예측 활동으로 바꾸고, 스트리밍 AOSC 와 회의 통째 평가로 잰다. 공개 기준은 Multitalker Parakeet SDM 37.4(세션 cpWER, 실제 화자 분리, 1.12 s)다.
- E4 데이터: 모의 대화(3–8 인, 잔향)로 화자 분리·받아쓰기를 함께 개선한다.

## 5. 목표 수치에 대한 판단
- 공개된 AMI SDM cpWER 20 미만은 정답 화자 구간을 받은 오프라인 시스템(DiCoW 16.3, SA-DiCoW 18.1)뿐이다.
- 실제 화자 분리 + 스트리밍 공개 수치는 37.4(Multitalker Parakeet 0.6B)다.
- 우리 창 cpWER 는 창마다 짝을 새로 정해 세션 cpWER 보다 쉽다. 수치를 견줄 때 같은 평가(세션, meeteval 정규화)로 맞춰야 한다.

## 6. 결과와 구조 요건 검토 (2026-10-26)
결과(1.7B 회의 + 4 슬롯 final 출발, AMI δ10 창 cpWER, eval):

| 방식 | IHM-mix | SDM |
|---|---|---|
| 한 스트림 | 49.91 | 53.62 |
| E0 정답 구간을 나중에 붙임 | 34.72 | 38.98 |
| E3 예측 활동(E1, 임계 0.45) + 화자별 디코드 | 25.35 | 34.50 |
| E2 정답 활동 + 화자별 디코드 | 21.20 | 28.56 |

- E1 의 AMI 창 DER(최적 짝, 임계 0.45)은 eval IHM-mix 18.67, SDM 23.03 이다. 상세는 `raw/sources/experiments/2026-10-25-ami-sdm-analysis/README.md` 에 있다.

사용자 지적: "E1~3 은 성능적으로 더 올릴 수 있는 가능성을 보여주지만 우리 프로젝트의 모델 구조 요건을 충족 못 시키는거 같은데".
- 맞다. 3 절의 형식 (a)(슬롯마다 따로 디코드)는 다음 요건을 어긴다.
  - 출력이 화자별로 갈라져, 두 사람을 한 스트림에서 함께 보고 번역·턴을 정하는 구조가 아니다.
  - 화자 수만큼 디코드해 계산이 늘어난다.
  - E1 가지는 비인과이고, E3 은 인코더 두 벌을 쓰는 캐스케이드다.
  - 계획 단계에서는 학습 형식이 바뀐다는 점만 적고 요건 충돌을 짚지 못했다.
- 진단으로 얻은 것:
  - E2 의 이득은 전부 귀속에서 나왔다(E2 cp 28.56 ≈ 한 스트림 agn 28.31). 화자별 디코드 구조가 받아쓰기를 좋게 하지는 않았다.
  - E0 의 39 는 디코드가 끝난 뒤 붙인 한계다. 디코드하는 동안 화자 정보를 쓰는 한 스트림의 한계가 아니다.
- 요건을 지키는 대안(3 절 형식 (b)의 구체화, 사용자 결정 대기):
  - 인과 화자 분리 헤드를 받아쓰기 인코더 위에(같은 가중치) 둔다.
  - 같은 forward 안에서 슬롯 활동을 LLM 입력(오디오 soft token + Σ p_s·e_s)에 섞는다.
  - 출력은 지금의 한 스트림(슬롯 태그 + 번역 + EOT)이다.
  - 시험 순서는 E5(정답 활동 조건, 형식 상한) → E6(헤드 예측과 함께 학습) → E7(AOSC, 회의 통째·실시간)이다.
  - 구성과 단계를 그림으로 정리한 페이지가 있다(2026-10-26, 사용자 "통합 계획이 어떻게되고 모델이 어떻게 구성할건지 그림과 함께 설명해줘"): https://claude.ai/artifact/TRLYS96Xn8axTepaFv56WJ
    - 가지에서 E1 을 인과로 바꿔 받아쓰기 인코더 위에 둔다. 슬롯 임베딩(4 × 2048, 0 초기화)을 오디오 토큰에 더한다. LLM 쪽 보조 화자 헤드는 끈다.
    - E5 판단: SDM eval 이 E2(28.6) 쪽이면 E6 으로 간다. E0(39.0) 쪽이면 E5b(켜진 슬롯마다 활동 주입 인코더 토큰, 출력은 한 스트림)를 먼저 한다. E4 데이터는 나란히 간다.
  - **E5 결과(2026-10-27, q17-e5 — F1 레시피 + 슬롯 활동 입력, 평가 95934 amiact)**
    - AMI 창 δ10 cp 는 SDM eval 29.24, IHM-mix eval 23.39 다(F1 final 53.94 · 49.43 대비 −24.7 · −26.0, 회의 bootstrap 유의).
    - 귀속 몫이 1.0–2.1 로 거의 사라졌다. 남은 것은 받아쓰기(agn 22.4–27.8)다.
    - E2(28.6) 수준이라 판단 기준을 통과했다. 한 스트림 + 슬롯 활동 입력이 화자별 디코드 상한에 닿는다.
    - 다음 관건은 E6(인과 헤드 예측 활동)이 이 상한을 얼마나 지키느냐다.
  - **E1c(인과 가지, 2026-10-27 — `raw/sources/experiments/2026-10-26-data-expansion/e1c_eval.sh`)**
    - 같은 AMI 창에서 최적 짝 DER 은 eval IHM-mix 23.08 · SDM 27.04 다(dev 최적 임계 0.45 — E6 의 값과 같다).
    - E1(양방향) 18.67 · 23.03 보다 약 4 pt 높다. 대부분 놓침이다(SDM eval miss 15.4 대 E1 0.5 임계 15.9 → 18.4) — 미래를 못 보니 말 시작을 늦게 잡는다.
    - E6 헤드는 이 가지에서 시작해 함께 학습한다(학습 중 프레임 정확도 0.89 → 0.94).

## 7. E7 — 회의 통째·실시간 화자 유지(AOSC) 구현 (2026-10-27)
사용자: "E7 을 먼저 구현해놓자".

### 문제
- 지금 헤드와 LLM 은 창마다 "창에서 먼저 말한 사람 = A" 로 슬롯을 정한다.
- 회의를 통째로 디코드하면 인코딩 조각마다 슬롯이 새로 정해져, 같은 사람이 조각마다 다른 슬롯이 된다.
- 예전 V2 의 회의 통째 평가에서 귀속이 무너진 이유가 이것이다.

### 방법(Streaming Sortformer AOSC, arXiv:2507.18446)
- **입력:** 헤드 입력 임베딩(탭 가중합 → LN → 투영, 512 차원)을 [화자 캐시 · FIFO · 새 청크] 로 이어 인과 헤드에 넣고, 청크 몫 출력만 쓴다.
- **캐시 순서:** 화자(도착 순서)마다 대표 프레임을 묶어 둔다. 캐시 앞 화자가 늘 앞 슬롯이 되어 회의 내내 같은 슬롯을 유지한다.
- **캐시 갱신:** FIFO 가 넘치면 가장 오래된 프레임과 지금 캐시를 후보로 고른다.
  - 점수: S_i = log P_i + Σ_{j≠i} log(1−P_j), P_i < 0.5 이면 −∞.
  - 가산점: 새 프레임 +0.05, 화자마다 상위 33·66 개 +2 ln 2·+ln 2.
  - 무음: 무음 평균 임베딩을 화자마다 3 칸 둔다.
  - 크기: 상위 188 칸.
- **설정:** 캐시 188, FIFO 188, 갱신 94 프레임, 추론 청크 6(0.48 s) — 헤드가 인과라 청크 크기는 출력에 영향이 없고 갱신 시점만 바꾼다.
- **학습(E7a):**
  - 원 논문처럼 오프라인 헤드(E6)를 스트리밍 흉내로 미세조정한다.
  - 긴 클립(90–120 s)을 청크 47 로 흘린다.
  - 캐시는 클립마다 반은 정답으로(같은 점수는 Gumbel 0.5 로 흩뜨림), 반은 그 pass 의 모델 예측으로 고른다(`--aosc-self 0.5`).
    - NeMo 의 스트리밍 학습은 모델 예측 캐시라 추론과 규칙이 같다. 정답 캐시만 쓰면 추론에서 만나는 오류 섞인 캐시를 학습에서 보지 못한다.
    - 모델 예측 캐시에서는 화자 묶음마다 정답 화자를 겹침이 가장 큰 일대일 짝으로 정하고, 정답과 겹침이 없는 묶음의 슬롯 정답은 0 이다.
  - 캐시 화자 순서를 0.5 확률로 섞는다(정답 슬롯도 같이).
  - 점수 세부는 NeMo `sortformer_modules` 와 맞췄다(2026-10-27 원본 코드 대조).
    - 겹침·불확실 프레임 제외: '그 화자만' 프레임(점수 > log 0.5)이 22 개 이상인 화자는 나머지 프레임을 캐시 후보에서 뺀다.
    - log(1 − P_j) 하한은 0.25 이다.
    - 무음은 화자 확률 합 < 0.2 인 프레임이다.
  - 원 논문은 미세조정 없이 캐시만 붙이면 DER 27.6, 미세조정하면 19.9 였다.
- **LLM 쪽(E7b, 필요하면):**
  - `--global-slots P` 는 학습 창을 확률 P 로 대화 전체 도착 순서 슬롯으로 둔다. 회의 통째 디코드의 문맥 창에서는 첫 화자가 A 가 아닐 수 있기 때문이다.
  - 헤드가 있으면 `diar_align` 이 켜진다. 헤드 손실은 창 도착 순서로 정렬한 정답에 걸고, 슬롯 입력으로 주는 헤드 예측은 정답 자리로 맞춘다(AOSC 흉내).

### 코드(시험: 맥 753 통과 — 실패 15 는 로컬 환경)
- `vapasr/hf/diar.py`
  - DiarBranch.embed/head, AoscConfig, aosc_select, aosc_stream(추론), aosc_clip_loss(학습)
  - forward(aosc=…)(DDP 경로), arrival_sort, align_slots
- `experiments/diar_train.py --aosc`, `experiments/diar_extract.py`(E6 헤드 → diar.pt), `experiments/diar_eval.py run --session --session-modes aosc,resetS`(회의 단위 DER)
- `vapasr/hf/long_session.py`(segment_plan, encode_long(aosc=…))
- `experiments/cst_stage3_eval.py --aosc --diar-head-override`
- `v3_eval_task.sh` amifull(AOSC·DIAR), `slurm/submit-evals.sh`
  - `q17-e6-amifull`: E6 를 조각마다 새로 대 AOSC 로
  - `q17-e7-amifull`: E7a 헤드로
- `vapasr/data/cst_dialogue.py`(global_slots), `experiments/stage3_train.py --global-slots`, `modeling_vapasr.py`(diar_align)
- 제출·평가: `slurm/submit-q17-e7a.sh`(1 노드), `raw/sources/experiments/2026-10-26-data-expansion/e7_session_eval.sh`

### 확인(2026-10-27)
- 호스트 스모크(GPU 6, E6 헤드에서 20 step, 정답·모델 예측 캐시 섞음): 손실 0.35 → 0.26, frame_acc 0.81 → 0.86, 2.4 s/step(배치 2, 클립당 청크 약 27 개).
- 저장·재로드 대조: 손실 0.389247 같음, 가중치 차이 0.
- 서버 시험: test_diar 14 통과. test_long_session·test_slot_input·test_cst_dialogue 등은 107 통과.
  - 실패 1 은 test_stage3_train_flags 하위 프로세스 300 s 시간 초과(NFS import 지연)다. 맥에서는 4 s 에 통과한다.

### 결과 1 — E6 창 단위 평가(job 95936, 2026-10-27 18:50, `VAPKT-data/logs/e6-compare.log`)
AMI δ10 cpWER, 괄호는 귀속 몫이다. 짝 bootstrap 은 회의 단위 1000 회이고, E6 대 F1 은 모든 칸에서 95 % 구간이 0 을 포함하지 않는다.

| | F1 | E6 | E6 avg3 | E5 정답 활동(상한) | E6 + 정답 활동 |
|---|---:|---:|---:|---:|---:|
| IHM-mix dev | 54.82 (32.07) | 43.15 (20.38) | 42.17 (20.06) | 24.73 (1.81) | 31.79 (9.12) |
| IHM-mix eval | 49.43 (26.91) | 37.68 (15.15) | 35.53 (13.09) | 23.39 (1.00) | 28.02 (5.19) |
| SDM dev | 58.95 (32.24) | 48.87 (22.23) | 47.69 (21.23) | 29.25 (2.13) | 34.19 (7.66) |
| SDM eval | 53.94 (25.59) | 45.35 (17.29) | 44.15 (16.15) | 29.24 (1.40) | 34.39 (6.41) |

- F1 대비 −8.6 ~ −11.8 이고, 전부 귀속에서 나왔다(agn 같음).
- E5 상한까지는 14–20 이 남았다. 헤드 예측 오류 몫이다.
- E6 에 정답 활동을 넣으면 E5 보다 4.6–7.1 나쁘다(δ10). 학습 때 헤드 예측을 섞어(ρ) 활동 입력을 덜 믿게 된 것으로 본다.
- 같은 언어 대화 7 코퍼스(δ10): cp 26.31 → 22.26, 귀속 7.91 → 3.92, 대화 BLEU 39.71 → 41.44.
- TAXI 세션 통째: δ10 은 F1 20.9 → E6 19.4(mt4 18.9), δ4 는 25.0 → 21.1(mt4 23.1).
  - 짝 비교에서 턴 통째로 틀린 교대가 줄었다(δ10 −2.0, δ4 −3.5, 둘 다 유의).
- F1 avg3 의 FLEURS ko-en 악화(+11.67)는 E6 에서 사라졌다. 단일 화자 셋의 최대 악화는 +1.04(callhome-de)다.

### 결과 2 — 회의 단위 DER(호스트 GPU, `results/diar-session-{e6,e7a}`, 2026-10-27)
- AMI 공식 dev 18·eval 16 회의를 통째로 흘렸다. 참조는 회의 전체 도착 순서다.
- 표의 값은 회의 전체 한 순열의 최적 짝 DER(%)이다. 놓침·오경보·혼동을 다 넣었다.

| | E6 헤드, 90 s 조각마다 새로 | E6 헤드 + AOSC | **E7a 헤드 + AOSC** | E7a 헤드, 90 s 조각마다 새로 |
|---|---:|---:|---:|---:|
| IHM-mix dev | 66.27 | 52.23 | **34.95** | 64.12 |
| IHM-mix eval | 66.98 | 54.32 | **32.23** | 66.35 |
| SDM dev | 67.10 | 57.06 | **40.82** | 65.10 |
| SDM eval | 67.68 | 51.72 | **40.12** | 68.35 |

- **E7a:** 같은 AOSC 에서 E6 헤드보다 11.6–22.1 낮고, 조각마다 새로 정하는 방식보다 26–35 낮다.
  - E7a + AOSC 의 성분은 혼동 13.4–21.6, 놓침 12.5–15.3, 오경보 약 5 다.
  - E6 + AOSC 는 혼동 26–31, 놓침 17–20 이었다. 혼동과 놓침이 둘 다 줄었다.
- **조각마다 새로 정하면** E7a 도 E6 와 같은 수준(64–68)이다. 이득은 캐시를 쓰도록 학습한 데서 나왔다.
- **화자 수별(eval):** 3 인 회의는 IHM-mix 23.07·SDM 31.04, 4 인 회의는 33.32·41.20 이다.
- **도착 순서 그대로의 짝은 64–71 로 높다.** 슬롯이 회의 내내 같은 사람을 가리키지만, 순서가 회의 전체 도착 순서와 같지는 않다.
  - cpWER·최적 짝 DER 은 순열과 무관하다.
  - LLM 이 회의 단위 슬롯을 따르게 할지(E7b)는 순서보다 일관성의 문제다.

### 결과 3 — 회의 통째 cpWER, E6 LLM + E7a 헤드 + AOSC(job 96172, 2026-10-28 02:55, `results/stage3-eval-amifull-c30-aosc-e7a-q17-e6-final`)
AMI 34 회의 × IHM-mix·SDM 을 통째로 디코드했다(LLM 문맥 30 s). 표준 = Whisper 정규화 + meeteval(`-wn`).

| AMI(δ10) | cpWER(우리 정규화) | cpWER(표준) | agn(표준) | 귀속 몫 |
|---|---:|---:|---:|---:|
| IHM-mix eval | 68.23 | 67.34 | 21.06 | 46.2 |
| IHM-mix dev | 72.63 | 71.78 | 21.14 | 50.6 |
| SDM eval | 74.36 | 73.10 | 27.36 | 45.7 |
| SDM dev | 80.23 | 79.62 | 26.67 | 52.9 |

- **받아쓰기는 창 단위와 같다**(표준 agn IHM-mix eval 21.06 대 창 21.49, SDM 27.36 대 27.16). 회의 통째 디코드가 받아쓰기를 망치지는 않는다.
- **귀속이 무너졌다(46–53).** E7a 헤드의 회의 단위 DER(32–40, 혼동 13–22)보다 훨씬 크다.
- **LLM 이 헤드의 회의 단위 슬롯을 따르지 않는 것으로 본다.**
  - 회의마다 가설 슬롯 네 개를 다 쓰고(가장 많이 쓴 슬롯의 단어 몫 중앙값 38 %), 슬롯별 단어 수도 참조 화자 분포와 비슷하다.
  - 그런데 cp 가 크다(예: IHM-mix IS1009b 는 agn 16.0 에 cp 76.3). 한 화자의 말이 회의 동안 여러 슬롯으로 흩어진다.
  - E6 LLM 은 창 도착 순서 슬롯(창 안 첫 화자 = A)으로만 학습했다. 문맥 30 s 안의 첫 화자가 A 가 아닌 회의 단위 슬롯 입력과 부딪친다는 가설이다(7 절 순서 3 의 E7b 조건).
- 96461(E6 헤드로 조각마다 새로·AOSC)과 견주면 AOSC 가 LLM 에 무엇을 바꾸는지 가른다.
- 96461 의 E6 헤드 + AOSC 는 IHM-mix eval 77.81·SDM 82.75 다(우리 정규화, δ10). E7a 헤드가 8–9 낮지만 둘 다 귀속이 무너졌다.
  - 조각마다 새로 쪽은 몫 0 이 `--verify` 대조 버그로 멈췄다. 예측 활동 스트림까지 대조했기 때문이다. 대조를 고쳤고, 다시 내면 몫 0 만 돈다.

### 결과 4 — 사후 진단: 가설 단어를 활동으로 다시 귀속 (2026-10-28, `relabel_diag.sh`, 96172 원출력, δ10, 표준 정규화)
| AMI 회의 통째 cpWER | 태그 그대로 | E7a 활동으로 다시 귀속 | 정답 활동으로 다시 귀속 | agn |
|---|---:|---:|---:|---:|
| IHM-mix eval | 67.29 | **41.93** | 30.11 | 21.06 |
| IHM-mix dev | 71.77 | 53.32 | 32.30 | 21.14 |
| SDM eval | 73.08 | **55.49** | 35.22 | 27.36 |
| SDM dev | 79.56 | 60.45 | 36.83 | 26.67 |

- **LLM 태그가 헤드 슬롯을 따르기만 해도** eval cpWER 가 IHM-mix 25 pt, SDM 18 pt 내린다. 귀속 붕괴의 주원인은 LLM 태그다.
- **정답 활동이면 30.1·35.2 다.** E7a 헤드 품질이 그다음 병목이다(eval 에서 12–20 pt).
- **외부와 비교(세션 단위):** E7a 로 다시 귀속하면 41.9·55.5 로 VibeVoice-Streaming 1.5B(45.15·56.83, 480 s 조각)보다 낮다.
  - Parakeet(21.26·37.44)·VibeVoice 7B(27.48·39.01)에는 아직 못 미친다.
  - 정답 활동이면 SDM 35.2 로 Parakeet 37.4 보다 낮다.
- **디코드 때 태그 이끌기(96942) 첫 회의 7 개(IHM-mix δ10, 우리 정규화)**
  - cp 가 모든 회의에서 내렸다: ES2011a 69.2 → 49.9, IS1009a 80.9 → 46.0, TS3003a 48.6 → 23.7 등.
  - 바꾼 태그는 10–25 % 다.

### 결과 5 — 태그 이끌기 회의 통째 전체(job 96942, 2026-10-28 13:01, `results/stage3-eval-amifull-c30-aosc-e7a-guide-d10-q17-e6-final`, 재채점 `-wn`)
| AMI 회의 통째 δ10 | 태그 그대로(96172) | **태그 이끌기** | E7a 활동 사후 재귀속(결과 4) | agn(이끌기) |
|---|---:|---:|---:|---:|
| IHM-mix eval | 67.34 | **47.50** | 41.93 | 21.06 |
| IHM-mix dev | 71.78 | **57.36** | 53.32 | 21.80 |
| SDM eval | 73.10 | **61.57** | 55.49 | 27.90 |
| SDM dev | 79.62 | **67.79** | 60.45 | 26.28 |

- 앞 두 열은 meeteval(Whisper 문장 정규화) cpWER 다. 사후 재귀속과 agn 은 Whisper 단어 정규화다. 이끌기의 단어 정규화 cp 는 47.37·57.40·61.44·67.82 로 meeteval 과 0.1–0.2 차이다.
- **학습 없이 eval cp 가 IHM-mix 19.8 pt, SDM 11.5 pt 내렸다.** 사후 재귀속(진단 상한)과의 차이는 5.6·6.1 pt 남는다.
  - 이끈 태그를 다시 넣어 이후 디코드가 그 슬롯을 보므로 받아쓰기(agn)는 거의 그대로다(21.06·27.36 → 21.06·27.90).
- **외부와 비교(세션 단위):** IHM-mix 47.50 은 VibeVoice-Streaming 1.5B(45.15, 480 s 조각, 2.0 s)와 비슷하고 SDM 61.57 은 그보다 높다(56.83). Parakeet v1 21.26·37.44, VibeVoice 7B 27.48·39.01 과는 거리가 크다.

### 결과 6 — E7b 창(job 96963 AMI 창 δ10, 2026-10-28 13:16, `results/stage3-eval-ami-d10-q17-e7b-final`, 재채점 `-wn`)
| AMI 창 δ10 | E6 cp | **E7b cp** | agn E6 → E7b | 귀속 몫 E6 → E7b |
|---|---:|---:|---|---|
| IHM-mix eval | 37.09 | **35.74** | 21.49 → 22.72 | 15.52 → 12.92 |
| IHM-mix dev | 42.31 | **41.38** | 21.61 → 22.88 | 20.70 → 18.46 |
| SDM eval | 44.73 | **44.05** | 27.16 → 29.03 | 17.53 → 14.95 |
| SDM dev | 48.36 | **47.79** | 25.61 → 27.29 | 22.66 → 20.43 |

- cp 는 meeteval(Whisper), agn·귀속 몫은 Whisper 단어 정규화다.
- **귀속 몫은 2.2–2.6 pt 줄었지만 받아쓰기(agn)가 1.2–1.9 pt 나빠져** cp 는 0.6–1.4 pt 만 내렸다.
  - E7b 는 인코더를 얼리고 E6 final 에서 2,500 step 을 lr 4e-5 로 다시 데웠다. 받아쓰기가 나빠진 까닭은 아직 가르지 않았다.

**회의 통째(E7a 헤드 + AOSC, 96963 의 몫 8 개, 16:16 끝, `results/stage3-eval-amifull-c30-aosc-e7a-d10-q17-e7b-final`, 재채점 `-wn`)**

| AMI 회의 통째 δ10(meeteval cpWER) | E6 태그 그대로 | **E7b** | E6 + 태그 이끌기 | agn E6 → E7b |
|---|---:|---:|---:|---|
| IHM-mix eval | 67.34 | **55.53** | 47.50 | 21.06 → 21.92 |
| IHM-mix dev | 71.78 | **63.24** | 57.36 | 21.14 → 22.21 |
| SDM eval | 73.10 | **68.37** | 61.57 | 27.36 → 28.63 |
| SDM dev | 79.62 | **70.88** | 67.79 | 26.67 → 27.07 |

- E7b 학습으로 회의 통째 cp 가 4.7–11.8 pt 내렸다. 그래도 학습 없는 태그 이끌기(E6)보다 6–8 pt 높다. 귀속 몫은 여전히 33.6–43.9 다.
- 받아쓰기(agn)는 창과 같이 0.4–1.3 pt 나빠졌다.
- 남은 조합: E7b 에 태그 이끌기를 함께 쓰기(학습 없음, 평가 8 몫).

### 대책 셋 (2026-10-28, 사용자 "… 1·2·3 … 진행해보자")
1. **사후 진단:** `cst_stage3_eval --from-predictions … --window full --relabel-act <probs> [--relabel-oracle]`(`cst_eval.relabel_by_activity`).
   - 가설 단어를 방출 청크 − δ 근처 활동이 가장 큰 슬롯으로 다시 나눠 cp 를 잰다. 근거는 E7a AOSC 확률(`diar_eval --session --save-probs`)과 정답 활동 둘이다.
   - LLM 태그가 헤드 슬롯을 따랐다면 얻었을 귀속을 어림한다.
2. **디코드 때 태그 이끌기:** `--act-tag-guide`(`long_session.make_retag`, `realtime_decode(retag=)`), 평가 묶음 `q17-e7-amifull-guide`.
   - 모델이 화자 태그를 내면, 그 뒤 말이 올 오디오 [k − δ − 2, k] 의 활동이 가장 많은 슬롯 태그로 바꾼다. 모델 슬롯도 그 구간에서 말하면 그대로 둔다.
   - 바꾼 태그를 다시 넣으므로 이후 디코드도 그 슬롯을 본다. 행에 retag{tags, changed} 가 남는다.
3. **E7b:** `slurm/submit-q17-e7b.sh`.
   - E6 final 에서 F1 레시피로 2,500 step(2 노드), `--global-slots 0.5 --freeze-encoder` 다. 인코더를 얼려 E7a 헤드를 그대로 끼울 수 있다.
   - 평가 q17-e7b 는 회의 통째 AOSC + E7a 헤드 8 몫과 AMI 창 4 몫이다.

### 순서
1. E6 헤드 그대로 회의 단위 DER 을 잰다(90 s 조각마다 새로 대 AOSC — AOSC 학습 전 기준선, 호스트 GPU).
2. E7a(AOSC 헤드 미세조정, 1 노드 5,000 step — 스모크가 공유 GPU 에서 2.4 s/step 이라 3–4 h 어림; job 96171, E6 평가 95936 이 끝난 뒤 시작, 평가 96172)를 학습하고, 회의 단위 DER 과 `q17-e7-amifull`(E6 LLM + E7a 헤드, 회의 통째 cpWER)을 잰다.
3. 회의 통째에서 LLM 이 회의 단위 슬롯을 따르지 못하면(귀속 몫이 크면) E7b(`--global-slots 0.5`, E6 final 에서 짧게)를 한 뒤, 그 인코더로 헤드를 다시 AOSC 미세조정한다.
4. 실시간 경로(`realtime.py`·`live.py`)에 헤드·AOSC 를 붙인다.

## 8. E8 — 받아쓰기 지연 안에서 헤드에 앞을 보이기(헤드 출력 지연 R) (2026-10-27)
사용자: "그런데 아직까지 cpWER 이 한참 모자르네." → "E8 구현해놓자".

### 근거
- AMI eval δ10 창 cp 는 SDM 45.35 = agn 28.06 + 귀속 17.29 다.
  - E5 정답 활동이면 귀속이 1.4 로 떨어진다. 가장 큰 몫이 귀속이고, 형식이 아니라 헤드 예측 품질의 문제다(7 절 결과 1).
- 인과 가지(E1c)의 창 DER 은 양방향(E1)보다 약 4 높고, 대부분 놓침이다(말 시작을 늦게 잡는다 — 6 절).
- LLM 은 단어를 δ(학습 4–10 프레임, 320–800 ms) 늦게 낸다. 그래서 프레임 f 의 활동이 R ≤ δ 늦게 와도, 그 단어를 낼 때는 이미 와 있다.

### 방법
- **헤드:** 구조는 그대로 인과다(층마다 j ≤ i). 학습 정답만 R 만큼 민다.
  - 위치 p 의 출력이 프레임 p − R 의 활동을 맞힌다(`vapasr.hf.diar.shift_frames`). 헤드가 R 프레임 뒤까지 보고 판단하므로 놓침을 겨눈다.
  - 창 앞 R 프레임과 끝 패딩 위치는 손실에서 뺀다.
- **LLM 입력:** 위치 p 의 헤드 출력(= 프레임 p − R)을 오디오 토큰 p 에 그대로 더한다. 학습 중 ρ 로 섞는 흔든 정답도 R 만큼 민다.
  - 추론 코드(encode·chunk_embed·창/회의 디코드)는 바꿀 것이 없다.
- **R = 4(320 ms):** 학습 지연 δ4–10 의 가장 짧은 값이다. `stage3_train` 은 R > min(δ) 를 막는다.
- **코드**
  - `configuration_vapasr.act_shift`
  - `modeling_vapasr.forward`: 헤드 손실 정답·마스크·흔든 정답 입력을 민다.
  - `stage3_train --act-shift`
  - `diar_train --shift`: 헤드만 학습.
  - `diar_eval`: diar.pt 의 shift 로 출력 [R:] 을 참조 [:K − R] 과 맞춘다. 창 끝 R 프레임은 뒤를 볼 오디오가 없어 뺀다.
  - `diar_extract`: shift 를 넘긴다.
  - `cst_stage3_eval --diar-head-override`: shift 가 다르면 멈춘다.
  - AOSC 경로(`encode_long(aosc)`·`diar_eval --session aosc`)는 아직 막았다. 캐시 점수가 p − R 시점과 어긋난다.
- **시험:** 맥 test_diar·test_slot_input 25 개가 통과했다(shift_frames·unshift, forward 의 헤드 손실·ρ 0 흔든 정답 이동·ρ 1 헤드 예측·R 0 이면 예전 그대로).

### 실험
- E8 = E6 레시피 그대로 + `--act-shift 4`(`slurm/submit-q17-e8.sh`).
  - 제출(2026-10-27 23:57): 학습 job 96581 은 96461·96172 가 끝난 뒤 시작하고, 평가 job 96582 가 그 뒤를 잇는다.
  - 시작점·데이터·step·헤드 시작점은 E6 과 같다: 5 단계 ck3000, F1 데이터, 5,000 step, 2 노드, E1c 헤드. E6 대비 차이는 R 하나다.
- 평가는 `q17-e8` 이다(q17-e6 과 같은 v3 묶음, 헤드 예측 활동, ck3000·ck4000·final·avg3, AMI 4 몫).
  - 정답 활동 상한(amiact)은 넣지 않았다. `--act-oracle` 이 활동을 밀지 않는다.
- 판단: AMI 창 cp, 특히 귀속 몫과 TAXI 를 E6 과 회의·세션 짝으로 비교한다. 헤드 DER 은 `diar_extract` → `diar_eval` 로 E1c·E6 헤드와 견준다.

### E8 결과 (2026-10-28 — 학습 96581, 평가 96941 AMI 창 δ10, 헤드 DER 은 호스트 `head_window_der.sh`)
| AMI 창 | E6 cp | **E8 cp** | E6 헤드 DER | **E8 헤드 DER**(R 4) | 놓침 E6 → E8 |
|---|---:|---:|---:|---:|---|
| IHM-mix eval | 37.68 | 39.99 | 23.84 | **21.04** | 16.04 → 13.40 |
| IHM-mix dev | 43.15 | 44.47 | 24.55 | **22.03** | 15.14 → 12.62 |
| SDM eval | 45.35 | 46.19 | 27.15 | **24.59** | 17.74 → 15.49 |
| SDM dev | 48.87 | 49.50 | 28.06 | **25.40** | 17.43 → 14.92 |

- DER 은 최적 짝, 임계 0.5, 창 끝 R 프레임 제외다.
- **표준 정규화(Whisper + meeteval)로도 같다.** E6 → E8 은 IHM-mix eval 37.09 → 39.43, dev 42.31 → 43.61, SDM eval 44.73 → 45.62, dev 48.36 → 48.99 다.
  - 받아쓰기(agn, Whisper)는 21.5 → 21.6(IHM-mix eval), 27.16 → 27.17(SDM eval)로 같다. 귀속만 0.5–2.3 늘었다.
- **헤드는 좋아졌다.** 네 셋 모두 DER 이 2.5–2.8 내렸고, 대부분 놓침이다. 받아쓰기 지연 안에서 앞을 보게 한다는 설계 의도대로다.
- **LLM 을 거친 cp 는 좋아지지 않았다**(+0.6 – +2.3, 귀속 몫도 늘었다). 체크포인트 사이 흔들림(E6 final 대 avg3 1–2 pt)과 비슷한 크기다.
- **해석:** LLM 이 활동 입력을 충분히 따르지 않아(7 절 결과 3·4 와 같은 뿌리) 헤드 개선이 cp 로 옮겨 가지 않는다. 태그가 활동을 따르게 되면(태그 이끌기·E7b) 헤드 개선이 cp 로 이어질 수 있다.
- **외부 참고(세션 단위·조건 다름):** Streaming Sortformer DER 은 IHM 16.67·SDM 20.57 이다(arXiv 2609.10265 표 1). Nemotron 3 Diarization 은 AMI SDM 12.8 @1.04 s 다.

## 9. E9 — 슬롯 블록 형식: 청크마다 모든 슬롯 태그를 입력으로 (2026-10-28)
사용자: "그럼 시퀀스를 개선해볼까? … 1 블록 내에서 <SPK_A~D> 가 한번씩 무조건 입력으로 주어지게끔" → "<SPK_B> 토큰을 예측해야 하잖아 … 손해가 있지 않을까"
→ "<BEGIN> 등의 특수 토큰도 예측하도록 하면 좋을거 같은데" → "응 그렇게 E9 구현해놓자" → "<SPK_*> 의 수를 추후에 조절할 수 있도록 해줘. 즉 최소 샘플의 화자 수, 최대 4명 화자로 해서 랜덤으로 선택되게끔".

### 근거
- 7 절 결과 4: LLM 이 태그를 헤드 활동대로만 내도 회의 통째 eval cpWER 가 IHM-mix 25 pt, SDM 18 pt 내린다. 지금 형식에서는 LLM 이 태그를 고르고, 그 선택이 활동 입력을 따르지 않는다.
- E9 는 LLM 이 태그를 고르지 않게 한다. 청크마다 선언 슬롯 태그를 모두 정해진 입력으로 주고, 슬롯마다 "이 슬롯이 이번 청크에 낼 것"만 묻는다. 태그 임베딩에 그 슬롯의 활동을 더해 활동이 그 결정에 바로 닿게 한다.

### 형식
```
청크 k:  ▮k  <SPK_A> [<BEGIN>] [단어…] [<EOT>]  <SPK_B> …  <SPK_C> …  <SPK_D> …  ▮k+1
라벨 :   각 슬롯 자리 = <BEGIN> | 단어 | <EOT> | <NEXT_AUDIO>(이 슬롯은 이번 청크 끝)
```
- **태그는 입력이다.** label 이 없다.
  - 슬롯의 멈춤은 다음 입력 위치(다음 태그·다음 청크 오디오)의 label `<NEXT_AUDIO>` 다. 위치를 더하지 않는다. 가중은 next_weight(0.3, 한국어 0.15)다.
  - 오디오 바로 뒤 첫 태그의 label 은 없다(태그는 정해져 있다).
- **`<BEGIN>`** = 동결 registry 의 `<ONSET>`(151721)이다. 새 어휘는 없다.
  - E6 config 에서 예약·차단된 행이라 평균 + 잡음으로 다시 초기화하고 차단을 푼다.
  - episode 시작 + 4 청크(δ 와 무관)에 그 슬롯의 첫 출력으로 낸다. δ ≥ 4 라 첫 단어보다 늦지 않다. 가중 2.
- **`<EOT>`** = episode 끝 + δ 청크(마지막 단어 뒤)다.
  - 발화 구간 표지라 soft label 이 없다. 전사 없는 발화·맞장구·합성 대화에도 낸다. E6 의 턴 끝 `<EOT>`(p_end soft, 맞장구·합성 제외)와 다르다.
- **전사 없는 음성**은 그 화자 슬롯의 전사·멈춤 label 만 가리고 `<BEGIN>`·`<EOT>` 는 학습한다. E6 는 그 청크를 모든 슬롯에서 가렸다.
- **flush**는 `<EMPTY_AUDIO>` + 태그 블록(남은 출력), 빈 라운드, 마지막 멈춤 label 을 실을 `<EMPTY_AUDIO>` 순이다.
- **선언 슬롯 수**(사용자 요청)는 학습 창마다 [창 화자 수, 4] 에서 고르게 뽑는다(`--declare-range`). 한 화자 창은 1–4 다. 추론 때는 prefix 의 선언 수(`--declare`)로 정한다.
- **길이**: 청크당 위치가 E6 의 2 + 말에서 1 + D + 말로 는다(4 슬롯이면 약 2.3 배). 학습 예산(GPU 당 16,384 토큰)은 같아서 step 당 창 수가 준다.

### 모델
- `add_slot_blocks`: `<ONSET>` 행 재초기화·차단 해제, `adapter_tagact` = Linear(12 → D, bias 없음) 0 초기화(붙인 직후 출력은 그대로).
- 태그 위치 임베딩 += adapter_tagact([a_s(k), a_s(k − 1), …, a_s(k − 11)]) — 그 슬롯의 최근 12 청크 활동이다.
  - δ(최대 10) 안의 어느 시점을 볼지는 LLM 이 고른다(지연 벡터로 준다).
  - 학습은 오디오 토큰과 같은 act_in(ρ 로 헤드 예측 또는 흔든 정답)이고, 디코드는 헤드 예측·AOSC 다.
- 오디오 토큰의 슬롯 활동 입력(E6 `adapter_slot`)은 그대로 둔다.

### 디코드
- **창(배치)**: `batch_decode.decode_blocks_batch` — 행마다 한 위치씩 넣는다(`BlockState`). 태그·출력 뒤 예측이 멈춤(`<NEXT_AUDIO>`·슬롯 태그·`<EMPTY_AUDIO>`)이면 다음 태그로 간다.
- **회의 통째·실시간**: `realtime.realtime_block_decode`(LongThinker·GraphThinker `run_all`).
  - [오디오, 태그 D 개]를 한 forward 로 미리 넣는다. 멈춤이 아닌 슬롯에서만 그 태그 뒤로 되돌아가 한 토큰씩 낸다.
  - 모든 슬롯이 멈추는 청크는 forward 한 번이다. 출력은 한 토큰씩 디코드와 같다(시험).
- 출력 기록은 E6 와 같은 [(k, 태그), (k, 토큰) …] 꼴이다. 채점(`parse_emitted`)·재채점·meeteval·귀속 오류 분해가 그대로 돈다.

### 평가
- AMI 창 δ10과 회의 통째(E7a 헤드 + AOSC) δ10 이다(묶음 `q17-e9`). 평가가 config.slot_blocks 로 슬롯 블록 디코드를 고른다.
- **LLM 쪽 DER**: 가설 `<BEGIN>`·`<EOT>` 를 학습 규칙의 역으로 슬롯 구간으로 바꾼다(`cst_eval.activity_from_events`). 헤드 DER 과 같은 프레임 마스크로 잰다(행·요약 `llm_der`, `ami_report`).
- **디코드 비용**: 회의 통째 행에 forwards·chunk_ms(청크 ms 평균)·chunk_ms_p90 이 남는다.

### 코드
- 데이터(`vapasr/data/cst_dialogue.py`): `block_streams`·`build_block_sequence`·`block_problems`, `CstDialogueDataset(slot_blocks, declare_range)`·`block_sequence`·`slot_mask_chunks`, `collate_cst` tag_chunk.
- 패킹: `packing.pack_batch` tag_chunk(표본 · 2²⁰ + 청크).
- 모델(`modeling_vapasr.py`): `add_slot_blocks`·`tag_act_embed`·`tag_embeds`·`block_act`, `build(tag_chunk)`. 태그 없는 행(replay)은 0 가중으로 묶는다(DDP).
- 디코드: `batch_decode.BlockState`·`decode_blocks_batch`, `realtime.realtime_block_decode`·`GraphThinker.run_all`, `long_session.LongThinker.run_all`·`long_block_decode`.
- 평가: `cst_eval.parse_emitted(begin_id, events)`·`activity_from_events`·`emitted_from_blocks`, `cst_stage3_eval`(자동 선택·`--oracle --blocks`·llm_der·chunk_ms), `cst_attr_errors`, `ami_report`.
- 학습: `stage3_train --slot-blocks --begin-weight --tag-act-lags --declare-range`, `submit-q17-f1.sh --transcribe-only`, `slurm/submit-q17-e9.sh`, 평가 묶음 `q17-e9`.
- 시험 `tests/test_slot_blocks.py` 13 개가 통과했다(아래 서버 dry-run 이 찾은 경계 회귀 시험 포함).
  - 배치 디코드 상태기계가 정답 시퀀스를 그대로 따라간다.
  - 추측 prefill 이 한 토큰씩 디코드와 같다.
  - 오라클 글자·시각·슬롯이 E6 정답과 같다.
  - 슬롯별 마스크, 선언 수 범위, 태그 활동 값, packing 을 확인했다.
  - 맥 관련 시험 223 개도 통과했다. 실패 5 개는 로컬 환경(rapidfuzz 없음 등) 문제다.
- **서버 확인(2026-10-28, mxc 호스트 CPU — 코드 18 파일 md5 대조 뒤)**
  - 시험 97 개가 통과했다(test_slot_blocks·test_cst_decision·test_slot_input·test_v2_slots·test_long_session·test_packing).
  - **첫 dry-run 이 순서 위반을 찾았다.** 셋마다 64 창 × δ7 의 불변식 검사에서 δ4 에서만 21 창(dlg_de 4·dlg_mt 4·dlg_far 10·dlg_sw 3)이 나왔다.
    - 창 시작을 뺀 시각이 80 ms 경계 바로 아래(2.999…)면 `chunk_of`(+1e-9)와 글자 버킷 `int()` 가 한 청크 어긋난다.
    - 그래서 δ = begin_lag(4)이고 첫 글자 시각 = 발화 시작이면 `<BEGIN>` 이 첫 글자 뒤로 갔다.
    - `<BEGIN>`·`<EOT>` k_min 을 버킷과 같은 식 int(t/0.08)로 고치고 회귀 시험을 더했다(예전 식이면 실패). 커밋 6496db0.
  - **고친 뒤 dry-run**(E9 제출 인자 그대로, `tmp/e9-check/dryrun2`): 19 셋 모두 불일치 0 이다.
    - 지문은 slot_blocks{begin_weight 2, begin_lag 4, tag_act_lags 12}·declare_range·ast_frac 0·sim_trans_frac 0·번역 타깃 없음·global_slots 0.5 다.
    - 시퀀스 예(ami ES2008d, δ10, ↵ = 입력 위치의 멈춤 label): `A <SPK_A> <SPK_B>↵ <SPK_3>↵ <SPK_4>↵ A↵ <SPK_A> <ONSET> <SPK_B>↵ … <SPK_A> ' guess' <SPK_B>↵ …`
  - **AMI 오라클**(`--oracle --blocks`, 창 20–90 s 이어 덮기, 2,396 창 × δ4·10, `tmp/e9-check/oracle-ami-blocks2`)
    - cp·slot 0.00, EOT 재현율 0.999–1.000 이다.
    - agn 2.3–2.4 는 겹침 구간을 시각순으로 잇는 순서 차이(정답 시각 대 방출 시각)라 형식 오류가 아니다.
    - LLM DER 바닥값은 δ4 0.33–0.49 %, δ10 1.19–1.36 % 다(대부분 놓침). 창 앞에서 시작한 발화는 첫 글자 − δ 부터 잡히고 구간 끝은 청크 단위라서다.

### 실험(제출 2026-10-28 15:51 — 학습 job 97163, 평가 job 97164)
- E9 는 E6 final 에서 시작한다. F1 데이터·가중치·lr 4e-5 를 쓰고 받아쓰기만 한다(`--transcribe-only`: 번역·교대/머무름 가중·EOT-문맥 dropout 없음, replay 번역 끔).
  - 덧붙임: `--slot-input --diar-head <E1c> --slot-blocks --declare-range --global-slots 0.5 --freeze-encoder`.
  - 5,000 step, 2 노드다. 인코더를 얼려 E7a 헤드를 회의 통째 평가에 끼운다.
- 제출: `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-e9.sh)` — 학습 뒤 평가 `q17-e9` 가 잇는다.
  - 사용자가 15:51 에 냈다. 학습 job 97163(SA_SFT_FullDuplex, hpc-34·35, 산출 `/soundai/Model/VAPASR/stage3-q17-e9`), 평가 job 97164(SA_EVAL_FullDuplex, 97163 성공 뒤)다.
- 판단: E6·E7b 와 같은 조건(δ10, 표준 정규화)으로 창 cp·회의 통째 cp·LLM DER·청크 ms 를 비교한다.
