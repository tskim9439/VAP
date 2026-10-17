---
author_id: tskim
created: 2026-10-17
source_type: source-drop
---

# Stage 3 화자 슬롯 활동 보조 손실(화자 정체 헤드) 구현과 파일럿 준비 (2026-10-17, mxc 호스트)

## 배경
- 사용자 요청(2026-10-17): "2번 구현해줘". 2번은 화자 귀속 개선 후보 2 순위인 화자 정체 보조 손실이다.
- 정체 회복 파일럿([[2026-10-17-stage3-recover-pilot]])은 step 2,000 에서 연속 뒤바뀜을 줄이지 못했다.
  - 회복을 가르치는 것만으로는 부족하고, "이 목소리가 누구인가" 라는 신호 자체가 약하다고 봤다.
  - 오류 분해([[2026-10-16-stage3-main-r1]])에서 오배정의 73 % 가 턴 통째 오배정이다.

## 구현 (기본 끔)
- **라벨 `vapasr/data/cst_dialogue.slot_activity`:** 청크(80 ms)마다 [슬롯 A 말함, 슬롯 B 말함] 0/1 이다.
  - 슬롯 순서는 직렬화와 같은 `window_speakers`(창에서 먼저 말한 사람 = A)다.
  - "말함" 은 episode 구간이다. 전사 없는 발화도 포함하고, gap < 0.25 s 는 병합한다. 겹침 구간은 둘 다 1 이다.
  - `CstDialogueDataset(spk_activity=True)` 이면 `__getitem__` 이 `activity` (K, 2) 를 낸다.
  - `collate_cst` 가 `activity` (B, Kmax, 2)·`activity_mask` (B, Kmax) 로 묶는다. 인코더 출력과 줄이 같아 `packing.pack_batch` 가 그대로 편다.
- **모델 `vapasr/hf/modeling_vapasr.py`:**
  - `add_spk_head(slots=2, weight)`: LLM 의 audio 위치 hidden 에 2 층 헤드(hidden → 256 → 2)를 붙인다.
    - Phase 2 lane 활동 헤드와 같은 경로다(`gather_audio_targets`·`activity_bce`).
    - 설정 `config.spk_slots`·`spk_weight` 를 저장하므로 다시 불러오면 헤드도 생긴다.
  - 손실 = 원래 손실 + `spk_weight` × BCE 다. 디코드는 헤드를 쓰지 않는다.
  - 텍스트 화자 태그는 δ 청크 늦게 나온다. 그래서 지금 청크의 슬롯을 맞히려면 목소리로 정체를 추적해야 한다.
  - 활동 라벨이 없는 행(replay)은 헤드를 0 가중으로 그래프에 묶는다. DDP(`find_unused_parameters=False`)가 쓰이지 않은 파라미터를 기다리지 않게 하기 위해서다.
- **trainer `vapasr/hf/trainer.py`:**
  - `activity`·`activity_mask` 를 모델로 넘긴다.
  - 새 헤드(`spk_head.*`)는 adapter 와 같은 lr(1e-3) 그룹에 넣는다.
- **`experiments/stage3_train.py`:**
  - `--spk-act-weight W`(0 = 끔)
  - 지문 `config.stage3.spk_act_weight` 는 켤 때만 넣는다.
  - 로그 `loss_spk`·`spk_acc`, parity 배치에 activity 를 넣는다.
- **`slurm/submit-stage3-train.sh`:** `--spk-act-weight W`(기본 0)
- **테스트:**
  - `tests/test_cst_dialogue.py`(+2):
    - `slot_activity` 값과 슬롯 밖 화자, collate·pack
    - 헤드 손실 가중, 라벨 없는 행의 0 기울기 묶음, eval 모드, config 로 다시 만들 때 헤드 생성
  - mxc: 관련 테스트 95 개 통과(packing·p2_losses·lane·semcommit 손실 포함)
  - 로컬 맥: 실패 3 개는 `test_packing` varlen 이다. 맥의 transformers 버전이 낮아서(`DynamicCache`·`masking_utils` 없음)이고 변경과 무관하다. mxc 에서는 통과했다.

## GPU 스모크 (호스트 GPU 2·4 torchrun, 20 step, `stage3_smoke_spk.sh`, `smoke-train.txt`, `smoke-parity.txt`)
- 레시피: r1(p7)과 같다. 맞장구 1.0,3, 대화 KO 2 · EN 0.7 · DE 0.7, replay 각 0.75, lr 4e-5. 여기에 `--spk-act-weight 0.5` 를 더했다. world 2 다.
- **DDP:** replay 만 돈 step(2·4·12·18)도 멈추지 않았다. 라벨 없는 행의 0 기울기 묶음이 동작한다.
- **헤드 학습:** `loss_spk` 0.70 → 0.56–0.59, `spk_acc` 0.53 → 0.70(20 step)
- **final 재로드 parity OK:**
  - 키 누락·추가·값 차이 0
  - `loss_act` 메모리 0.595 / 재로드 0.595
  - 저장 키 `spk_head.{0,2}.{weight,bias}`, config `spk_slots` 2·`spk_weight` 0.5·`stage3.spk_act_weight` 0.5

## 파일럿 st3-p9-spkhead (사용자 제출: job 89868, 2026-10-17 14:58 — 결과는 아래)
- **경과:**
  - 14:58 hpc-44·55 에서 시작했다. 모델 로드 156 s, 화자 슬롯 헤드 켬(가중 0.5)
  - 학습 전 점검은 대화·replay 모두 불일치 0 이다. 섞기는 r1 과 같다(steps/epoch 625, world 16).
  - 15:12 step 10 에 이르렀다. `loss_spk` 0.94 → 0.88, `spk_acc` 0.42 다(warmup 초반).
- **비교 대상:** r1 checkpoint-2000(Stage 3 최종). 레시피를 똑같이 두고 `--spk-act-weight 0.5` 만 더한다.
- **일정:** r1 과 같은 cosine 일정(`--max-steps 4000`, 2 노드)으로 내고 checkpoint-2000 끼리 비교한다.
- **제출 명령(mxc, 사용자):**

  ```bash
  cd /soundai/users/tskim/VAPKT
  (MAX_RESTART=5 bash slurm/submit-stage3-train.sh --run st3-p9-spkhead --partition apex --nodes 2 --backchannel 1.0,3 --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,rep_ko:0.75,rep_en:0.75,rep_de:0.75 --lr 4e-5 --max-steps 4000 --spk-act-weight 0.5)
  ```

  - 산출은 `/soundai/Model/VAPASR/stage3-st3-p9-spkhead` 이다. `--dry-run` 으로 확인했다(`--recover-p 0 --spk-act-weight 0.5`).
- **평가 큐(걸어 둠, 병렬):** ck2000 → GPU 1, ck3000 → GPU 2, final → GPU 4
  - 각각 대화 평가 → 오류 유형 분해 → 단일 화자 퇴행 순서다.
  - 로그는 mxc `results/stage3-regress/queue-p9-gpu{1,2,4}.log` 다.
- **판정(ck2000 대 r1 ck2000, δ10, 대화 단위 짝 bootstrap):**
  - 1 차: cp − agn(11.4)·통째 오배정(7.5 pt)·이어짐(4.5 pt)이 준다.
  - 같이 볼 것: agn(17.0)·EOT·단일 화자 퇴행이 나빠지지 않는다.
  - 헤드의 `spk_acc` 추이(학습 로그)

## 결과 (2026-10-17)
- **학습(`train-events.txt`):** 15:03 – 16:35, 4,000 step(4,868 s), 선점·재시작 없음, final 재로드 parity OK(`loss_act` 0.110 / 재로드 같음)
- **화자 헤드(`spk-head-trend.txt`):** `spk_acc` 는 step 0–499 0.78 → 500–999 0.91 → 2,000–2,499 0.94 → 3,500–3,999 0.95 다.
  - 슬롯별 0/1 을 청크마다 맞힌 비율이라 무음(둘 다 0)도 들어 있다.
- **평가:** 호스트 GPU 1·2·4 병렬, 18:24 에 끝났다. 비교용으로 r1 ck3000 의 오류 분해도 더 돌렸다.
- **대화 평가(맞장구 참조, 코퍼스 평균, `dialogue-eval-vs-r1.txt`):**

  | δ10 | r1 ck2000 | p9 ck2000 | r1 ck3000 | p9 ck3000 | r1 final | p9 final |
  |---|---|---|---|---|---|---|
  | cp | 28.6 | 29.2 | 30.5 | **26.8** | 31.0 | 27.9 |
  | agn | **17.0** | 19.4 | 17.9 | 18.0 | 18.0 | 18.0 |
  | EOT 재현율 | 0.666 | 0.655 | 0.630 | **0.673** | 0.632 | 0.659 |
  | EOT 정밀도 | **0.626** | 0.606 | 0.575 | 0.610 | 0.573 | 0.602 |
  | EOT 보류 오방출 | 0.173 | **0.156** | 0.185 | 0.182 | 0.176 | 0.183 |

  | δ4 | r1 ck2000 | p9 ck2000 | r1 ck3000 | p9 ck3000 | r1 final | p9 final |
  |---|---|---|---|---|---|---|
  | cp | 30.3 | 29.9 | 31.3 | 29.4 | 31.6 | **28.3** |
  | agn | **17.6** | 20.1 | 18.4 | 18.7 | 18.3 | 18.6 |

- **오류 유형(δ10, 단위 가중, `attr-errors-vs-r1.txt`):**

  | | r1 ck2000 | p9 ck2000 | p9 ck3000 | r1 final | p9 final |
  |---|---|---|---|---|---|
  | cp − agn | 11.4 | 9.6 | **8.6** | 12.8 | 9.8 |
  | 긴 턴 통째 | 5.4 | 4.5 | **4.2** | 5.9 | 4.9 |
  | 짧은 턴 통째 | 2.1 | 2.0 | **1.8** | 2.1 | 1.9 |
  | 통째 홀로 / 이어짐 | 3.0 / 4.5 | 2.3 / 4.2 | **2.1 / 3.9** | 2.7 / 5.3 | 2.3 / 4.4 |
  | 이른 교대 · 헛교대 | 1.3 · 0.2 | 1.7 · 0.4 | 1.4 · 0.4 | 1.8 · 0.4 | 1.5 · 0.3 |
  | 가설/참조 교대 | 0.91 | 1.29 | 1.11 | 1.10 | 1.12 |

- **대화 단위 짝 bootstrap(397 대화, 95 % 구간, 단위 가중, `bootstrap-vs-r1.txt`):**

  | 비교(δ10) | cp | agn | cp − agn | 통째 | 이어짐 |
  |---|---|---|---|---|---|
  | p9 ck2000 − r1 ck2000 | +0.71 [−0.60, +2.08] | **+2.48** | **−1.77** | **−1.00** | −0.26 |
  | p9 ck3000 − r1 ck3000 | **−3.95 [−5.22, −2.71]** | +0.13 [−0.13, +0.37] | **−4.07** | **−1.78** | **−0.79** |
  | p9 final − r1 final | **−2.91 [−4.24, −1.52]** | +0.13 | **−3.04** | **−1.24** | **−0.86** |
  | p9 ck3000 − r1 ck2000 | **−1.76 [−3.10, −0.50]** | **+1.05 [+0.69, +1.42]** | **−2.81** | **−1.44** | −0.57 |
  | p9 final − r1 ck2000 | −0.52 [−1.85, +0.76] | **+1.09** | **−1.61** | −0.74 | −0.07 |

  - δ4 도 같은 방향이다.
    - p9 ck3000 − r1 ck3000: cp −1.98, cp − agn −2.31
    - p9 final − r1 ck2000: cp −1.96 [−3.34, −0.57]
  - 언어별(δ10, p9 ck3000 − r1 ck3000): KO cp −3.9, EN −4.3, DE −3.9. 모두 구간 밖이다.
- **단일 화자 퇴행(축약, 세트당 100 발화, `regress-vs-r1.txt`):** p9 ck3000 은 대부분 r1 ck2000 과 ±0.5 pt 안팎이다.
  - en→de δ4 BLEU 는 21.5 → 23.6 으로 낫다.
  - Kspon other δ4 CER 은 8.5 → 10.0(final 10.3)으로 나쁘다. s4 는 7.9 다.
  - VoxPopuli δ10 WER 은 7.1 → 7.7 로 나쁘다.

## 읽기
- **화자 정체 헤드는 귀속을 크게 줄인다.**
  - 같은 step 끼리 cp − agn 이 3–4 pt 준다(ck3000 −4.1, final −3.0). 받아쓰기(agn)는 그대로다(+0.13, 구간 안).
  - 줄어든 것은 주로 턴 통째 오배정이다(긴 턴 −1.2, 홀로 −0.9). 연속 뒤바뀜도 0.8 pt 준다.
  - 세 언어 모두에서 줄었다.
- **ck2000 은 아직 덜 됐다.**
  - agn 이 +2.5 pt 나쁘고 교대를 과하게 낸다(가설/참조 1.29). 헤드가 adapter lr(1e-3)로 LLM hidden 을 크게 흔드는 초반 영향으로 보인다.
  - ck3000 에서 agn 이 r1 과 같아지고 교대 비율도 1.11 로 돌아온다.
- **r1 ck2000(현재 최종) 대비 p9 ck3000:**
  - 이득: δ10 cp −1.8(구간 밖), cp − agn −2.8, δ4 cp −0.9
  - 손해: agn +1.0, EOT 정밀도 −0.016
  - agn 증가는 헤드 탓이 아니다. r1 도 step 2,000 → 3,000 에 같은 만큼 오른다(17.0 → 17.9). 영어·독일어 대화 창을 되풀이한 과적합 탓으로 본다.
- **단일 화자:** Kspon other δ4 가 학습이 길수록 나빠진다(9.3 → 10.0 → 10.3). 나머지는 r1 과 비슷하다.

## 파일
- `stage3_smoke_spk.sh`·`smoke-train.txt`·`smoke-parity.txt`: 2 GPU 스모크
- `train-events.txt`·`spk-head-trend.txt`: 학습 경과·parity·헤드 추이
- `dialogue-eval-vs-r1.txt`(`cmp_p9.py`)·`eval-*.json`: 대화 평가
- `attr-errors-vs-r1.txt`(`cmp_attr_p9.py`)·`attr-errors-*-summary.json`(`attr-errors-r1-ck3000-summary.json` 포함): 오류 유형
- `bootstrap-vs-r1.txt`(`boot_p9.py`): 짝 bootstrap(언어별 포함)
- `regress-vs-r1.txt`: 단일 화자 퇴행
