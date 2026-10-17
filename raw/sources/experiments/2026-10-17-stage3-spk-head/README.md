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

## 파일럿 st3-p9-spkhead (제출 대기 — 사용자)
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

## 파일
- `stage3_smoke_spk.sh`·`smoke-train.txt`·`smoke-parity.txt`: 2 GPU 스모크
