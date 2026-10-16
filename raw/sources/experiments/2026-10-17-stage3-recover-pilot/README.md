---
author_id: tskim
created: 2026-10-17
source_type: source-drop
---

# Stage 3 정체 회복 학습(태그 뒤집기) 구현과 파일럿 준비 (2026-10-17, mxc 호스트)

## 배경
- 사용자 요청(2026-10-17): "1번 구현해서 파일럿 준비해줘". 1번은 화자 귀속 개선 후보 1 순위인 정체 회복 학습이다.
- 근거는 오류 유형 분해([[2026-10-16-stage3-main-r1]], ck2000 δ10)다.
  - 오배정 10.2 pt 의 73 % 가 턴을 통째로 다른 화자에 붙인 것이다.
  - 통째 오배정의 60 %(4.5 pt)는 연속된 턴이 함께 뒤바뀐 것이고, 그 67 % 가 창 끝까지 이어진다.
  - 교대 하나를 놓치거나 더 내면 매핑이 뒤집힌 채 되돌아오지 않는다.
  - 학습은 정답 태그를 문맥으로만 주므로(teacher forcing) 틀린 매핑에서 회복하는 예가 없다.

## 구현 (`vapasr/data/cst_dialogue.py`, 기본 끔)
- **`Recover(p, boundary_frac=0.5, max_lag=8, max_boundaries=2)`:** 대화 창마다 확률 p 로 뒤집기 구간 하나를 넣는다.
- **`pick_flip`:** 실제 청크(flush 밖)의 방출 원소 순서에서 구간 [n0, n1) 을 고른다.
  - 회복 종류를 먼저 고른다(boundary_frac). 그 종류로 회복할 수 있는 시작만 후보로 둔다.
  - 시작 n0 은 반반이다.
    - missed: 교대 지점이다. 그 태그를 빼 직전 화자로 이어 쓴 것처럼 만든다.
    - spurious: run 안이다. 헛 태그를 넣은 것처럼 만든다.
  - 회복 n1:
    - boundary: n0 뒤 1–2 번째 교대 지점이다. 문맥 슬롯이 이미 그 화자라 **태그 없이** 이어 쓰는 것이 정답이다. 목소리로 정체를 판단해야 한다.
    - inrun: n0 뒤 8 원소 안의 run 안이다. **바로잡는 태그**가 정답이다.
  - n0·n1 은 텍스트 토큰이고 마스크 청크 밖이다.
- **`retag`:** 구간 원소의 문맥 슬롯을 반대로 두고 태그를 다시 단다. 규칙은 원래와 같다(첫 방출 앞, 슬롯이 바뀔 때). 뒤집지 않으면 원래 시퀀스와 같다.
- **label:** 구간 [첫 원소(앞 헛 태그 포함), 회복 지점의 첫 출력) 의 출력 label 은 −100 이다(텍스트·태그·`<NEXT_AUDIO>`·EOT 결정).
  - 틀린 문맥에서 이어 쓰는 법은 가르치지 않는다. 회복 지점만 정답이다.
- **적용 범위:** `__getitem__`(학습)만 rng 를 넘긴다. 평가(`cst_stage3_eval.window_ref`)와 학습 전 점검은 뒤집지 않는다.
- **`problems_of`:** 뒤집은 시퀀스를 따로 검사한다.
  - 슬롯별 다시 읽기는 뒤집은 문맥 기준으로 한다.
  - 구간 출력 label 은 모두 −100 이어야 한다.
  - 회복 지점 정답: inrun 이면 그 화자 태그, boundary 면 태그 없이 그 원소다.
- **`experiments/stage3_train.py`:**
  - 옵션 `--recover-p`·`--recover-boundary-frac`·`--recover-max-lag`·`--recover-max-boundaries`
  - 학습 전 점검에 뒤집은 시퀀스(p = 1)를 더했다.
  - 지문 `config.stage3.recover` 는 켤 때만 넣는다. 끄면 이전 run 의 지문이 그대로다.
  - 로그 `flip_per_dlg_step`, dry-run 에 뒤집은 예를 찍는다.
- **`slurm/submit-stage3-train.sh`:** `--recover-p P`(기본 0)
- **테스트(`tests/test_cst_dialogue.py`):**
  - retag 항등
  - missed/boundary · missed/inrun · spurious/boundary 시퀀스와 label
  - pick_flip 의 마스크·텍스트·확률 조건
  - 데이터셋: rng 없으면 원래와 같음, 네 종류 모두 나옴, 점검 통과, collate `n_flip`
  - 한 화자 창 점검
  - 단어 첫 토큰 제한: 시작·회복 지점은 단어 첫 토큰(디코드가 공백으로 시작)만 쓴다.
    - 첫 dry-run 에서 한국어 헛 태그·회복 태그가 단어 중간(' 한' `<SPK_B>` '단')에 들어갔다. 실제 오류는 단어 경계에서 나므로 고쳤다.

## 실데이터 점검 (`recover_stats.py`, `recover-stats.txt`)
- 조건: 학습 대화 셋 창 800 개 × δ4·10, p = 1
- **뒤집기 성공(창 안에 후보가 있음):** KO 96 %, EN·DE 100 %
- **시퀀스 점검:** 불일치 0. 시작·회복 지점이 단어 첫 토큰이 아닌 경우도 0 이다.
- **종류 비율:**

  | 셋 | missed/boundary | missed/inrun | spurious/boundary | spurious/inrun |
  |---|---|---|---|---|
  | KO | 0.14 | 0.19 | 0.22 | 0.45 |
  | EN | 0.23 | 0.22 | 0.25 | 0.30 |
  | DE | 0.21 | 0.23 | 0.24 | 0.33 |

  - 경계 회복은 KO 36 %, EN 47 %, DE 45 % 다. 한국어는 턴이 길고 마스크가 많아 경계 후보가 없는 창이 많다.
  - 첫 판(끝 종류를 나중에 고름)은 KO 23 % 였다. 끝 종류를 먼저 고르게 바꿨다.
- **가려지는 label:** 뒤집은 창에서 평균 10–13 %(중앙값 5–7 %)다. p = 0.5 면 대화 label 의 약 5–6 % 다.
- 구간 길이는 중앙값 5 원소, p90 17–22 원소다.

## dry-run (CPU, 대화 셋만, 학습 전 점검 64 항목 × δ4–10)
- 점검은 원래 시퀀스와 뒤집은 시퀀스(p = 1) 모두 불일치 0 이다.
- 단어 첫 토큰 판(`dry-run.txt`)에서도 점검은 불일치 0 이다. 헛 태그는 단어 경계 앞에 놓인다.
  - dlg_en `spurious/inrun`: ' out' ' of' 뒤 `<SPK_B>`[m] ' my'[m] … ' too'[m], 회복 지점 `<SPK_A>` ' often'
  - dlg_de `spurious/inrun`: ' jeden' 뒤 `<SPK_B>`[m] ' fall'[m] ' den'[m] ' arm'[m], 회복 지점 `<SPK_A>` ' aus'
  - dlg_ko `spurious/inrun`: `<SPK_B>`[m] ' 한단말이야' `<EOT>`[m] 뒤 긴 무음의 `<NEXT_AUDIO>` 도 가려진다. 같은 화자가 다시 말할 때 `<SPK_A>` 로 회복한다. 무음이 긴 구간은 가려지는 label 이 많다.
- 제한 전 판(`dry-run-before-word-start.txt`)에서는 한국어 태그가 단어 중간에 들어갔다.

## GPU 스모크 (호스트 GPU 1, 20 step, `stage3_smoke_recover.sh`, `smoke-train.txt`, `smoke-parity.txt`)
- 레시피: r1(p7)과 같다. 맞장구 1.0,3, 대화 KO 2 · EN 0.7 · DE 0.7, replay 각 0.75, lr 4e-5. 여기에 `--recover-p 0.5` 를 더했다.
- 학습 전 점검: 대화 셋 원래·뒤집은 시퀀스와 replay 모두 불일치 0
- `flip_per_dlg_step` 은 5–14 다(대화 배치의 약 절반).
- 손실은 정상 범위다. s4 init 이라 `<EOT>` 손실이 4.9 → 3.1 로 내려가는 초반이다.
- **final 재로드 parity OK.** 지문 `config.stage3.recover` = {p 0.5, boundary_frac 0.5, max_lag 8, max_boundaries 2}
- 스모크는 단어 첫 토큰 제한을 mxc 에 올리던 무렵 시작해, 어느 판으로 돌았는지 확실하지 않다. 제한은 후보 위치만 바꾸고 학습 경로는 같다.

## 파일럿 st3-p8-recover (제출 대기 — 사용자)
- **비교 대상:** q06-st3-r1 checkpoint-2000(Stage 3 최종). 레시피를 똑같이 두고 `--recover-p 0.5` 만 더한다.
- **같은 LR 일정을 쓴다:** r1 과 같이 `--max-steps 4000`(cosine)·2 노드로 내고 checkpoint-2000 끼리 비교한다.
  - 2,000 step 으로 줄이면 cosine 이 끝까지 내려가 r1 ck2000 과 일정이 달라진다.
  - checkpoint-3000·final 은 정체 회복이 step 2,000 뒤의 나빠짐(가설 교대 증가)을 줄이는지 본다.
- **제출 명령(mxc, 사용자):**

  ```bash
  cd /soundai/users/tskim/VAPKT
  (MAX_RESTART=5 bash slurm/submit-stage3-train.sh --run st3-p8-recover --partition apex --nodes 2 --backchannel 1.0,3 --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,rep_ko:0.75,rep_en:0.75,rep_de:0.75 --lr 4e-5 --max-steps 4000 --recover-p 0.5)
  ```

  - 산출은 `/soundai/Model/VAPASR/stage3-st3-p8-recover` 이다. `--dry-run` 으로 명령이 맞는 것을 확인했다.
- **평가 큐(걸어 둠, 호스트 GPU 1, `ckpt_eval_queue.sh` — 오류 유형 분해 단계 추가):**
  - checkpoint-2000 · 3000 · final 이 생기면 차례로 돈다.
  - 대화 평가(맞장구 참조) → 오류 유형 분해(`cst_attr_errors.py`) → 단일 화자 축약 퇴행
  - 로그는 mxc `results/stage3-regress/queue-p8-gpu1.log` 다.
- **판정 기준(ck2000 대 r1 ck2000, δ10):**
  - 1 차: 통째 오배정의 이어짐(4.5 pt)과 cp(28.6)가 준다.
  - 같이 볼 것:
    - agn(17.0)이 나빠지지 않는다. 구간 label 을 가려 전사 신호가 약 5 % 준다.
    - EOT 정밀도·보류 오방출이 나빠지지 않는다.
    - 단일 화자 퇴행이 r1 과 비슷하다.

## 파일
- `recover_stats.py`·`recover-stats.txt`: 실데이터 점검
- `stage3_dry_recover.sh`·`dry-run.txt`(단어 첫 토큰 판)·`dry-run-before-word-start.txt`: dry-run
- `stage3_smoke_recover.sh`·`smoke-train.txt`·`smoke-parity.txt`: GPU 스모크
- `ckpt_eval_queue.sh`: 평가 큐(오류 유형 분해 포함)
