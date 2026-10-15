---
author_id: tskim
created: 2026-10-15
source_type: source-drop
---

# Stage 3 학습 스크립트·평가 도구 준비와 스모크 (2026-10-15, mxc 호스트)

## 결정 (사용자, 2026-10-15)
1. Stage 3 학습은 `semcommit_train.py` 를 넓히지 않고 **별도 스크립트**로 만든다. 그 파일에는 다른 작업의 커밋 안 된 변경이 있다.
2. 섞는 비율(대화·replay·언어)과 EOT 규칙은 **파일럿에서 비교해 정한다.**

## 코드
- **`experiments/stage3_train.py`**
  - 셋:
    - 대화: `dlg_ko`·`dlg_en`·`dlg_de`. `CstDialogueDataset`, split v2 held-out 제외
    - replay: `rep_ko`·`rep_en`·`rep_de`. Stage 2 와 같은 `AstStreamDataset`, `<SEM_END>`·`<EOT>` 중립
  - 모델: Stage 2 체크포인트 + `add_semantic_tokens(turn=True, reinit_turn=True)`
  - 나머지는 semcommit_train 규약과 같다:
    - 인코더 학습·[56,3] 고정·SpecAugment·replay 증강·varlen
    - 선점 저장·재개 지문(`config.stage3`)·final 재로드 parity
  - 파일럿 선택:
    - `--weights`(셋 가중치)
    - `--eot-mode soft|hard`
    - `--backchannel off|s,토큰`
    - `--decision-weight`
    - `--start-mode grid|mixed`
    - `--replay-files-frac`
- **`vapasr/hf/stage3.py`:**
  - `WeightedRoundRobinLoader`: step 마다 셋을 가중치로 고른다(결정적 smooth weighted round-robin). epoch = 대화 셋을 한 번 도는 길이.
  - `collate_stage3`: 대화 → `collate_cst`, replay → `collate_semcommit`
  - 지문 도우미
- **`vapasr/data/cst_dialogue.py`:** `eot_mode` 를 더했다. hard = p_end ≥ 0.5 → 1, 아니면 0.
- **`slurm/stage3-train-apex.sbatch` + `slurm/submit-stage3-train.sh`:**
  - replay 입력·레시피는 q06-s4-ast 와 같다.
  - 산출은 `/soundai/Model/VAPASR/stage3-<run>` 이다.
  - 실패 재시도 한도는 서브셸 `(MAX_RESTART=N bash …)` 로 준다.
- **`experiments/cst_stage3_eval.py` + `vapasr/data/cst_eval.py`(P4 평가 도구):**
  - held-out 창을 두 슬롯 prefix 로 배치 스트리밍 디코드한다(`decode_batch`, `--verify` = 직렬/배치 대조).
  - 지표:
    - slot(형식대로 A↔A) · cp(두 슬롯 순열 최소, 화자 귀속) · agn(화자 무시) 오류율. 한국어 CER(공백 제외), 영어·독일어 WER
    - 슬롯 뒤집힘 비율
    - EOT 재현율(양성 = p ≥ 0.8)·정밀도(p ≥ 0.5 참조와 짝)·보류 오방출(p ≤ 0.3)·지연
  - 전사 없는 음성 청크에서 나온 가설은 버린다.
- 테스트: `tests/test_stage3_train.py`(5), `tests/test_cst_eval.py`(4), `tests/test_cst_dialogue.py`(11)

## dry-run (`dryrun.sh`, `dryrun-summary.json`)
- 모델 없이 데이터셋만 만들었다. replay 는 words 샤드 5 % 다.

  | 셋 | 크기 | 비고 |
  |---|---|---|
  | dlg_ko | 90,799 창 | held-out 437 대화 제외 |
  | dlg_en | 11,414 창 | held-out 38 대화 제외 |
  | dlg_de | 11,195 창 | held-out 82 대화 제외 |
  | rep_en | 33,990 항목 | |
  | rep_ko | 145,665 항목 | |
  | rep_de | 53,315 항목 | |

- 셋마다 32 항목 × δ4–10 시퀀스 불변식 불일치는 0 이다.
- prefix: `language <SPK_A>Korean <SPK_B>Korean<asr_text><DELAY_10>`

## 학습 스모크 (`smoke-train.sh`, `smoke-train.log`, `smoke-parity.json`)
- 조건: 호스트 GPU 0 한 장, 20 step, 대화 셋마다 300 창, replay 1 % 샤드·셋마다 300 항목, s4 와 같은 레시피(varlen·16,384 토큰 예산)
- 섞기: 기본 가중치에서 대화 0.67 · replay 0.33 이다(steps/epoch 76).
- 손실:
  - 대화 `<EOT>` 손실(loss_turn)이 step 1 의 4.97 에서 step 16–20 의 1.2–1.7 로 빠르게 내려간다.
  - replay 손실은 0.31–0.51 이다.
  - s/step 1.5–4(단일 GPU, 초반 컴파일 포함)
- **final 재로드 parity OK:**
  - 키 누락·값 차이 0, 인코더 저장 키 638
  - 손실 메모리 0.46555 / 재로드 0.46535
  - tokenizer·`config.stage3`·`<EOT>` 등록 일치

## 평가 스모크 (`eval-smoke.sh`, `eval-smoke-summary.json`)
- 위 20 step 체크포인트로 KEC·134-2 held-out 창 16 개씩, δ4·10 을 돌렸다.
- 직렬/배치 대조는 독일어·한국어 × δ4·10 모두 통과했다.
- 20 step 모델은 거의 모든 출력을 슬롯 A 로 낸다. 예: KEC 창 하나에서 B 의 말이 A 로 붙는다. 그래서 slot·cp 오류가 크다.

  | | slot | cp | agn |
  |---|---|---|---|
  | 134-2 δ4 (CER %) | 88.3 | 60.4 | 20.4 |
  | KEC δ4 (WER %) | 77.6 | 57.1 | 25.1 |

- 지표 계산이 의도대로 도는지만 확인한 스모크다. 수치 자체는 의미가 없다.

## 오라클 점검 (평가 도구 관문, 7 코퍼스 held-out 창 200 개 × δ4·10, CPU)
- `--oracle` 은 학습 타깃 시퀀스를 디코드 출력처럼 바꿔 채점한다. 기대값은 slot·cp 오류 0, 양성 EOT 재현율 1 이다.
- **처음에는 KO CER 7–15 %, DE WER 2.5–8 %, EOT 재현율 0.78–0.99 가 나왔다.**
  - 원인: 가설은 전사 없는 음성 청크(마스크)의 토큰을 버리는데, 참조는 창 안 모든 토큰을 썼다.
  - 마스크가 많은 한국어(출력의 27–31 %)일수록 오류가 컸다.
- **고침:** 참조도 그 δ 의 타깃 시퀀스에서 마스크 청크 원소를 뺀 것으로 만든다(`cst_stage3_eval.py`).
- **다시 점검:**
  - 모든 코퍼스·δ 에서 slot·cp 0.00, EOT 재현율 1.000, 지연 δ4 0.30–0.32 s · δ10 0.76–0.80 s(설계값)
  - agn(화자 무시)은 0.1–1.8 % 다. 겹침에서 시각순 정렬(방출 시각 vs 원래 끝 시각)이 조금 달라서다.
  - 정밀도 0.56–0.80, 보류 오방출 1.0 은 오라클이 모든 episode 끝(보류 포함)에 EOT 를 내기 때문이다(soft 타깃 그대로).
- 고치기 전에 시작한 s4 기준점 평가는 `--from-predictions` 로 원출력을 다시 채점한다(디코드 없이).

## 기준점 평가 (진행 중)
- `eval-s4-baseline.sh`: Stage 2 최종(s4)을 같은 평가로 잰다. 7 코퍼스 held-out 창 200 개씩, δ4·10.
- mxc `results/stage3-eval-s4-20261015/` 에 쓴다.
- s4 는 `<EOT>` 가 차단돼 있고 두 슬롯 형식을 배우지 않았다. agn(화자 무시) 오류가 대화체 받아쓰기의 출발점이다.
