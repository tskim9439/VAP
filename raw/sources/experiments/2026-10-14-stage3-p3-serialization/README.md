---
author_id: tskim
created: 2026-10-14
source_type: source-drop
---

# Stage 3 P3 직렬화 — 데이터셋·모델 연결·전 코퍼스 dry-run·GPU 스모크 (2026-10-14, mxc)

## 코드
- `vapasr/data/cst_dialogue.py`(신규): Stage 3 시퀀스와 데이터셋.
  - `cst_head_text`
  - `build_cst_sequence`
  - `CstDialogueDataset`(DialogueWindowDataset 상속)
  - `collate_cst`
- `vapasr/hf/modeling_vapasr.py`:
  - `forward(sem_only_mask=)`: `<SEM_END>` logit 만 뺀다.
  - `add_semantic_tokens(turn=True, reinit_turn=True)`: 예약만 돼 있던 `<EOT>` 행을 다시 초기화한다.
- `vapasr/hf/trainer.py`: `labels_alt`·`soft_w_full`·`sem_only_mask` 를 모델로 넘긴다.
- `vapasr/hf/packing.py`: `ROW_KEYS` 에 `sem_only_mask` 를 더했다.
- 테스트 `tests/test_cst_dialogue.py` 10 개
  - 위 4 개 파일과 함께 바꾼 관련 테스트 53 개가 mxc 에서 통과했다.
  - 맥에서는 `test_packing` 3 개가 옛 transformers 때문에 import 오류로 실패한다. 이번 변경과 무관하다.
- dry-run: `experiments/cst_stage3_dryrun.py`

## 형식
- **prefix:** `<assistant 머리> language <SPK_A>{L} <SPK_B>{L}<asr_text><DELAY_d>`
  - 같은 언어는 창 안에서 먼저 말한 화자가 A 다. 한 화자만 있어도 두 슬롯을 선언한다.
  - tokenizer 로 되읽으면 `language <SPK_A>German <SPK_B>German<asr_text>` 이다. 슬롯 태그는 특수 토큰 하나씩(151707·151708)이다.
- **본문:**
  - Stage 1·2 버킷(⌊t/80 ms⌋ + δ), 스트림 0 = A, 1 = B
  - 화자 태그는 첫 방출과 화자가 바뀔 때만 낸다.
- **`<EOT>`:**
  - episode(같은 화자 발화를 gap < 0.25 s 로 병합)마다 그 화자의 마지막 토큰 뒤에 둔다.
  - 청크 = max(마지막 토큰 청크, ⌊(end + 0.24)/0.08⌋). Phase 2 와 같다.
  - 창 끝을 넘는 episode 와 창 안 토큰이 없는 episode 는 `<EOT>` 를 내지 않는다.
- **soft label:**
  - 결과는 대화 전체 시간축으로 정한다. Phase 2 는 창 안 episode 만 봐서 창 끝 근처 교대를 놓쳤다.
  - 결정 위치는 `<EOT>` 다. 그 `<EOT>` 를 내려고 화자 태그를 새로 연 경우는 태그 위치다.
  - label = 결정 토큰(p_end) / `<EOT>` 뒤 토큰(1 − p_end). 결정 위치 가중은 `decision_weight` 다.
- **flush:** `build_interleaved_events` 는 flush(창 끝 뒤 `<EMPTY_AUDIO>` 라운드)에 화자 태그를 넣지 않는다(단일 화자 시절 규약). Stage 3 는 화자가 바뀌는 곳에 다시 넣는다(`tag_flush`).
- **마스크:** 전사 없는 음성 청크는 label −100 이다(Phase 2 규칙). 3 화자 이상 창은 버린다.
- **`<SEM_END>`:** 중립(`sem_only_mask`)이다. 단일 화자 replay 행은 기존 `sem_mask`(SEM·EOT 둘 다 중립)를 쓴다.

## 전 코퍼스 dry-run (`dryrun.json`; mxc `results/cst-stage3-dryrun-20261014/`)
- split v2 held-out 을 빼고 20–40 s 창(hop 10 s), 창마다 δ ∈ {4..10} 하나로 검사했다.
- 검사 항목: 태그 규칙, 슬롯별 토큰·EOT 순서, soft 결정 위치, EOT 수
- **113,389 창 전부 문제 0, `all_ok` True.** 오디오 표본(코퍼스당 20 창) 길이도 모두 맞았다.

| 코퍼스 | 대화(제외) | 창 | 1 화자 창 | 시퀀스 길이 p50·max | 창당 텍스트 p50 | 창당 EOT·soft | 출력 label 마스크 p50·p90 |
|---|---|---|---|---|---|---|---|
| 71631 | 698 (59) | 20,540 | 6.9 % | 840 · 1,265 | 59 | 5.9 · 4.9 | 27 % · 60 % |
| 134-1 | 1,337 (155) | 26,465 | 8.2 % | 854 · 1,282 | 76 | 6.9 · 5.4 | 27 % · 65 % |
| 134-2 | 1,426 (223) | 43,749 | 6.4 % | 830 · 1,241 | 42 | 5.2 · 4.1 | 31 % · 59 % |
| otoSpeech | 382 (38) | 11,414 | 12.3 % | 849 · 1,547 | 66 | 7.2 · 7.1 | 0 · 0 |
| KEC | 150 (21) | 2,165 | 6.2 % | 898 · 1,391 | 119 | 9.5 · 8.7 | 8 % · 28 % |
| ZTC | 334 (39) | 5,206 | 5.1 % | 863 · 1,230 | 85 | 12.3 · 10.9 | 11 % · 26 % |
| NSC | 254 (22) | 3,850 | 0.9 % | 861 · 1,230 | 75 | 6.8 · 5.9 | 21 % · 43 % |

- 3 화자 이상 창은 0 이다. 일곱 코퍼스 모두 2 인 대화다.
- 최대 길이는 1,547 토큰이다. Stage 2 의 packing 예산 16,384 의 1/10 이다.
- 태그 위치 결정(다른 화자가 먼저 말해 EOT 앞에 태그를 새로 연 경우)은 soft 결정의 0.1–0.5 % 다.
- **soft p_end 분포**(결정 위치 기준):

  | 코퍼스 | 0.0 | 0.3 | 0.5 | 0.8 | 1.0 |
  |---|---|---|---|---|---|
  | 71631 | 26.7 % | 18.0 % | 21.5 % | 10.5 % | 23.3 % |
  | 134-1 | 28.5 % | 13.1 % | 29.6 % | 4.8 % | 24.1 % |
  | 134-2 | 20.1 % | 18.3 % | 21.5 % | 13.8 % | 26.3 % |
  | otoSpeech | 28.4 % | 10.5 % | 21.3 % | 13.3 % | 26.5 % |
  | KEC | 16.0 % | 10.7 % | 45.3 % | 2.4 % | 25.7 % |
  | ZTC | 25.3 % | 15.7 % | 37.7 % | 2.9 % | 18.4 % |
  | NSC | 13.4 % | 8.3 % | 35.7 % | 3.3 % | 39.3 % |

## 맞장구와 EOT 결과 (`bc_outcomes.py`, 코퍼스당 150 대화)
- 1 s 이하이면서 3 토큰 이하인 episode 를 맞장구로 봤다. 이것을 '상대가 말함'에서 빼면 결과 분포가 바뀐다.
  - both(0.5): 24–50 % → 11–32 %. 대부분 hold 로 옮겨 간다.
  - 독일어 사람 전사(KEC·ZTC)는 맞장구가 많아 both 가 가장 크게 준다: KEC 0.501 → 0.322, ZTC 0.471 → 0.228.

  | 코퍼스 | shift | silence | both | hold_long | hold_short |
  |---|---|---|---|---|---|
  | 71631 지금 → 맞장구 제외 | 0.261 → 0.240 | 0.046 → 0.068 | 0.394 → 0.287 | 0.112 → 0.169 | 0.177 → 0.228 |
  | 134-2 | 0.315 → 0.265 | 0.090 → 0.140 | 0.282 → 0.178 | 0.142 → 0.205 | 0.166 → 0.207 |
  | otoSpeech | 0.208 → 0.177 | 0.078 → 0.109 | 0.241 → 0.112 | 0.096 → 0.141 | 0.371 → 0.455 |
  | KEC | 0.210 → 0.196 | 0.009 → 0.023 | 0.501 → 0.322 | 0.089 → 0.170 | 0.183 → 0.281 |
  | ZTC | 0.132 → 0.100 | 0.020 → 0.053 | 0.471 → 0.228 | 0.117 → 0.225 | 0.252 → 0.387 |
  | NSC | 0.391 → 0.366 | 0.026 → 0.051 | 0.419 → 0.290 | 0.059 → 0.130 | 0.095 → 0.154 |

- 이 비교를 위해 `CstDialogueDataset(backchannel=(1.0, 3))` 선택을 넣었다(기본 끔 = Phase 2 규칙). 켜면:
  - 맞장구 episode 는 상대의 결과 판정에서 빠진다.
  - 맞장구 자신은 `<EOT>` 를 내지 않는다. 전사는 그대로다.
  - 켤지는 P5 에서 정한다.

## GPU 스모크 (`gpu_smoke.py`, 호스트 GPU 4, s4 final, 저장 없음)
- **`<EOT>` 켜기:** `add_semantic_tokens(turn=True, reinit_turn=True)` 뒤 `<EOT>` 가 차단 해제되고 `sem_registry` 에 들어갔다. 예약 목록에서 빠졌고 행을 다시 초기화했다.
- **forward·backward:** KEC·134-2 Stage 3 배치(6 창)로 bf16 autocast·인코더 학습 모드에서 돌렸다.

  | | KEC | 134-2 |
  |---|---|---|
  | loss | 1.94 | 1.49 |
  | loss_turn | 6.1 | 6.9 |
  | n_turn = n_soft | 50 | 29 |

  - 처음에는 `<EOT>` 확률이 낮아 loss_turn 이 높다.
  - grad 는 유한하다. `<EOT>` 행에는 grad 가 있고 `<SEM_END>` 행은 grad 0 이다(sem_only_mask 중립).
- **패딩 vs packing:** eval 모드 손실 차이가 0.0014(KEC)·0.0005(134-2)다(bf16 수준). train 모드 차이 0.057 은 인코더 증강·dropout 의 무작위성이다.

## 남은 것
- 학습 스크립트 연결(P5):
  - `experiments/semcommit_train.py` 에 Stage 3 셋(CstDialogueDataset·collate_cst)과 replay 셋을 넣는다.
  - `--turn-end` 대신 `add_semantic_tokens(turn=True, reinit_turn=True)` + `turn_weight` 로 켠다.
  - 이 파일에는 다른 작업(Codex)의 커밋 안 된 변경이 있어 이번에 손대지 않았다.
- 평가(P4): 두 슬롯 prefix 로 스트리밍 디코드, ORC-WER·cpWER·EOT 지표
