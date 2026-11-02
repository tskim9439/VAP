---
type: output
status: active
created: 2026-11-02
updated: 2026-11-02
summary: E13 — 활동 있는 슬롯만 태그(활동 게이트)와 슬롯 블록 번역(<TRANS> 칸) 구현, E11 에서 이어 학습할 제출 준비
sources:
  - '[[output-e11-g3-architecture-20261102]]'
  - '[[output-diarization-integration-20261025]]'
  - '[[output-n3d-branch-20261029]]'
  - '[[output-diarizer-replacement-20261031]]'
raw_authors:
  - tskim
---

# E13 — 활동 게이트 슬롯 블록 + 슬롯 블록 번역 (2026-11-02)

- 요청: 사용자 "번역을 다시 넣어서 학습할 수 있도록 코드를 작성해줘. 그리고 생각해봤는데, 그럼 현재 구조에서는 <SPK_A~D> 까지 활성화할 필요는 없을거 같아. 이 고정 슬롯도 해제해보자."
- 정한 것(사용자 선택, 2026-11-02)
  - 고정 슬롯 해제 = **활동 있는 슬롯 태그만** 넣는다. 태그는 여전히 입력이라 LLM 이 화자를 고르지 않는다(E9 의 장점 유지).
  - 번역 = **받아쓰기 + 번역 둘 다**(Stage 4 와 같다). 창 일부를 번역 모드로 두고, 번역 모드에서는 슬롯마다 전사 칸 뒤에 번역 칸이 온다.
- 배경: E9 부터(10-28) 받아쓰기만 학습했다(F1·E6 까지는 번역 포함). 슬롯 블록 학습 코드는 번역 타깃을 거부했다([[output-e11-g3-architecture-20261102]]).

## 1. 형식
```
청크 k  : [오디오 k] [<SPK_A>] (<ONSET>) (so) ■ [<TRANS>] (그래서) ■          ← B·C·D 는 최근 활동이 없어 태그 없음
청크 k+1: [오디오 k+1] [<SPK_A>] (the) ■ [<TRANS>] ■ [<SPK_C>] (yeah) ■ [<TRANS>] (응) (<EOT>) ■
[ ] 정해진 입력(label 없음)   ( ) LLM 출력   ■ <NEXT_AUDIO> 예측(그 칸 끝 — 다음 입력 위치의 label)
번역 모드가 아닌 창은 [<TRANS>] 칸이 없다(E9 와 같고 태그만 게이트).
```
- **활동 게이트**(`--slot-gate N`, 기본 12 = 태그 활동 칸 수)
  - 학습: 라운드 k 에 슬롯 s 태그를 넣는 것은 정답 활동(전사 없는 발화 포함)이 창 [k − N + 1, k] 에 있을 때와 그 라운드에 출력이 있을 때다(`gate_rounds`). 태그 임베딩에 더하는 활동 벡터와 같은 칸이다.
  - 출력(단어 = 소리 + δ, <ONSET> = 시작 + 4, <EOT> = 끝 + δ, δ ≤ 10)은 늘 활동 창 안이라, 정답 활동이면 출력 때문에 따로 여는 일이 없다.
  - 추론: 예측 활동(N3D·학생) 창 ∪ 열린 발화(그 슬롯이 낸 <ONSET> 수 > <EOT> 수 — 커밋 f17e9ca). 태그가 하나도 없는 라운드는 오디오 하나다.
    - 수로 세는 까닭: 번역 모드에서는 발화의 <EOT> 가 번역 꼬리 뒤(번역 칸)라 다음 발화의 <ONSET>(전사 칸)보다 늦을 수 있다(서버 dry-run 의 dlg_mt 예).
  - `--gate-fa P`(기본 0.1): 학습 창의 슬롯마다 그 확률로 말 없는 0.3–3.2 s 구간 하나를 더 연다(예측 활동의 헛 검출 — 태그 뒤 멈춤을 배운다).
- **번역 칸**
  - 번역 모드 창은 열린 슬롯마다 [태그] 전사 칸 뒤 [<TRANS>] 입력과 번역 출력이 온다. <TRANS> 도 정해진 입력이다(label 없음).
  - 번역 타깃·목표 언어는 Stage 4 와 같다(대화 번역 타깃, 한·독 → 영, 영 → 독·한, 교차 언어 합성은 상대 언어). 번역 시각(stable prefix 확정)은 episode 안으로 잘라 끝 + δ 안에 나간다.
  - <EOT> 는 그 episode 의 마지막 행에 번역이 있으면 번역 흐름, 없으면 전사 흐름에 둔다(같은 청크 — 번역 꼬리 뒤).
  - 가림: 참조 번역이 없는 행의 방출 청크는 그 슬롯 번역 칸 label, 되살린 행의 번역은 입력만(label −100), 마지막 행에 번역이 없는 episode 의 <EOT> label.

## 2. 구현 (커밋 dd50274)
- 데이터 `vapasr/data/cst_dialogue.py`
  - `gate_rounds`, `block_streams(trans_of, n_out)`(번역 흐름 = D + 슬롯), `build_block_sequence(gate, trans_id, tr_input, tr_hide, eot_hide)`, `block_problems(trans_id)`(게이트 배치·번역 다시 읽기·칸 뒤 멈춤).
  - `CstDialogueDataset(slot_gate, gate_fa)`, 슬롯 블록 번역 모드(학습 rng 가 ast_frac 확률로 목표를 고른다), `est_lens`(게이트·번역 몫).
  - 끄면(`slot_gate` 0, 번역 없음) E9 시퀀스·난수 그대로다.
- 디코드
  - `batch_decode.BlockState(gate_lags, act, trans_id, begin_id, eot_id)`, `decode_blocks_batch(trans)`, `block_stops(trans)`(<TRANS> 도 멈춤), `block_gate`(config.slot_gate).
  - `realtime.realtime_block_decode(trans)`(칸을 태그처럼 한 forward 로 미리 넣고 되돌아간다), `long_session.long_block_decode(trans)`.
  - 게이트는 `config.slot_gate` 로 켜진다(받아쓰기 평가 `cst_stage3_eval` 는 그대로 쓰면 된다).
- 평가: `cst_eval.emitted_from_blocks(trans_id)`·`parse_emitted4(begin_id)`, `cst_stage4_eval`(슬롯 블록 모델이면 번역 칸 디코드·오라클).
- 학습
  - `stage3_train --slot-gate N --gate-fa P`(config.slot_gate, 지문 키 slot_gate). 슬롯 블록에서 `--dlg-trans`·`--sim-trans-frac` 를 받는다. 학습 전 검사에 학습 rng 경로(헛 구간·번역 추첨)를 더했다.
  - `slurm/submit-q17-f1.sh --blocks-trans`: 번역 타깃·번역 행 앞 공백·replay 번역(80 %)은 켜고, 슬롯 블록이 받지 않는 교대/머무름 가중·EOT-문맥 dropout 은 뺀다.
  - `slurm/submit-q17-e13.sh`: E11 final 에서 E11 레시피(N3D 활동, 인코더 고정, 선언 수 범위, global slots 0.5) + 게이트 12 + 헛 구간 0.1 + 번역, 5,000 step. 평가 묶음 `q17-e13*`(AMI 창·회의 통째 받아쓰기).
- 시험
  - 새 시험 `tests/test_slot_gate_trans.py` 10 개가 통과했다.
    - 게이트 배치, 출력·label 이 E9 와 같음, 헛 구간은 학습 rng 에서만 연다.
    - 번역 칸 토큰이 Stage 4 네 흐름 정답과 같음, 가림 규칙.
    - `BlockState`(게이트·번역 칸)가 정답 시퀀스를 그대로 되짚고, 추측 prefill 디코드가 한 토큰씩 디코드와 같다.
  - 기존 시험: 슬롯 블록 20 개와 대화·Stage 3 120 개가 통과했다. 바뀐 모듈을 쓰는 시험 315 개도 통과했다(맥 환경에 rapidfuzz 가 없는 2 개 제외).
  - 서버: 추적 코드 534 개 md5 일치.

## 3. 서버 데이터 점검 (2026-11-02, 호스트 CPU)
- `raw/sources/experiments/2026-11-02-e13/e13_dry.sh`: `submit-q17-e13.sh --dry-run` 이 만든 인자 그대로 `stage3_train --dry-run`(셋마다 32 창 × δ 4–10).
- 19 셋(대화 7 그룹·합성 9·replay 3) 모두 불일치 0 이다(DRY_EXIT 0, 11:50–12:28). 번역 모드(목표마다)와 학습 rng 경로(헛 구간·번역 추첨)를 함께 봤다.
- 예 시퀀스(dlg_mt 번역): 말이 없는 동안은 오디오만 오고, 말하는 슬롯에만 [태그][<TRANS>] 가 붙는다. 번역은 문장 끝에 몰려 나오고 <EOT> 가 번역 칸 끝에 온다.
- swbd·nia23mt 는 번역 타깃이 없어 늘 받아쓰기 모드다(Stage 4 와 같은 규칙).

## 4. 다음
- 제출(사용자): `(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-e13.sh --partition low_p_hpc)`(게이트만은 `--no-trans`).
- 번역 평가 묶음(`cst_stage4_eval` — 대화 held-out·TAXI 등)은 학습이 도는 동안 만든다.
- 걸리는 점
  - 추론 게이트는 예측 활동을 따른다. 화자 분리기가 놓친 말은 그 슬롯 태그가 없어 받아쓸 수 없다(열린 발화 규칙이 이미 시작한 말은 지킨다). 대신 LLM 이 활동을 어기고 다른 슬롯에 쓰는 일(E11 7–12 %)은 구조적으로 줄어든다.
  - E11 에서 받아쓰기만 이어 학습한 E12 대조는 agn 이 +0.7·+0.9 나빠졌다([[output-n3d-branch-20261029]] 6.8 절). 번역을 더한 이어 학습의 받아쓰기 손해를 같은 기준으로 본다.
