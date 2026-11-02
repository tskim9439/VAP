---
type: output
status: active
created: 2026-11-02
updated: 2026-11-02
summary: E13 — 활동 게이트 슬롯 태그와 슬롯 블록 번역(<TRANS> 칸) 구현·제출, AI Hub 031 폴더 이름 변경으로 빠진 replay 경로 수정
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

## 4. 제출과 평가 (2026-11-02)
- 사용자 제출: 학습 job 99484(hpc 2 노드, 5,000 step), AMI 받아쓰기 평가 job 99485(q17-e13 — 학습 뒤), 번역 평가 job 99512(q17-e13-trans — 학습 뒤).
- 번역 평가 묶음 `q17-e13-trans`: held-out 대화 창 번역(v3_eval_task.sh dlg — cst_stage4_eval 슬롯 블록 번역 디코드, δ4·10, 코퍼스마다 100 창)을 E13 final 과 E11 final 로.
- 경로 스모크(`raw/sources/experiments/2026-11-02-e13/trans_smoke.sh`, 호스트 GPU 6, E11 final, nsc·otoSpeech 각 2 창, δ10): 끝까지 돌았고 받아쓰기는 정상(독 WER 28.8, 영 20.9)이다.
  - 번역 칸에서 낸 토큰은 0 이다. E11 은 E9 부터 받아쓰기만 학습해 <TRANS> 칸을 본 적이 없어 칸마다 바로 멈춘다. 그래서 99512 의 E11 줄은 기준선 0 이다.
  - 견줄 값은 번역을 학습한 마지막 판 E6(태그를 LLM 이 내는 형식)의 같은 평가다. d10 기준 한→영 BLEU 30–36, 독→영 40–58, 영→독 41, 영→한 44(`results/stage4-eval-q17-e6-final/dlg-summary.log`).

## 5. 다음
- 교차 언어(합성 xl·TAXI) 번역 평가는 슬롯 순서를 맞춰야 한다. 학습은 언어로 정한 슬롯에 전문가 활동을 정답 자리로 맞추지만, 추론의 N3D 채널은 도착 순서라 prefix 의 슬롯 언어를 도착 순서로 정해야 한다(아직 하지 않았다).
- 걸리는 점
  - 추론 게이트는 예측 활동을 따른다. 화자 분리기가 놓친 말은 그 슬롯 태그가 없어 받아쓸 수 없다(열린 발화 규칙이 이미 시작한 말은 지킨다). 대신 LLM 이 활동을 어기고 다른 슬롯에 쓰는 일(E11 7–12 %)은 구조적으로 줄어든다.
  - E11 에서 받아쓰기만 이어 학습한 E12 대조는 agn 이 +0.7·+0.9 나빠졌다([[output-n3d-branch-20261029]] 6.8 절). 번역을 더한 이어 학습의 받아쓰기 손해를 같은 기준으로 본다.

## 6. replay 오디오 경로 — AI Hub 031 폴더 이름 변경 (2026-11-02)
- 증상: E13 학습 로그(99484)의 오디오 조립 실패 320 줄이 모두 replay `ahbc-train-*` 항목(AI Hub 031 방송 한국어, `asr-words-v1/parts/asr-aihub-bc-train-*`)의 `FileNotFoundError` 다. 로그는 작업 프로세스마다 5 줄까지만 찍어 실제 실패는 더 많다. 실패한 항목은 이웃 항목으로 바뀌어 학습은 계속된다.
- 원인: 공용 DB 의 `/soundai/DB/raw/aihub/031.방송콘텐츠 한국어-영어 통·번역 음성 데이터` 가 `031.broadcasting_koen` 으로 바뀌었다(옆에 `031_trash` 10-30 14:12, 새 폴더 11-01 11:44).
  - 새 폴더는 안쪽 한글 이름이 모두 NFD 다. 옛 manifest 경로는 맨 위 폴더만 NFD 이고 안쪽은 NFC 라 이름만 바꿔서는 찾지 못한다.
  - 새 폴더에는 INNORIX 내려받기 조각(`*.zip.irx*`, 마지막 변경 11-01 21:41)이 남아 있다. 다시 내려받는 중이거나 멈춘 상태로 보인다(관리자 확인 필요).
- 영향(학습 로그의 031 실패 수로 확인):
  - 10-30 18:49 이후 시작한 학습 전부에서 ahbc 가 빠졌다. E12 다섯 판(e12 98836, -shuf 98956, -remap 98958, -noise 98960, -ctrl 98962)과 E13 99484 다.
  - E11 은 실패 1 건으로 영향이 없다.
  - E13 의 EN·KO replay words 파일 827 개 중 142 개가 ahbc 다.
- 수정(커밋 a87d4af): `vapasr/data/archive.py` 의 `MOVED`(옛 → 새 폴더)와 `resolve_path` 를 `read_member`(압축)·`load_utt_audio`(파일)가 부른다.
  - 옛 폴더는 NFC 로 비교하고, 나머지 경로는 그대로·NFD·NFC 중 있는 것을 쓴다.
  - 옮긴 폴더 밖 경로는 I/O 없이 그대로 지나간다. 찾은 경로만 기억한다.
  - 공용 폴더는 고치거나 링크를 걸지 않았다.
- 서버 점검(`raw/sources/experiments/2026-11-02-e13/moved031_check.sh`, 호스트 CPU, 15:03–15:08)
  - Linux 에서 `tests/test_moved_paths.py` 4 개가 통과했다(정규형이 실제로 갈리는 파일 시스템).
  - replay manifest 의 ahbc 431,447 줄이 가리키는 압축 14 개 중 13 개를 새 폴더에서 찾았다. 압축마다 항목 하나를 학습 로더 경로(`assemble_stream`)로 읽었고 모두 정상이다.
  - `TS_한-영(미국)_한국어음성_교양.zip` 은 새 폴더에도 `031_trash` 에도 없다. 이 압축의 14,273 줄(3.3 %)은 계속 이웃 항목으로 바뀐다. 파일이 생기면 그때부터 저절로 찾는다.
- E13 반영: 돌고 있는 99484 는 이미 옛 코드를 import 했다. 선점 뒤 다시 시작할 때부터 ahbc 가 다시 섞인다(학습 도중에 replay 구성이 바뀐다).
