---
author_id: tskim
created: 2026-10-13
source_type: source-drop
---

# q06-s4-ast(final) 학습·평가 (2026-10-12 19:25 – 2026-10-13 23:2x)

## 학습
- 레시피: `slurm/submit-q06-s4-ast.sh`
  - q06-s3-ast/final 에서 시작, 2 epoch, 14,860 step
  - s3 데이터·레시피에 Common Voice 4 영어 words 232,845 행과 그 단조 번역 타깃(영→독 230,083, 영→한 228,474 행)을 더했다
  - 번역 타깃 639 만 행, 영어 셋 689,738 항목·2,487 h
- 실행 경과(모두 사용자 제출):

  | job | 파티션 | 경과 |
  |---|---|---|
  | 85484 | hpc | 로딩 중 선점 3 번. 20:26 시작분이 step 5,640 에서 rank 0 DataLoader worker SIGKILL 로 23:45 실패. 선점이 재시도 한도(3)를 다 써서 끝났다([[2026-10-13-query-s4-train-failure]]) |
  | 85823 | hpc(MAX_RESTART=8) | checkpoint-5000 에서 재개, step 8,160 까지. 10:55 사용자 취소(low_p_hpc 로 옮겨 달라는 요청) |
  | 85952 | low_p_hpc(MAX_RESTART=20) | checkpoint-8000 에서 재개. 재시작 14 번(로그의 선점 신호는 12 번, 대부분 로딩 중). 학습 재개는 8000·9000·11000·12000 네 번이다. 두 번은 선점 저장이 끊겨 불완전한 checkpoint-11533·13000 이 남았고, 각각 11000·12000 에서 이어 갔다. 20:35 완료 |

  `train-85952-events.txt` 는 시작·선점·재개 기록과 parity.json 이다.
- 결과: parity True(메모리 loss 0.32976, 재로드 0.32944), 인코더 638 키 저장
- 속도: 평소 0.55–0.67 s/step. 2026-10-12 21:35–22:45 와 10-13 오전 일부 구간에서 2–4 s/step(원인 미확인, 공유 저장소 경합 추정).
- 노드 메모리(사용자 확인, hpc-30, 10-13 10:3x): 1,820 GB 중 사용 1,289 GB, 가용 531 GB. SIGKILL 의 원인은 확인하지 못했다.

## 평가 (job 85953, low_p_hpc 1 노드 × 8 GPU, 20:35 – 23:2x)
- `slurm/submit-eval-q06-s4-ast.sh --after 85952`. s3 와 같은 평가셋·설정이다.
  - δ4·6·8·10 + final, batch 64, 꼬리 무음 1 s, --verify, fp32
- 직렬/배치 일치 검사 304 개 모두 통과
- 파일:
  - `summaries.json`: 7 개 결과의 groups·translation·aa_wer
  - `mt-s4-vs-s3.txt`: 번역 BLEU 짝 bootstrap 1,000 회(문장 재표본, ast_eval 과 같은 토큰화)
    - chrF·LAAL(평가 규약 (k+1)·80 ms)·빈 번역(침묵 / 전사만)
  - `asr-bootstrap-s4-vs-s3.txt`: 받아쓰기 짝 bootstrap 2,000 회(LibriSpeech 화자 군집)
  - `cmp_s4_s3.py`: 위 두 표를 만든 스크립트(`python cmp_s4_s3.py mt|asr`)
  - `koen_diag.py` · `koen-diag.txt`: 한→영 δ10 하락 진단(번역 모드 한국어 전사 오류, 이상치)
- 베이스라인 비교용 점수는 [[2026-10-12-simulst-baselines]] 에 있다.
  - `scores-all-v3.json`: s4 추가, 인코더 청크 보정 LAAL
  - `scores-xcomet-xl-v2.json`: s4 추가, 빈 번역 0 점
  - mxc 원본: `results/simulst-baselines-20261011/ours-s4-enc4/`, `results/xcomet-20261012/`(s4 캐시 3 개)
  - `ours-s4-enc4-partial/` 은 평가 도중 번역 셋만 먼저 변환한 판이다. 지우지 않고 남겨 두며, 위 점수에는 쓰지 않았다.

## 한→영 δ10 하락 진단 (`koen-diag.txt`)
- δ10 BLEU 16.95 → 14.90 이다. 짝 bootstrap 으로 유의하다([−2.98, −1.10]). δ4·final 은 유의하지 않다.
- 같은 문장의 번역 모드 한국어 전사도 나빠졌다.
  - pooled CER: δ10 8.28 → 10.18 %, δ4 10.23 → 10.67 %
  - CER > 50 % 인 문장: δ10 3 → 9 개
- 늘어난 실패는 앞부분 몇 마디만 내고 끊기는 전사다. 예: "그때는 제가", "그냥", "공통점이 있다 보니", "음 그건 그"
- 받아쓰기 전용 Kspon 에서는 변화가 없다.
- 원인은 확인하지 않았다.
