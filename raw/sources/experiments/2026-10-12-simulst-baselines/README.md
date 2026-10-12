---
author_id: tskim
created: 2026-10-12
source_type: source-drop
---

# 스트리밍 번역 베이스라인 비교 (2026-10-11 – 2026-10-12, mxc 호스트 GPU)

## 공통 조건 (`experiments/simulst_common.py`)
- 평가셋: `VAPKT-data/data/evaluation/ast-fleurs-covost2-v1/`
  - FLEURS 전체: 독→영 862, 영→독 641, 영→한 497, 한→영 382
  - FLEURS 영어 음량 보정본(peak05)
  - CoVoST 2 고정 부분집합 `manifest-max30s-covost2-sub7.jsonl`: `awk NR%7==1`, 독→영 1,931 · 영→독 2,217
- 입력: 원 발화 + 꼬리 무음 1 s(16 kHz). 우리 모델 평가와 같다.
- 단어 방출 시각: 그 단어의 마지막 글자가 나온 때까지 넣은 오디오 길이(계산 시간 제외, computation-unaware).
- 채점: `vapasr/data/ast_eval.py`
  - sacreBLEU. 영→한은 글자 토큰화, 나머지는 13a
  - chrF2
  - LAAL·AL(s). |X| = 원 발화 길이
- 명령: `python experiments/simulst_common.py score --pred-dir <폴더들> --out scores-all.json`

## 시스템
| 시스템 | 크기 | 설정 | 실행기 |
|---|---|---|---|
| 우리 q06-s3-ast(·s2) | 약 1.2B | δ4·6·8·10, final(오프라인) | `simulst_common.py from-ours`(평가 예측을 변환) |
| SeamlessStreaming | 2.5B | 판단 문턱 0.5·0.7, fp32 | `simulst_seamless.py` |
| IWSLT26 연결형 | 약 6B | 청크 0.64·0.96 s, 첫 대기 1 s | `simulst_iwslt.py` |
| Hibiki-Zero | 3B | 독→영만, fp16, seed 42 | `simulst_hibiki.py` |
| SimulStreaming | 1.55B + 9B | 청크 1.0 s, frame_threshold 25 | `simulst_simulstreaming.py` |

- IWSLT26 연결형 구성: Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B + Qwen3-4B-Instruct-2507
- SimulStreaming 구성: Whisper large-v3 + EuroLLM-9B-Instruct(CTranslate2)

## SimulStreaming 첫 실행 무효 (c1000-ft25)
- BLEU 가 0.9–7.9 로 나왔다. 출력이 반복·환각으로 망가졌다:
  - 같은 문장 되풀이, '.' 연속, "*Music*", "Hello, my name is …"
  - hyp/ref 단어 수 비 중앙값 1.75 (CoVoST 독→영)
- 원인으로 보는 것: 공식 README 가 권장(Recommended)하는 VAC(Silero VAD, `--vac`)를 끄고 무음까지 Whisper 에 넣었다.
  - 아래 4 문장 대조로 확인했고, 전체 재실행 점수(BLEU 16~34)로 최종 확인했다.
- Whisper 단계만 따로 돌린 결과 (CoVoST 독일어 4 문장, 청크 1.0 s, 꼬리 무음 포함):

  | 참조 | VAC 없음 | VAC 켬 |
  |---|---|---|
  | Mückenstiche sollte man nicht aufkratzen. | Gut / . / Mückenstiche sollte man nicht aufkratzen / . / : / - | Mückenst / iche sollte man nicht auf / kratzen. |
  | Ich habe eine neue Arbeit. | . / Ich habe / eine neue Arbeit / . / . / . | Ich habe ein / neue Arbeit. |
  | Neueste Nachrichten anzeigen. | Nein / . / Neueste Nachrichten anzeigen. ×3 | Neueste / Nachrichten anzeigen. / Ne |
  | Rohkost ist für Säuglinge nicht geeignet. | Hallo / ! / Ich bin der Chef von Solklingen. / Und heute zeige ich euch, wie man eine | Rohkost / ist für Säuglinge nicht gee / ignet. |

- 처리:
  - 실행기에 VAC 를 기본으로 넣었다(0.04 s 조각, 음성 구간 끝의 is_final 을 번역 쪽에 넘김). 설정 이름은 `c1000-ft25-vac` 다.
  - 6 개 조각을 GPU 6·0 에서 다시 돌렸다. 결과 폴더: `results/simulst-baselines-20261011/simulstreaming-vac/`(아래 재실행 절)
  - 첫 실행의 예측(`simulstreaming/`)은 지우지 않고 남겨 둔다. `scores-all.json` 에도 그대로 있으며, `tables.txt` 에 무효로 표시했다.
- Silero VAD 는 오프라인 torch.hub 로 불러온다. torch.hub 가 `trusted_list` 를 쓰므로, 읽기 전용인 `vapkt-baselines/torch-hub` 를 호스트 `/tmp/sa_tskim/torch-hub` 로 복사해 쓴다.

## SimulStreaming 재실행 (c1000-ft25-vac, 11:01–13:41)
- 호스트 GPU 6·0, 6 개 조각, 7,668 행
- rank 1·4 는 번역 방향을 바꿀 때 CT2 생성기를 다시 올리다 GPU 메모리 부족으로 멈췄다(13:07). 끝난 행은 건너뛰고 이어서 마쳤다(`queue-simul-vac-retry`, stderr 는 이어 붙임).
- 한→영 37 행은 공식 코드 안의 오류로 실패해 빈 번역으로 남았다. 오류 내용:
  - Whisper 토크나이저 `split_tokens_on_unicode` 의 IndexError 34 개
  - `process_iter` 의 `min()` 빈 시퀀스 3 개
- 결과(BLEU / chrF / LAAL):
  - FLEURS 독→영 28.24 / 61.01 / 2.27, 영→독 30.27 / 60.11 / 2.19, 영→한 34.28 / 29.86 / 2.72, 한→영 16.20 / 46.02 / 2.48
  - CoVoST 독→영 33.31 / 60.06 / 2.04, 영→독 25.95 / 54.71 / 2.04

## 우리 지연 보정 (scores-all-v2.json)
- 평가 예측의 방출 시각은 (디코더 step k + 1) × 80 ms 다. 인코더 lookahead 가 빠져 있다.
- Nemotron 인코더 설정(호스트에서 확인):
  - `att_context_style chunked_limited`, `att_context_size [56, 3]`
  - `conv_context_size [8, 0]`(인과), 인과 서브샘플링(8 배)
  - `streaming_cfg valid_out_len=4`, chunk_size [25, 32] mel 프레임
- 그래서 프레임 k 는 4 프레임 청크 ⌊k/4⌋ 가 다 들어온 뒤에 나온다. 보정 시각 = min(4(⌊k/4⌋+1), K) × 80 ms 이고, 입력이 끝난 뒤 flush 라운드는 그대로 둔다.
  - 함수: `simulst_common.encoder_chunk_times`
  - 명령: `from-ours --encoder-chunk 4` → `ours-s3-enc4/`, `ours-s2-enc4/`
- 청크 경계(README 의 "[56,3] = 0.32 s")를 썼으므로 보수적이다. 서브샘플링 의존(프레임 j 는 mel 8j 까지)과 STFT 창을 따지면 약 0.06 s 이르다.
- 영향: LAAL +0.11~0.15 s. 예: FLEURS 독→영 δ10 1.81 → 1.93, 영→한 δ10 4.36 → 4.51
- `scores-all-v2.json`·`tables-v2.txt` 는 다음 6 개 폴더를 채점했다:
  - 보정한 s3·s2(`ours-s3-enc4`, `ours-s2-enc4`)
  - Seamless·IWSLT·Hibiki(그대로)
  - SimulStreaming 재실행(`simulstreaming-vac`)
- `scores-all.json`·`tables.txt` 는 보정 전 첫 판이다.

## 파일
- `scores-all.json`: 시스템/설정/데이터셋별 BLEU·chrF·LAAL·AL·빈 번역 수(122 칸)
- `tables.txt`: 위 파일을 세트·방향별로 정리. `python make_tables.py scores-all.json > tables.txt`
- `scores-all-v2.json`·`tables-v2.txt`: 지연 보정 + SimulStreaming 재실행 판(위 절)
- mxc 예측 원본: `VAPKT-data/results/simulst-baselines-20261011/{ours,ours-s3,seamless,iwslt,hibiki,simulstreaming,simulstreaming-vac}/`
  - IWSLT 예측은 채점 때 `/tmp/sa_tskim/score-iwslt` 에 모아 읽었다.
