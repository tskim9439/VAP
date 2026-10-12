---
author_id: tskim
created: 2026-10-12
source_type: source-drop
---

# q06-s3-ast(final) 학습·평가 (2026-10-11 22:35 – 2026-10-12 10:3x)

## 학습 (job 84802, hpc 2 노드)
- 제출: `slurm/submit-q06-s3-ast.sh --partition hpc` (사용자)
  - 시작점: q06-s2-ast/final
  - 2 epoch, 14,288 step
  - 번역 모드 0.8, 영어 음량 증강 0.5 · RMS [−70, −20] dBFS
- 시작·선점 기록:

  | 시작 | 노드 | 결과 |
  |---|---|---|
  | 23:04 | hpc-39·40 | 23:05 선점 |
  | 23:08 | 같은 노드 | 23:10 선점 |
  | 23:13 | hpc-22·23 | 학습. 01:31 선점(step 4,040, checkpoint-4000 은 01:30:57 에 완전 저장) |
  | 01:47 | hpc-31·32 | 준비 중 선점 |
  | 01:56 | hpc-33·38 | checkpoint-4000 에서 재개(지문 일치), 07:30 완료 |

- 결과: parity True, 인코더 638 키, train_loss 0.210, 마지막 top1_text 0.90 (s2 0.875)
- 속도:
  - 0.66 s/step 로 시작했다.
  - 00:45–01:11 에 호스트에서 CoVoST 영어 정렬 worker 3 개가 `/soundai` 를 크게 읽고 쓰는 동안 약 4 s/step 까지 느려졌다.
  - worker 를 멈추자 1.4 → 0.63 s/step 으로 돌아왔다.
  - 공유 저장소 경합으로 본다. 학습 중에는 호스트에서 저장소를 많이 쓰는 작업을 피한다.

## 평가 (job 85088, hpc 1 노드 × 8 GPU)
- 제출: `slurm/submit-eval-q06-s3-ast.sh --partition hpc` (사용자)
  - `slurm/eval-single-turn-queue.sbatch` 가 작업 목록(`results/evalq-q06-s3-ast/tasks.txt`, 44 개)을 GPU 8 장에 나눈다.
- 평가셋: s2 와 같다(FLEURS 4 방향, FLEURS 영어 음량 보정, VoxPopuli·Kspon, YODAS-de, LibriSpeech). CoVoST 2 는 베이스라인 비교용 고정 부분집합 `manifest-max30s-covost2-sub7.jsonl`(7 행마다 1 행, 4,148 행)이다.
  - s2 는 CoVoST 전체를 평가했으므로, 같은 id 만 골라 비교했다(`cmp_covost.py`).
- 설정: δ4·6·8·10 + final, batch 64, 꼬리 무음 1 s, --verify
- 직렬/배치 일치 검사 실패 1 건:
  - 위치: CoVoST 영어 δ10, rank 1, 표본 common_voice_en_684385 (`parity-fail-covost-rank1-English-d10.json`).
  - 청크 53 에서 `<TRANS>` 와 전사 토큰 1899 의 순서만 뒤바뀌었다(배치 계산의 미세한 수치 차이로 점수가 비슷한 두 후보가 엇갈림).
  - 처리: 그 rank 의 나머지는 `--verify` 없이 같은 설정으로 호스트 GPU 6 에서 이어 마쳤다(설정 지문에는 verify 가 없다).
  - 나머지 63 개 검사는 통과했다.
- 파일:
  - `summaries.json`: 7 개 결과의 groups·translation·aa_wer
  - `fleurs-s3-vs-s2.txt` · `covost-sub7-s3-vs-s2.txt`: 번역 BLEU·chrF·LAAL, 빈 번역(침묵 / 전사만 내고 번역 없음) 수
  - `asr-bootstrap-s3-vs-s2.txt`: 받아쓰기 짝 bootstrap 2,000 회(LibriSpeech 화자 군집)
  - `job-85088.log`
