---
author_id: tskim
created: 2026-10-11
source_type: source-drop
---

# q06-s2-ast(final) 전체 평가: 받아쓰기·번역 (mxc 호스트, 2026-10-11 09:15–14:07)

- 체크포인트: `/soundai/Model/VAPASR/semcommit-q06-s2-ast/final`
  - job 83866, 14,288 step, 2026-10-11 07:56 완료, 재로드 parity 통과
- 실행: 평가 대기열 `evalq-driver.sh`. `experiments/eval_single_turn_asr.py run` 을 shard 단위로 돌렸다.
  - 설정: δ4·6·8·10 + final, batch 64, 꼬리 무음 1 s, --verify
  - GPU: 처음 4 장(0·1·2·5)이 하나씩 가져갔다. 남은 CoVoST shard 4 개는 `evalq-extra-gpu6.sh` 로 GPU 6 에서 동시에 돌렸다.
  - GPU 6 으로 옮긴 이유: 진행 중에 `vapasr/data/ast_dataset.py` 를 바꿔야 했다. 그 전에 8 개 shard 의 코드 해시를 같게 고정했다.
  - 기본 driver 의 CoVoST summarize 는 "미완료"로 실패했다(GPU 6 shard 진행 중). 14:07 에 손으로 다시 summarize 했다.
  - 기록: `evalq-driver.log`
- 결과 폴더(mxc `results/`)와 예측 수(모두 complete):
  - `ast-fleurs-q06-s2-ast`(11,910), `ast-fleurs-en-peak05-q06-s2-ast`(5,690), `ast-covost2-q06-s2-ast`(145,175)
  - `single-turn-vpaa-kspon-q06-s2-ast`(31,575), `single-turn-yodas-de-q06-s2-ast`(24,305)
  - `single-turn-librispeech-{clean,other}-q06-s2-ast`(13,100 / 14,695)
- 파일:
  - `summaries.json`: 위 7 개 summary 의 groups·translation·aa_wer 와 코드 해시
  - `translation-metrics.txt`(`peek_eval.py` — 원자료 2026-10-11-q06-s2-ast-fleurs-ceiling 에 있음): 번역 BLEU·chrF·LAAL·AL 과 전사 오류율
  - `asr-bootstrap-vs-stage1.txt`(`asr_bootstrap_s2.py`): Stage 1(q06-s1m3-d10) 대비 짝 bootstrap 2,000 회
    - LibriSpeech 는 화자 군집, 나머지는 발화 단위다.
    - `*` 는 95 % 구간이 0 을 포함하지 않는다는 뜻이다.
  - `covost-empty-translations.txt`(`peek_covost_empty.py`, `peek_tonly.py`): CoVoST 영→독의 빈 번역 진단
    - δ4 의 빈 번역 749 건 = 통째로 침묵 65 건 + 전사만 내고 번역 없음 684 건
    - 침묵 건의 RMS 중앙값은 −55.4 dBFS 다(정상 −26.0).
    - 전사만 낸 건은 2–8 s 짧은 문장에 몰리고, 그중 272 건은 δ10 에서도 비었다. final 모드에서는 모두 번역됐다.
  - `kspon-d10-repetition.txt`(`peek_d10.py`): Kspon clean δ10 이 나빠진 원인
    - 발화 하나(KsponSpeech_E00521)가 "에 에 에 …" 반복으로 오류 667 개를 만들었다. 이 발화를 빼면 Stage 1 보다 오류가 60 개 적다.
