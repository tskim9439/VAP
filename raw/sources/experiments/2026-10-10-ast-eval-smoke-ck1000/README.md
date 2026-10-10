---
author_id: tskim
created: 2026-10-10
source_type: source-drop
---

# ast-v1 평가 경로 시험: checkpoint-1000(q06-s2-ast), 세트당 16 발화 (mxc 호스트 GPU 0·1, 2026-10-10 21:3x)

- 목적: 수치가 아니라 평가 경로를 확인하는 것이다. 학습 1,000 step 이고 세트당 16 발화라 수치로 판단하지 않는다.
- 실행: `experiments/run_single_turn_asr_mxc.sh <ckpt> <out> <gpu> 32 16 6 <manifest> final`(δ6 + final, --verify 로 순차·배치 디코딩 일치 확인)
  - 번역: `ast-fleurs-covost2-v1/manifest.jsonl` → `results/ast-smoke-ck1000-st/`
  - 받아쓰기: `single-turn-vpaa-kspon-v1` → `results/ast-smoke-ck1000-asr/`
- 확인:
  - 체크포인트 config 에 `ast_format=ast-v1`, `<TRANS>`=151726, runaway_cap 32 가 있고, 태그 차단이 풀려 있다.
  - prefix 를 자동으로 바꾼다. parity 는 영·독·한 모두 PASS 다.
  - 출력이 전사와 번역으로 나뉜다. BLEU·chrF·LAAL·AL 이 계산된다.
- 발견:
  - CoVoST 2 영어 test 에 음성·문장 불일치 발화가 있다. 예: 142.5 s 음성의 참조가 15 단어 한 문장이다. 30 s 를 넘는 7 개(영→독만)가 모두 이렇다.
  - 이 한 발화가 LAAL −25 s, final WER 1,174 % 를 만들었다. 나머지 15 개의 LAAL 은 0.67–1.6 s 다.
  - 조치: `manifest-max30s.jsonl` 을 만들었다(CoVoST 의 30 s 초과 7 개 제외, FLEURS 는 그대로, 31,417 행). `ast_eval.translation_report` 에 LAAL·AL 중앙값을 더했다.
  - 1,000 step 에서 영→한 δ6 은 16 개 중 3 개가 번역 구간 없이 끝났다. 학습 뒤에 다시 본다.
- 파일: `summaries-and-exclusions.txt`(두 시험의 세트별 지표와 제외 목록)
