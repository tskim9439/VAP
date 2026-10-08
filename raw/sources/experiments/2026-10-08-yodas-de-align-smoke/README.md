---
author_id: tskim
created: 2026-10-08
source_type: source-drop
---

# YODAS-Granary 독일어 전처리 — 사전 점검 요약과 GPU 단계 시험 실행 (mxc, 2026-10-08)

- `prefilter-summary.json`: `experiments/yodas_de_prepare.py prefilter` 결과. 출력 루트 mxc `VAPKT-data/data/yodas-de-v1/`(inventory, prefilter)
  - 평가셋 대조 파일: Europarl-ST 독일어 dev/test 4 개(8-gram 103,130 개). FLEURS·CoVoST2 독일어는 아직 없음
- `align-summary-smoke1.json`: `experiments/yodas_de_align.py` 첫 시험(shard 6 개, 후보 104 발화). mxc `VAPKT-data/results/yodas-de-align-smoke-20261008/`
  - 정렬 실패 8 건은 모두 `invalid_or_out_of_clip_timestamp`(정렬기 끝 시각이 클립 길이를 넘음)
- `align-summary-smoke2.json`: 영·한 파이프라인과 같은 경계 보정(`selection_alignment_qc.classify_failed`: 80 ms 이하 넘침은 잘라 맞춤, 80–320 ms 검토, 그 이상 거부)을 넣은 두 번째 시험. mxc `VAPKT-data/results/yodas-de-align-smoke2-20261008/`
  - 실패 8 건 중 7 건이 80 ms 이하로 구제, 1 건은 넘침 300 ms("das" 끝 3.04 s, 클립 2.74 s)라 검토 대상으로 제외
- `validate_outputs.py`: 시험 출력 검증 스크립트(mxc vapasr env, CPU). 결과:
  - words 47 행(train 42, dev 5, 보정 7): tar 색인 = tarfile 재스캔, 오디오 길이 = duration_s(±2 ms), 토큰 시각 단조·범위 안, Qwen3-ASR tokenizer 재분할(`split_words` → `lexical_words`) 일치, 토큰 decode = display_text — 모두 위반 0
  - `SemCommitDataset(..., sem_free_unlabeled=True, ignore_labels=True, tok_check=전부)`: 47 항목, bad 0, 언어 머리 `language German<asr_text>` 로 시퀀스 생성 확인
- 시작 지연: 모델·라이브러리 import 에 약 11 분(NFS 위 conda env), shard 처리는 후보 48 발화에 약 9 s
