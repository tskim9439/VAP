---
author_id: tskim
created: 2026-10-10
source_type: source-drop
---

# 음성번역 평가셋: FLEURS + CoVoST 2 (mxc `data/evaluation/ast-fleurs-covost2-v1`, 2026-10-10)

- 원천(맥 T5 에 받아 사용자가 azcopy 로 올림):
  - mxc `data/corpora/fleurs`(google/fleurs: en_us·de_de·ko_kr 의 test·dev tsv와 음성 tar, ko·de train tsv)
  - mxc `data/corpora/covost2-fixie`(fixie-ai/covost2: en_de·de_en test parquet, Common Voice 4 mp3 포함)
  - 큰 파일은 HF sha256, tsv 는 크기로 검증했다. 업로드 뒤 30 파일의 크기가 맥과 같다.
- 만든 방법:
  - `experiments/eval_single_turn_asr.py prepare --fleurs-root … --covost-root … --workers 32`(호스트 CPU, 약 55 분, NFS 쓰기 병목)
  - FLEURS: 번역 참조는 같은 FLoRes 문장 id 의 목표 언어 전사(test → dev → train 순)다. 목표 언어에 없는 문장은 빠진다(영→한 497/647, 영→독 641/647).
  - CoVoST 2: mp3 를 16 kHz 모노 16-bit wav 로 바꿨다(scipy resample_poly). 번역문 라이선스는 CC BY-NC 4.0 이다.
- `coverage.json`: 세트별 원천 수·사용 수·시간
- 받아쓰기 참조는 원문 전사를 canonical_reference 로 정규화한 것이다. FLEURS 한국어 원문에는 아라비아 숫자가 있어(예 "15미터"), 한국어 받아쓰기 CER 은 보조 지표로만 본다.
