## [2026-09-22] query | speechlm 단일 화자 ASR manifest

- Changed: `wiki/outputs/output-speechlm-single-speaker-manifest.md`, `wiki/sources/source-speechlm-v1-encoder-alignment-2607.md`, `experiments/build_speechlm_single_speaker_arrow.py`, `experiments/speechlm_single_speaker_catalogs.json`
- Reason: speechlm 2607에서 파일당 단일 화자 또는 화자별 분리 채널 조건을 만족하는 DB만 추리고, 원 Arrow와 tar 메타데이터를 결합한 재현 가능한 새 Arrow를 만들기 위해서다.
- Next: 생성 완료된 47,662,558행 manifest에 기존 데이터 중복 제거와 source별 cap을 적용하고 교사 합의 QC로 넘긴다.
- By: tskim
