## [2026-09-08] task | 데이터 스키마 v1 (manifest·정렬 카드, 검증기, 로더 관문)

- Changed: `vapasr/data/schema.py`, `experiments/ds_cards.py`, `vapasr/uslm/mono_data.py`(카드 관문), `experiments/s1_build_manifest.py`(dataset.json), `experiments/s1_align.py --card`, `slurm/s2_prep.sbatch`, `wiki/outputs/output-dataset-schema-v1.md`
- Reason: 코퍼스·단계가 늘면서 manifest/정렬/캐시 계약이 코드에 흩어져 있어 스키마를 못 박고 카드(dataset.json/align.json)로 재현성·검증을 관리. 기존 행은 수정하지 않음.
- Next: 전 산출물 카드 생성 결과 확인, VAPASR_STRICT_SCHEMA 기본화 시점 결정, parquet 캐시.
- By: tskim
