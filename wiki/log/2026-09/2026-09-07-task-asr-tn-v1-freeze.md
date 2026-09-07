## [2026-09-07] task | asr-tn-v1.0.0 구현·audit·동결

- Changed: `vapasr/data/textnorm.py`, `vapasr/data/kspon.py`, `experiments/tn_audit.py`, `experiments/s1_build_manifest.py`, `experiments/s1_align.py`, `vapasr/uslm/mono_data.py`, `tests/test_textnorm.py`, `tests/golden/asr-tn-v1.0.0.json`, `wiki/decisions/decision-asr-tn-v1-freeze.md`, `wiki/sources/source-asr-tn-v1-audit.md`, `wiki/outputs/output-asr-tn-v1-spec.md`(동결 판정 절), `raw/sources/experiments/2026-09-07-asr-tn-v1-audit/`
- Reason: Stage 2 데이터 확장 전에 정규화 규약을 동결하라는 지시. spec 대로 구현하고 LibriSpeech 960 h·KsponSpeech 01–05 전체 audit 와 230 h diff 를 수행해 관문 7/8 통과, fingerprint 관문을 정렬기·학습기에 넣었다.
- Next: Kspon 01–05 review TSV 사용자 검토(관문 6) → librispeech-960·kspon-full manifest 생성(fingerprint) → align-asr-tn-v1/ 정렬·특징 추출 → scaling curve.
- By: tskim
