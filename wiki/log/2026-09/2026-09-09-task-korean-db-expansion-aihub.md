## [2026-09-09] task | 한국어 DB 확장 — AI Hub 71631 자유대화 · 031/033 방송 manifest

- Changed: `vapasr/data/aihub.py`(71631 stereo 리더 병렬화·NFC 디렉토리 매칭, 031/033 간투사 `/` 제거), `vapasr/data/textnorm.py` v1.3.0, `experiments/s1_build_manifest.py`(aihub71631-train/dev, aihub-bc-train), `wiki/outputs/output-dataset-schema-v1.md` §5b
- Reason: NIKL 추가 오디오가 없어 71631 → 031/033 → 98 순으로 한국어 데이터를 늘리기로 함. 결과 aihub71631-train 231,647 발화 159 h · dev 66,365 발화 44 h · aihub-bc-train 448,867 발화 578 h. 98 은 라벨 파일이 서버에 없어 보류.
- Next: `slurm/s2_prep.sbatch` 로 세 manifest 정렬(2 노드) → D2 TRAIN 에 추가. 98 라벨 확보 여부 확인.
- By: tskim
