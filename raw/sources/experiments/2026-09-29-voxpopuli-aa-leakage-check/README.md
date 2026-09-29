---
author_id: tskim
created: 2026-09-29
source_type: source-drop
---

# VoxPopuli-Cleaned-AA 와 학습 데이터 겹침 점검 (2026-09-29)

- 평가 세트: HF `ArtificialAnalysis/VoxPopuli-Cleaned-AA` @ `07adf4a2` 628 발화(로컬 T5 사본, 서버 업로드는 사용자).
- 학습 쪽 기준: mxc `labels/voxpopuli/asr_en.tsv`(공식 VoxPopuli ASR 주석, `|` 구분)의 id·session_id·speaker_id·split 과 train split 정규화 전사.
  서버에서는 이 네 칸과 train 전사만 로컬로 내려받아 비교했다(평가 데이터는 서버에 올리지 않음).
- 결과(`overlap.json`): 628 발화 모두 공식 test split. 같은 연설(paragraph) 0/347, 정규화 문장 완전 일치 0, 5-gram 포함률 ≥ 0.5 는 2 발화(상투 문구).
  세션 280/292 와 식별된 화자 207 명 중 80 명(그들의 train 발화 13,818 개)은 train 에도 있다 → 발화·전사 누수는 없고, 도메인·일부 화자가 겹친다.
- 방법: 세션 = 원 id 의 `-en_` 앞, paragraph = `en_<시각>`(마지막 `_n` 앞). 전사 비교는 소문자·구두점 제거 뒤 문장 일치와 단어 5-gram 포함률.
