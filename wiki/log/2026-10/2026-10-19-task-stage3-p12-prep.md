## [2026-10-19] task | Stage 3 파일럿 p12 준비 — 정렬 대상이 아니었던 격리 발화 정렬(NSC 12,309 되살림, 한국어 2.98 만은 오디오 없는 결손), 재료 revive-asr-v2, dry-run 통과

- Changed:
  - `experiments/stage3_revive_tokens.py --extra-align`(기본 정렬에 없는 발화만 채움)
  - raw [[2026-10-18-stage3-ko-revive]]: 재료 v2·p12 준비 절, `revive_extra_input.py`·`revive_v2_chain.sh`·로그
- Reason: 사용자 "NSC 는 정렬을 뽑으면 되지않아?" → "응 p12까지 진행해"
- 결과:
  - NSC 격리 발화 12,309(교사 Qwen 전사 정규화)를 `align-revive-v2` 에 정렬해 모두 되살렸다(5.7 h).
  - 134-1·134-2 의 미정렬 2.98 만은 전부 `audio_missing` 이었다(`meta.missing` 에 빠진 결손 조각). 되살리지 않는다.
  - `revive-asr-v2`: 학습 split `dlg_de` 되살림 15,066 → 25,929. NSC 창당 학습되는 교대 3.91 → 7.23
  - dry-run 6 셋 불일치 0. 평가 큐 3 개를 미리 걸었다.
- Next: 사용자가 p12(`st3-p12-revive2`)를 제출한다. p11 과 같은 step 에서 독일어 귀속을 비교한다.
- By: tskim
