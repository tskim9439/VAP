## [2026-10-20] task | Basis Conversations 1500(한·독·영) mxc 직접 다운로드 완료 — 1,353 파일 30.3 GB, sha256 불일치 0

- Changed: mxc `VAPKT-data/data/basis-conversations-1500/`(@05a19e0, ko·de·en + 메타데이터), T5 `vapkt-corpora/basis-conversations-1500/`(메타데이터만)
- Reason: 사용자 "이것도 mxc 서버에 바로 다운로드해줘." — 이번 건만 서버 직접(토큰은 표준 입력, `hf_fetch.py`). 상시 규칙(T5 경유)은 그대로다.
- 결과:
  - 라이선스 Basis Data License 1.0: 연구·상업 학습·평가 허용, 원천 재배포·재판매·공개 저장 금지, 사설 클라우드 허용
  - 화자별 채널(48 kHz FLAC), 대화 2–4 인. 제공 전사는 실시간 STT 라 학습 감독에 쓰지 말라고 명시돼 있다.
  - 20 s 창 기준 말한 화자 ≤ 2 명 시간: 한국어 8.1 h(전체 9.5), 독일어 7.2 h(7.7), 영어 122.7 h(174.4)
- Next: 화자 채널별 두 교사 의사 전사(WER < 3 %) → 2 화자 대화 창 → 번역 타깃
- By: tskim
