## [2026-10-14] query | NSC 2 인 대화 부분 확인 — Stage 3 독일어에 포함

- Changed:
  - [[output-stage3-prep-plan-20261014]] 의 데이터 표·NSC 항목·P1
  - [[decision-cst-training-stages-and-speaker-slots]] 의 확정 내용 4
- Reason:
  - 사용자 확인: "또 있지 않아?"는 NSC 를 말한 것이다.
- 결과:
  - NSC 업로드 목록에 2 인 대화 파일이 있다. `semispontaneousdialogs_{speaker,interlocutor}.wav` 276 세션이고, 48 kHz, 두 채널 길이가 같으며, 세션당 6–7.5 분, 약 35 h 다.
  - `interaction_##` 3,375 쌍은 같은 267 세션의 부분 조각이다.
  - 전사·턴 시각 파일은 업로드 목록에 없다(wav·xml 뿐). → 교사 ASR 의사 전사가 필요하다.
  - 그 밖에 받은 독일어 대화 코퍼스는 없다. Emilia(`/soundai/DB/Emilia-Dataset`)는 DE·KO·EN 모두 Git LFS 포인터만 있고 실제 파일이 없다.
- Next: P1 에서 KEC·ZTC 어댑터와 NSC 의사 전사를 함께 진행
- By: tskim
