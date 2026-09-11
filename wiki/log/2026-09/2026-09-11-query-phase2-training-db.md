## [2026-09-11] query | Phase 2 학습 DB 확정안 (한국어/영어, 중복 제거)

- Changed: `wiki/outputs/output-phase2-training-db.md` 생성, 데이터 목록 페이지에 링크
- Reason: 사용자 요청 — 나열이 아니라 언어별로 쓸 DB 를 정리하고, AI Hub 71631 = NIA24 134-1 = Phase 1 발화 crop 같은 동일 녹음 중복을 제거. 동일 자원 대응표, KO/EN 역할별 표, 중복 제거 규칙 5 개, Q0 우선 작업
- Next: Q0 — 134-1 실외 복원 manifest, stem 대조표, 134-2 검증, Switchboard 시각 검증
- By: tskim

## [2026-09-11] query | 선별 기준을 A 등급(화자별 채널/채널 기원 조각)만으로 변경, 시각 라벨 불요

- Changed: `output-phase2-training-db.md` §6(A 등급 KO/EN 표·합계), 186 복지 콜센터(화자별 스마트폰 녹음 확인)·raw/132(화자쌍·발화 화자별 파일 확인) 편입, 회의·인터뷰 제외
- Reason: 사용자 — 시각은 Qwen ForcedAligner 로 다시 뽑으므로 시각 라벨 불요, A 등급만 고려. 시간축 없는 조각은 준자연 스트림으로 쓰되 turn 손실 마스크
- Next: Q0 — 134-2 채널 기원 검증, Switchboard 복원, 186·132 준자연 스트림 생성기
- By: tskim

## [2026-09-11] query | SpokenWOZ·AliMeeting·AISHELL-4 검토, Fisher 라이선스

- Changed: `output-phase2-training-db.md` §6.2 에 세 코퍼스 행 추가, `question-spokenwoz-channel-structure` resolved
- Reason: 사용자 질문. SpokenWOZ 는 두 트랙·8 kHz·단어 시각(논문 원문)이지만 겹침을 규칙으로 억제; AliMeeting 은 헤드셋 채널(A)이나 중국어; AISHELL-4 는 어레이만(B)+중국어 → 제외. Fisher 는 LDC 유료(비회원 요금 로그인 필요)
- Next: 회사·협력 기관의 LDC 회원 여부 확인
- By: tskim
