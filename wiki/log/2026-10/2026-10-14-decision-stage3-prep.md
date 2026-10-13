## [2026-10-14] decision | CST 학습 단계 확정(4 단계)과 Stage 3 준비 계획

- Changed:
  - [[decision-cst-training-stages-and-speaker-slots]] accepted(확정 내용 절 추가)
  - [[output-staged-training-plan-20261001]] 에 단계 번호 대체 표시
  - [[output-stage3-prep-plan-20261014]] 신규
- Reason:
  - 사용자 결정(2026-10-14):
    - Stage 4·5 를 합쳐 확정
    - Stage 3 출력은 ast-v1 태그 + 턴 끝 `<EOT>`
    - 3 인 이상 대화도 두 화자만 골라 사용
    - 독일어는 새로 받은 NSC·ZEC 등도 고려
- 결과:
  - Phase 2 산출물이 재사용 가능하다(mxc `data/phase2/`): 정제 대화 7 종, 정렬, mono 혼합, split v1, 창(KO 약 850 h, otoSpeech 105 h, 회의)
  - 새 독일어 코퍼스(`data/` 아래 10-10 수령):
    - KEC: 남부 독일어 자유 대화 40 h, 79 명, 화자별 채널 + TextGrid
    - ZTC: 사용자가 말한 'ZEC' 로 보인다. Zurich Tangram Corpus, 스위스식 표준 독일어 과제 대화 42 h, 29 쌍, 화자별 채널 + 사람 전사 + MAuS
    - NSC: 155 h, 300 명. 상대가 거의 한 사람이고 2 인 구간에 전사가 없어 보류
- Next: P0 점검(창·조립·held-out 누설) → P1 독일어 어댑터 → P2 회의 2 화자 창 → P3 직렬화
- By: tskim
