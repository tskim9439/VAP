---
type: task
status: open
owner: tskim
due: 2026-09-17
priority: p1
created: 2026-09-03
updated: 2026-09-11
summary: otoSpeech·CANDOR 라이선스 동의 및 확보, SpokenWOZ 채널 구조 확인
sources:
  - [[output-streaming-vap-research-plan]]
---

# 영어 코퍼스 확보

## 배경

→ [[source-conversation-corpora]]

otoSpeech 104h + CANDOR 850h 로 약 950시간이면 SpokenWOZ 없이 시작 가능하다.

## 완료 조건

- [ ] otoSpeech 다운로드 — custom non-commercial 라이선스 조건 확인
      (**voice cloning 금지 조항이 파생 모델 배포에 미치는 영향** 검토)
- [ ] CANDOR 확보 — CC BY-NC. 상업적 파생물 불가가 향후 문제인지 판단
- [ ] CANDOR 는 video chat 기반 → **네트워크 지연·에코가 timing 라벨을 왜곡하는지 검증**
- [ ] SpokenWOZ 배포 파일의 채널 수 확인 → [[question-spokenwoz-channel-structure]]
- [ ] 코퍼스별 실사용 가능 시간 집계

## 진행 기록

- 2026-09-03: 생성.
- 2026-09-11: otoSpeech 는 mxc 에 반입됨(`/soundai/DB/raw/otoSpeech16k`, 104.9 h). 사용자 결정 — otoSpeech 학습 사용, CANDOR 는 확보 가능하면 사용. 남은 것: CANDOR 신청·다운로드(로컬 경유 업로드), 라이선스 원문 재확인.
