---
type: question
status: seed
created: 2026-09-03
updated: 2026-09-03
summary: SpokenWOZ 249시간이 화자별 분리 채널로 배포되는지 — 아니면 VAP 학습에 쓸 수 없다
sources:
  - [[source-conversation-corpora]]
---

# SpokenWOZ 는 dual-channel 인가?

## 질문

SpokenWOZ 249시간 전화 대화가 **화자별로 분리된 채널**로 배포되는가,
아니면 믹스된 mono 인가?

## 왜 중요한가

[[voice-activity-projection]] 은 화자별 분리 오디오를 전제한다. mono 믹스라면
diarization 을 먼저 돌려야 하고, diarization 오류가 VAP target 에 그대로 전파되어
**결과 해석이 불가능해진다.** 그 경우 SpokenWOZ 를 학습 데이터에서 제외하는 편이 낫다.

## 지금까지 아는 것

- SpokenWOZ 는 전화 기반 human-to-human task-oriented dialogue, 249시간,
  음성-텍스트 정렬 어노테이션 보유.
- 전화 녹음이라 원본은 2채널일 가능성이 높지만 **공개 배포 형식은 확인되지 않았다.**
- [[source-conversation-corpora]] 의 다른 코퍼스는 채널 구조가 확인되었다
  (AI Hub stereo ✅, CANDOR 화자별 분리 ✅, otoSpeech dual-channel ✅).

## 답을 얻는 방법

실제 배포 파일 1개를 받아 `soxi` / `ffprobe` 로 채널 수를 확인하고,
2채널이면 두 채널의 상관을 계산해 진짜 분리인지(누설 정도) 본다.

## 상태

미해결. 우선순위는 낮다 — 영어 데이터는 otoSpeech + CANDOR 만으로도 약 950시간이라
SpokenWOZ 없이 시작할 수 있다. → [[task-secure-english-corpora]]
