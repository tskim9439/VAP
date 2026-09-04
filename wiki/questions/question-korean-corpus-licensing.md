---
type: question
status: seed
created: 2026-09-03
updated: 2026-09-03
summary: 재배포 가능한 한국어 stereo dyadic 대화 코퍼스가 존재하는가, AI Hub 어노테이션 파생물은 공개 가능한가
sources:
  - [[source-conversation-corpora]]
---

# 한국어 대화 데이터의 배포 가능성

## 질문

1. AI Hub 감정 태깅 자유대화의 **어노테이션 파생물**(타임스탬프 + 이벤트 라벨)을
   공개 배포할 수 있는가? 이용약관 원문의 근거는?
2. AI Hub 외에 **재배포 가능한 한국어 stereo dyadic 대화 코퍼스**가 존재하는가?

## 왜 중요한가

[[decision-korean-benchmark-release-scope]] 전체가 이 답에 달려 있다.
공개 불가면 "한국어 benchmark 신설" 이라는 기여 자체를 다르게 서술해야 한다.

## 지금까지 아는 것

- AI Hub: **내국인만 신청 가능**, 승인 필요, 승인된 사용자로 재배포 제한.
- KsponSpeech 는 대화가 아니라 발화 단위 ASR 코퍼스 → VAP 학습에 부적합.
- 영어 쪽은 CANDOR(CC BY-NC), otoSpeech(non-commercial) 로 최소한 연구 목적
  배포가 명확하다. 한국어에 대응물이 없다.

## 답을 얻는 방법

- AI Hub 이용약관·데이터 활용 조건 원문 확인. 필요하면 운영기관에 직접 문의하고
  **답변을 문서로 남긴다** (추정 금지).
- 국립국어원 모두의말뭉치, KAIST/ETRI 대화 코퍼스 등 대안 조사 — 조사 시
  **stereo 여부와 타임스탬프 유무를 먼저 본다.** 이 둘이 없으면 VAP 에 못 쓴다.

## 상태

미해결. → [[task-verify-aihub-stereo-and-access]]
