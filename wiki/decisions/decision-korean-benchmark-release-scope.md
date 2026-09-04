---
type: decision
status: active
decision_status: proposed
owner: tskim
review: 2026-12-24
created: 2026-09-03
updated: 2026-09-03
summary: AI Hub 재배포 제약으로 한국어 벤치마크는 오디오가 아닌 어노테이션·툴킷만 공개하는 방식으로 설계
sources:
  - [[source-conversation-corpora]]
  - [[source-turnbench]]
---

# 결정: 한국어 turn-taking benchmark 의 공개 범위

## 맥락

[[source-turnbench]] 는 **영어 전용** 이다. 한국어 turn-taking benchmark 는
이 연구의 유력한 기여 후보다. 초안은 AI Hub 에서 10–30시간을 분리해 사람이
EOT/HOLD/BACKCHANNEL/INTERRUPTION 을 라벨하자고 제안했다.

**문제**: [[source-conversation-corpora]] 확인 결과 AI Hub 감정 태깅 자유대화는
**내국인만 신청 가능** 하고 승인된 사용자로 재배포가 제한된다.
→ **오디오를 포함한 "Korean TurnBench" 를 공개 배포할 수 없다.**

벤치마크의 가치는 재현성인데, 외부 리뷰어가 오디오를 받을 수 없으면 벤치마크로서
기능하지 못한다.

## 검토한 선택지

| 선택지 | 장점 | 단점 |
|--------|------|------|
| **A. 어노테이션 + 툴킷만 공개** | AI Hub 데이터의 풍부함을 그대로 활용. 법적 문제 없음 | 사용자가 AI Hub 승인을 직접 받아야 함 (내국인 한정) → 사실상 국내 전용 |
| B. 재배포 가능한 한국어 대화 코퍼스를 새로 찾음 | 완전 공개 가능 | stereo dyadic 한국어 코퍼스가 존재하는지 불명 |
| C. 소규모 자체 녹음 | 완전한 배포 권리 | 비용·시간이 크고 규모가 작음 |
| D. 공개 포기, 내부 평가셋으로만 사용 | 가장 단순 | 논문 기여로 주장할 수 없음 |

## 결정 (잠정)

**A 를 기본으로 하되 B 를 병행 조사한다.**

공개 범위:

```text
공개 O   타임스탬프 + 이벤트 라벨 (EOT/HOLD/INT/BACKCHANNEL)
         AI Hub 파일 ID + 오프셋으로 참조
         어노테이션 가이드라인, IAA 수치, 평가 스크립트
         baseline 결과표

공개 X   오디오, 전사 원문
```

즉 **"Korean TurnBench" 가 아니라 "AI Hub 상의 turn-taking 어노테이션 레이어 +
평가 툴킷"** 으로 포지셔닝한다. 이는 정직하고 법적으로 안전하며, 국내 연구자에게는
완전히 재현 가능하다.

B 안 조사 결과 재배포 가능한 한국어 stereo 대화 코퍼스가 있으면 그쪽으로 전환한다.
→ [[question-korean-corpus-licensing]]

## 결과 / 파급

- 논문에서 기여를 **"공개 벤치마크"가 아니라 "어노테이션 프로토콜 + 평가 툴킷 +
  한국어 분석"** 으로 서술해야 한다. 과대 주장하면 리뷰에서 문제가 된다.
- 어노테이션 저장 형식이 오디오와 분리 가능해야 한다 (파일 ID + 시간 오프셋).
  설계 단계에서 반영한다. → [[task-korean-benchmark-design]]
- AI Hub 이용약관 원문을 직접 읽고 "어노테이션 파생물 공개" 가능 여부를
  **문서로 확인** 해야 한다. 추정으로 진행하지 않는다.

## 재검토

2026-12-24 — 어노테이션 설계 완료 시점.
