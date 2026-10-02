---
type: task
status: open
owner: tskim
due: TBD
priority: p0
created: 2026-10-02
updated: 2026-10-02
summary: CST-Bench 원본 — 실제 2인 교차언어 대화 코퍼스 접근·라이선스 확인(TAXI·Verbmobil·CS-Dialogue·Bangor·ETRI), 즉시 쓸 것부터 확보
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
  - '[[decision-project-goal-cst-s2tt]]'
---

# CST 원본 대화 코퍼스 확보 (Phase 0)

## 배경
- [[cst-bench]] 는 같은 세션에서 서로 다른 언어를 쓰는 실제 2인 multi-turn 대화가 있어야 만들 수 있다.
- 지금 이 볼트에 그런 데이터는 없다. 낭독 병렬 음성·단순 합친 다언어 코퍼스는 명세상 배제다.

## 후보와 확인 항목

| 후보 | 언어 | 확인할 것 |
|---|---|---|
| BAS TAXI | DE↔EN | 배포처·라이선스(연구·변형·재배포), 전사·번역 형식, 시간 정보 |
| Verbmobil II Multilingual | DE↔EN, DE↔JA | 위와 같음 + 사람 통역사 부분집합 |
| BAAI CS-Dialogue | ZH/EN | 공개 여부·라이선스, 실제 교차언어 대화인지(코드스위칭 대화일 가능성 확인) |
| Bangor bilingual corpora | ES/EN, CY/EN, CY/ES | 같은 세션 교차언어 비율, 번역 유무 |
| ETRI/SiTEC | KO↔EN | 배포 경로(Phase 5) |

## 완료 조건
- [ ] 후보별 접근 경로·라이선스·형식 표를 `wiki/sources/` 에 남긴다(관측일·URL)
- [ ] 즉시 접근 가능한 코퍼스 1 개 이상을 T5 SSD(`/Volumes/Samsung_T5`)에 받는다. 서버 업로드는 사용자가 한다
- [ ] 명세 6절 필수 조건 충족 여부를 코퍼스마다 판정한다

## 진행 기록
- 2026-10-02: 생성(목표 전환).
