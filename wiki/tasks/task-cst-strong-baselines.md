---
type: task
status: doing
owner: tskim
due: TBD
priority: p1
created: 2026-10-02
updated: 2026-10-05
summary: CST 기준선 B0–B4 — strong SimulST 선정, 2×SimulST 양방향 라우팅, VAD·VAP·Oracle 턴 경계 결합
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
  - '[[cst-bench]]'
---

# strong pipeline 기준선 (Phase 3)

## 배경
- 명세의 최우선은 기준선이 상호작용에서 어디서 실패하는지 보이는 것이다.
- B3 의 VAP 는 재현해 둔 것([[output-vap-turnbench-baseline-reproduction]])을 쓴다.
- 원천 쪽 스트리밍 인식은 이 볼트의 백본([[output-stage1-pilot-eval-20261001]])을 후보로 둔다.

## 완료 조건
- [ ] strong SimulST 후보 조사와 선정(언어쌍별 공개 모델, 지연 설정)
- [ ] B0 오프라인 ST, B1 2×SimulST
- [ ] B2 VAD 라우팅, B3 VAP 라우팅, B4 Oracle 턴 경계
- [ ] CST-Bench v0 로 전 조건 평가, 실패 사례 수집

## 진행 기록
- 2026-10-02: 생성.
- 2026-10-05: 범위 변경([[decision-cst-paper-v1-scope]]) — 베이스라인은 S2TT 지원 시스템(SeamlessStreaming, 동시 캐스케이드, 오프라인 ST + AlignAtt, StreamSpeech·InfiniSST 는 한 방향만), 정답 분할·실제 조건(VAD + 언어 판별) 래퍼. VAP 라우팅은 v1 에서 뺌.
- 2026-10-05: B0 오프라인 상한을 TAXI 에서 실행(job 80130) → [[output-cst-taxi-offline-baselines-20261005]]. SeamlessStreaming·COMET 모델은 사용자 업로드 대기.
