---
type: task
status: open
owner: tskim
due: 2026-12-24
priority: p1
created: 2026-09-03
updated: 2026-09-03
summary: 한국어 turn-taking 어노테이션 프로토콜 설계 — 샘플링, IAA, 배포 형식
sources:
  - [[output-streaming-vap-research-plan]]
---

# 한국어 벤치마크 설계

## 배경

→ [[decision-korean-benchmark-release-scope]], [[korean-turn-taking-cues]]

초안은 "10~30시간을 3인이 라벨" 이라고만 했다. 프레임 단위 라벨링은 비현실적이므로
**이벤트 후보 지점을 샘플링해 라벨** 하는 방식으로 설계해야 한다.

## 완료 조건

- [ ] 후보 지점 정의 — IPU(inter-pausal unit) 경계에서 샘플링, **3,000~5,000 지점**
- [ ] 어노테이션 가이드라인 작성 — [[source-turnbench]] 프로토콜 준용
- [ ] **어미별 층화 샘플링** (-요 / -습니다 / -는데 / -고 / -면 / -서) →
      [[korean-turn-taking-cues]] 가설을 데이터로 검증 가능하게
- [ ] 3인 어노테이터 + 2/3 consensus, **Fleiss κ 보고** (TurnBench 는 0.78)
- [ ] 파일럿 100지점으로 가이드라인 검증 후 본작업
- [ ] **배포 형식**: 오디오 없이 `AI Hub 파일 ID + 시간 오프셋 + 라벨` 로 저장
- [ ] 어노테이션 비용·기간 산정

## 진행 기록

- 2026-09-03: 생성.
